import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import * as packService from "../../services/capabilities";
import * as editorService from "../../services/editor";
import {
  enableMaskPixels,
  installEditorPanelHooks,
  loadBaseImage,
  openPeopleWithTwoToRemove,
  paintOnCanvas,
  renderPanel,
} from "./editorPanelTestKit";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getDevices: vi.fn() };
});
vi.mock("../../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/generation")>();
  return {
    ...actual,
    uploadGenerationInitImage: vi.fn(),
    createGenerationJob: vi.fn(),
    getGenerationJob: vi.fn(),
    fetchGenerationCapabilities: vi.fn(),
    cancelGenerationJob: vi.fn(),
  };
});
vi.mock("../../services/capabilities", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../services/capabilities")>()),
  provisionPack: vi.fn(),
  getProvisionStatus: vi.fn(),
}));
vi.mock("../../services/editor", () => ({
  fetchEditorCapabilities: vi.fn(),
  detectPeople: vi.fn(),
  segmentEditorObject: vi.fn(),
  segmentEditorBox: vi.fn(),
  healEditorImage: vi.fn(),
  fetchInitImage: vi.fn(),
}));

installEditorPanelHooks();

const PEOPLE: editorService.EditorPerson[] = [
  { id: 1, box: [10, 10, 120, 400], score: 0.9, role: "subject", face: [30, 30, 60, 70], anchor: [60, 200] },
  { id: 2, box: [200, 30, 300, 420], score: 0.8, role: "background", face: null, anchor: [250, 220] },
  { id: 3, box: [320, 30, 400, 420], score: 0.7, role: "background", face: null, anchor: [360, 220] },
];

describe("EditorPanel: people", () => {
  it("detects people in the background and switches from Choose to brush without painting", async () => {
    vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: true, people: true });
    vi.mocked(editorService.detectPeople).mockResolvedValue([
      { id: 1, box: [10, 10, 120, 400], score: 0.9, role: "subject", face: [30, 30, 60, 70], anchor: [60, 200] },
      { id: 2, box: [200, 30, 300, 420], score: 0.8, role: "background", face: null, anchor: [250, 220] },
    ]);
    renderPanel();
    await loadBaseImage();
    await waitFor(() => expect(editorService.detectPeople).toHaveBeenCalledWith("aabbcc"));

    fireEvent.click(await screen.findByRole("radio", { name: "People" }));
    expect(await screen.findByText("People in the photo")).toBeInTheDocument();
    const row = screen.getByRole("listitem", { name: "Person 2" });
    expect(within(row).getByRole("radio", { name: /Remove/ })).toHaveAttribute("aria-checked", "true");

    fireEvent.click(screen.getByTestId("person-outline-2"));
    expect(within(row).getByRole("radio", { name: /Keep/ })).toHaveAttribute("aria-checked", "true");
    expect(editorService.segmentEditorObject).not.toHaveBeenCalled();

    fireEvent.click(within(row).getByRole("radio", { name: /Remove/ }));
    enableMaskPixels();
    vi.mocked(editorService.segmentEditorBox).mockResolvedValue(new Blob(["png"], { type: "image/png" }));
    fireEvent.click(screen.getByRole("button", { name: en["editor.people.accept.one"] }));
    await waitFor(() => expect(screen.getByRole("radio", { name: "Brush" })).toHaveAttribute("aria-checked", "true"));
    expect(screen.getByLabelText("Editor settings")).toHaveAttribute("data-view", "action");
  });

  it("adds the chosen people as one batch, switches to Brush and shows the summary", async () => {
    const choices = await openPeopleWithTwoToRemove();
    vi.mocked(editorService.segmentEditorBox).mockResolvedValue(new Blob(["png"], { type: "image/png" }));

    fireEvent.click(screen.getByRole("button", { name: "Add 2 silhouettes" }));

    expect(await screen.findByText("2 silhouettes in the erase")).toBeInTheDocument();
    expect(editorService.segmentEditorBox).toHaveBeenCalledTimes(3);
    expect(vi.mocked(editorService.segmentEditorBox).mock.calls.map(([, box]) => box)).toEqual(choices.boxes);
    expect(screen.getByRole("radio", { name: "Brush" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText("Person 2")).toBeInTheDocument();
    expect(screen.getByText("Person 3")).toBeInTheDocument();
    expect(screen.getByText(en["editor.people.polishKeys"])).toBeInTheDocument();
    expect(screen.getByText(en["editor.people.polishHint"])).toBeInTheDocument();
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled();
  });

  it("shows progress while segmenting and a notice for people it could not cut out", async () => {
    await openPeopleWithTwoToRemove();
    let finishLast: (blob: Blob) => void = () => undefined;
    vi.mocked(editorService.segmentEditorBox)
      .mockResolvedValueOnce(new Blob(["png"], { type: "image/png" }))
      .mockRejectedValueOnce(new Error("timeout"))
      .mockReturnValueOnce(new Promise((resolve) => { finishLast = resolve; }));

    fireEvent.click(screen.getByRole("button", { name: "Add 2 silhouettes" }));

    expect(await screen.findByRole("button", { name: "Segmenting 3 of 3…" })).toBeDisabled();
    finishLast(new Blob(["png"], { type: "image/png" }));
    expect(await screen.findByText(en["editor.people.batchTitle.one"])).toBeInTheDocument();
    expect(screen.getByText(en["editor.people.failed.one"])).toHaveAttribute("aria-live", "polite");
  });

  it("clears the notice for people it could not cut out when its batch is undone", async () => {
    await openPeopleWithTwoToRemove();
    vi.mocked(editorService.segmentEditorBox)
      .mockResolvedValueOnce(new Blob(["png"], { type: "image/png" }))
      .mockRejectedValueOnce(new Error("timeout"))
      .mockResolvedValueOnce(new Blob(["png"], { type: "image/png" }));
    fireEvent.click(screen.getByRole("button", { name: "Add 2 silhouettes" }));
    const notice = en["editor.people.failed.one"];
    expect(await screen.findByText(notice)).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });

    await waitFor(() => expect(screen.queryByText(notice)).not.toBeInTheDocument());
  });

  it("undoes the whole batch in one step", async () => {
    await openPeopleWithTwoToRemove();
    vi.mocked(editorService.segmentEditorBox).mockResolvedValue(new Blob(["png"], { type: "image/png" }));
    fireEvent.click(screen.getByRole("button", { name: "Add 2 silhouettes" }));
    await screen.findByText("2 silhouettes in the erase");

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });

    await waitFor(() => expect(screen.queryByText("2 silhouettes in the erase")).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: en["editor.undo"] })).toBeDisabled();
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeDisabled();
  });

  it("goes back to choosing without the batch and with the same choices", async () => {
    await openPeopleWithTwoToRemove();
    vi.mocked(editorService.segmentEditorBox).mockResolvedValue(new Blob(["png"], { type: "image/png" }));
    fireEvent.click(screen.getByRole("button", { name: "Add 2 silhouettes" }));
    await screen.findByText("2 silhouettes in the erase");

    fireEvent.click(screen.getByRole("button", { name: en["editor.people.backToChoosing"] }));

    expect(await screen.findByText("People in the photo")).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "People" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("button", { name: en["editor.undo"] })).toBeDisabled();
    const row = screen.getByRole("listitem", { name: "Person 1" });
    expect(within(row).getByRole("radio", { name: /Keep/ })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByRole("button", { name: "Add 2 silhouettes" })).toBeEnabled();
  });

  it("back to choosing removes only the batch and keeps the strokes painted after it", async () => {
    await openPeopleWithTwoToRemove();
    vi.mocked(editorService.segmentEditorBox).mockResolvedValue(new Blob(["png"], { type: "image/png" }));
    fireEvent.click(screen.getByRole("button", { name: "Add 2 silhouettes" }));
    await screen.findByText("2 silhouettes in the erase");
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.brush"] }));
    paintOnCanvas();

    fireEvent.click(screen.getByRole("button", { name: en["editor.people.backToChoosing"] }));
    await screen.findByText("People in the photo");
    expect(screen.queryByText("2 silhouettes in the erase")).not.toBeInTheDocument();

    // Deshacer ahora quita el trazo, no el lote: si el lote siguiera, Quitar seguiría activo.
    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeDisabled();
  });

  it("undoing the batch and then the stroke before it clears both", async () => {
    vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: true, people: true });
    vi.mocked(editorService.detectPeople).mockResolvedValue(PEOPLE);
    enableMaskPixels();
    renderPanel();
    await loadBaseImage();
    paintOnCanvas();
    fireEvent.click(await screen.findByRole("radio", { name: "People" }));
    await screen.findByRole("button", { name: "Add 2 silhouettes" });
    vi.mocked(editorService.segmentEditorBox).mockResolvedValue(new Blob(["png"], { type: "image/png" }));
    fireEvent.click(screen.getByRole("button", { name: "Add 2 silhouettes" }));
    await screen.findByText("2 silhouettes in the erase");

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    await waitFor(() => expect(screen.queryByText("2 silhouettes in the erase")).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled();

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeDisabled();
  });

  it("drops a batch that finishes after the image was closed", async () => {
    await openPeopleWithTwoToRemove();
    let finish!: (blob: Blob) => void;
    vi.mocked(editorService.segmentEditorBox).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
    fireEvent.click(screen.getByRole("button", { name: "Add 2 silhouettes" }));
    await waitFor(() => expect(editorService.segmentEditorBox).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole("button", { name: en["editor.newImage"] }));
    await act(async () => finish(new Blob(["png"], { type: "image/png" })));
    await loadBaseImage();

    expect(screen.queryByText("2 silhouettes in the erase")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: en["editor.undo"] })).toBeDisabled();
  });

  it("offers the person detector when People is unavailable and enables the tool once it lands", async () => {
    // El detector "aparece" recién cuando termina la descarga: así el refetch de
    // capacidades que dispara el montaje no lo habilita antes de tiempo.
    let detectorInstalled = false;
    vi.mocked(editorService.fetchEditorCapabilities).mockImplementation(async () => ({
      tapSelect: true,
      people: detectorInstalled,
    }));
    vi.mocked(editorService.detectPeople).mockResolvedValue([
      { id: 1, box: [10, 10, 120, 400], score: 0.9, role: "subject", face: null, anchor: [60, 200] },
    ]);
    vi.mocked(packService.provisionPack).mockResolvedValue({ jobId: "job-1" } as never);
    vi.mocked(packService.getProvisionStatus).mockImplementation(async () => {
      detectorInstalled = true;
      return { jobId: "job-1", status: "done" } as never;
    });
    renderPanel();
    await loadBaseImage();

    fireEvent.click(await screen.findByRole("radio", { name: "People" }));
    expect(screen.getByRole("radio", { name: "People" })).toBeEnabled();
    expect(await screen.findByText(en["editor.people.missingDetector"])).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    expect(await screen.findByRole("listitem", { name: "Person 1" })).toBeInTheDocument();
    expect(vi.mocked(packService.provisionPack).mock.calls[0]).toEqual(["person-detector", undefined, false]);
  });
});
