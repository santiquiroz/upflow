import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import * as editorService from "../../services/editor";
import * as generationService from "../../services/generation";
import * as zones from "./zones";
import {
  BASE_IMAGE,
  CAPABILITIES,
  enableMaskPixels,
  installEditorPanelHooks,
  loadBaseImage,
  paintOnCanvas,
  protectObject,
  renderPanel,
} from "./editorPanelTestKit";

vi.mock("./zones", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./zones")>();
  return { ...actual, clipMaskToBox: vi.fn(actual.clipMaskToBox) };
});
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
vi.mock("../../services/editor", () => ({
  fetchEditorCapabilities: vi.fn(),
  detectPeople: vi.fn(),
  segmentEditorObject: vi.fn(),
  segmentEditorBox: vi.fn(),
  healEditorImage: vi.fn(),
  fetchInitImage: vi.fn(),
}));

installEditorPanelHooks();

describe("EditorPanel review zones", () => {
  const ZONE_BOX: [number, number, number, number] = [0, 0, 100, 100];
  const TOKENS: Record<string, string> = {
    "mask.png": "mask-token",
    "redo-base.png": "redo-base",
    "redo-mask.png": "redo-mask",
  };

  function completedJob(id: string, downloadUrl: string, zone: Partial<EraseZoneFixture>) {
    return {
      id, status: "completed", prompt: "background", negativePrompt: null, modelId: "eraser-auto", steps: 30,
      guidance: 7, width: 640, height: 480, seed: null, device: "dml:0", autoUpscale: false,
      createdAt: "2026-10-30T00:00:00Z", startedAt: null, finishedAt: null, progressPct: 100, stages: [],
      error: null, ownerId: null, downloadUrl,
      eraseZones: [{ index: 0, box: ZONE_BOX, engine: "lama+osor", guard: null, residualPerson: false, ...zone }],
    } as const;
  }

  type EraseZoneFixture = { engine: string; guard: "face" | "person" | null; residualPerson: boolean };

  beforeEach(() => {
    enableMaskPixels();
    vi.stubGlobal("fetch", vi.fn(async () => ({ blob: async () => new Blob(["result"], { type: "image/png" }) })));
    vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue({
      ...CAPABILITIES,
      models: [
        { id: "eraser-auto", name: "Auto", status: "installed", eraseOnly: true, supportsInpaint: true },
        { ...CAPABILITIES.models[0], inpaintOnly: true },
      ],
    });
    vi.mocked(generationService.uploadGenerationInitImage).mockImplementation(async (file: File) => ({
      ...BASE_IMAGE, initImageToken: TOKENS[file.name] ?? BASE_IMAGE.initImageToken,
    }));
    vi.mocked(generationService.createGenerationJob)
      .mockResolvedValueOnce({ id: "erase-job", status: "queued" } as never)
      .mockResolvedValueOnce({ id: "redo-job", status: "queued" } as never);
    vi.mocked(generationService.getGenerationJob).mockImplementation(async (id: string) => (id === "erase-job"
      ? completedJob("erase-job", "/erase.png", { guard: "person", residualPerson: true })
      : completedJob("redo-job", "/redo.png", { engine: "lama" })) as never);
  });

  afterEach(() => {
    vi.mocked(generationService.getGenerationJob).mockReset();
    vi.mocked(zones.clipMaskToBox).mockClear();
  });

  async function eraseAndReview() {
    renderPanel();
    await loadBaseImage();
    await waitFor(() => expect(screen.getByLabelText(en["editor.model.label"])).toHaveValue("eraser-auto"));
    paintOnCanvas();
    fireEvent.click(await screen.findByRole("button", { name: en["editor.submit.erase"] }));
    return screen.findByRole("listitem", { name: "Zone 1" });
  }

  it("shows the zones of the erase over the result", async () => {
    const row = await eraseAndReview();

    expect(screen.getByText("Review zones")).toBeInTheDocument();
    expect(within(row).getByText("Someone is still visible")).toBeInTheDocument();
    expect(within(row).getByText("LaMa + OSOR")).toBeInTheDocument();
    expect(screen.getByTestId("zone-marker-0")).toHaveTextContent("1");
    expect(screen.getByAltText("Erase result")).toHaveAttribute("src", "/erase.png");
  });

  it("redoes one zone with LaMa on the current result and a mask clipped to the zone", async () => {
    const row = await eraseAndReview();

    fireEvent.click(within(row).getByRole("button", { name: "Redo" }));
    fireEvent.click(within(row).getByRole("menuitem", { name: "LaMa" }));

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalledTimes(2));
    expect(vi.mocked(generationService.createGenerationJob).mock.calls[1][0]).toEqual(expect.objectContaining({
      modelId: "eraser-auto", eraseEngine: "lama", initImageToken: "redo-base", maskImageToken: "redo-mask",
    }));
    expect(fetch).toHaveBeenCalledWith("/erase.png");
    expect(zones.clipMaskToBox).toHaveBeenCalledWith(expect.anything(), ZONE_BOX);
    const files = vi.mocked(generationService.uploadGenerationInitImage).mock.calls.map(([file]) => file.name);
    expect(files).toEqual(expect.arrayContaining(["redo-base.png", "redo-mask.png"]));

    await waitFor(() => expect(within(row).getByText("No people")).toBeInTheDocument());
    expect(within(row).getByText("LaMa")).toBeInTheDocument();
    expect(screen.getByAltText("Erase result")).toHaveAttribute("src", "/redo.png");
  });

  function redoWith(row: HTMLElement, engine: string) {
    fireEvent.click(within(row).getByRole("button", { name: "Redo" }));
    fireEvent.click(within(row).getByRole("menuitem", { name: engine }));
  }

  it("bases a second redo of another zone on the first redo's result", async () => {
    const twoZones = completedJob("erase-job", "/erase.png", {});
    const jobs: Record<string, unknown> = {
      "erase-job": { ...twoZones, eraseZones: [...twoZones.eraseZones, { ...twoZones.eraseZones[0], index: 1, box: [200, 200, 300, 300] }] },
      "redo-job": completedJob("redo-job", "/redo.png", { engine: "lama" }),
      "redo-job-2": completedJob("redo-job-2", "/redo-2.png", { engine: "osor" }),
    };
    vi.mocked(generationService.getGenerationJob).mockImplementation(async (id: string) => jobs[id] as never);
    vi.mocked(generationService.createGenerationJob).mockResolvedValueOnce({ id: "redo-job-2", status: "queued" } as never);
    const first = await eraseAndReview();

    redoWith(first, "LaMa");
    await waitFor(() => expect(screen.getByAltText("Erase result")).toHaveAttribute("src", "/redo.png"));
    redoWith(screen.getByRole("listitem", { name: "Zone 2" }), "OSOR");

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalledTimes(3));
    expect(fetch).toHaveBeenLastCalledWith("/redo.png");
    await waitFor(() => expect(screen.getByAltText("Erase result")).toHaveAttribute("src", "/redo-2.png"));
    expect(within(first).getByText("LaMa")).toBeInTheDocument();
    expect(within(screen.getByRole("listitem", { name: "Zone 2" })).getByText("OSOR")).toBeInTheDocument();
  });

  it("keeps the previous result and shows the error on the row when a redo fails", async () => {
    vi.mocked(generationService.createGenerationJob)
      .mockReset()
      .mockResolvedValueOnce({ id: "erase-job", status: "queued" } as never)
      .mockRejectedValueOnce(new Error("redo failed"));
    const row = await eraseAndReview();

    redoWith(row, "LaMa");

    await waitFor(() => expect(within(row).getByRole("alert")).toHaveTextContent("redo failed"));
    expect(screen.getByAltText("Erase result")).toHaveAttribute("src", "/erase.png");
    expect(within(row).getByText("Someone is still visible")).toBeInTheDocument();
    expect(within(row).getByRole("button", { name: "Redo" })).toBeEnabled();
  });

  it("describes the background with the Replace model, the typed prompt and no eraser engine", async () => {
    const row = await eraseAndReview();

    redoWith(row, "Describe the background…");
    fireEvent.change(within(row).getByLabelText("What should be there"), { target: { value: "a brick wall" } });
    fireEvent.click(within(row).getByRole("button", { name: "Redo with this description" }));

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalledTimes(2));
    const params = vi.mocked(generationService.createGenerationJob).mock.calls[1][0];
    expect(params).toEqual(expect.objectContaining({
      modelId: "gen--sdxl", prompt: "a brick wall", initImageToken: "redo-base", maskImageToken: "redo-mask",
    }));
    expect(params).not.toHaveProperty("eraseEngine");
    expect(params).not.toHaveProperty("eraseStrategy");
  });

  it("locks the mask and retouch tools while the result is shown, with the reason", async () => {
    await eraseAndReview();

    expect(screen.getByRole("radio", { name: en["editor.tool.brush"] })).toBeDisabled();
    const restore = screen.getByRole("radio", { name: en["editor.tool.restore"] });
    expect(restore).toBeDisabled();
    expect(restore).toHaveAttribute("title", en["editor.review.keepEditingFirst"]);
  });

  it("keeping the result as the base drops the erase strokes and keeps the protection", async () => {
    vi.mocked(editorService.segmentEditorObject).mockResolvedValue(new Blob(["png"], { type: "image/png" }));
    renderPanel();
    await loadBaseImage();
    await waitFor(() => expect(screen.getByLabelText(en["editor.model.label"])).toHaveValue("eraser-auto"));
    await protectObject();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.brush"] }));
    paintOnCanvas();
    fireEvent.click(await screen.findByRole("button", { name: en["editor.submit.erase"] }));
    await screen.findByRole("listitem", { name: "Zone 1" });

    fireEvent.click(screen.getByRole("button", { name: en["editor.useAsBase"] }));

    await waitFor(() => expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeDisabled());
    expect(screen.getByRole("button", { name: en["editor.protect.clear"] })).toBeEnabled();
    // Adoptar el resultado es una sola acción de base: deshacerla deja la protección intacta.
    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    expect(screen.getByRole("button", { name: en["editor.protect.clear"] })).toBeEnabled();
  });
});
