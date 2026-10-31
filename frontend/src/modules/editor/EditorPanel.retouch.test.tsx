import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import type { InitImageResponse } from "../../lib/apiTypes";
import * as api from "../../lib/api";
import * as editorService from "../../services/editor";
import * as generationService from "../../services/generation";
import * as retouchApply from "./retouchApply";
import { MAX_BASE_VERSIONS } from "./useBaseVersions";
import {
  BASE_IMAGE,
  installEditorPanelHooks,
  loadBaseImage,
  paintOnCanvas,
  renderPanel,
  TestPointerEvent,
  withLamaCapabilities,
} from "./editorPanelTestKit";

vi.mock("./retouchApply", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./retouchApply")>();
  return { ...actual, applyRetouch: vi.fn(async () => new Blob(["retouched"], { type: "image/png" })) };
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

describe("EditorPanel: diseño de tres columnas y retoque", () => {
  it("a restore push keeps the mask history and Ctrl+Z undoes the base before the mask", async () => {
    renderPanel();
    await loadBaseImage();
    paintOnCanvas();
    await waitFor(() => expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled());

    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.restore"] }));
    const canvas = screen.getByTestId("editor-mask-canvas");
    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, pointerId: 2 });
    fireEvent.pointerMove(canvas, { clientX: 30, clientY: 30, pointerId: 2 });
    fireEvent.pointerUp(canvas, { pointerId: 2 });
    await waitFor(() => expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByText(en["editor.retouch.applying"])).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled();

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled();
    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeDisabled();
  });

  it("a base push does not refetch people detection", async () => {
    vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: true, people: true });
    vi.mocked(editorService.detectPeople).mockResolvedValue([]);
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(BASE_IMAGE)
      .mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "ddeeff" });
    renderPanel();
    await loadBaseImage();
    await waitFor(() => expect(editorService.detectPeople).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.restore"] }));
    const canvas = screen.getByTestId("editor-mask-canvas");
    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, pointerId: 3 });
    fireEvent.pointerUp(canvas, { pointerId: 3 });
    await waitFor(() => expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByText(en["editor.retouch.applying"])).not.toBeInTheDocument());
    expect(editorService.detectPeople).toHaveBeenCalledTimes(1);
    expect(editorService.detectPeople).toHaveBeenCalledWith("aabbcc");
  });
});

describe("EditorPanel: retoque y deshacer", () => {
  const applying = () => screen.queryByText(en["editor.retouch.applying"]);

  beforeEach(() => {
    vi.stubGlobal("PointerEvent", TestPointerEvent);
    vi.mocked(retouchApply.applyRetouch).mockClear();
  });

  async function restoreStroke(pointerId: number) {
    const canvas = screen.getByTestId("editor-mask-canvas");
    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, pointerId });
    fireEvent.pointerMove(canvas, { clientX: 30, clientY: 30, pointerId });
    fireEvent.pointerUp(canvas, { pointerId });
  }

  async function restoreAndWait(pointerId: number) {
    await restoreStroke(pointerId);
    await waitFor(() => expect(applying()).not.toBeInTheDocument());
  }

  function pressUndo() {
    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
  }

  it("twelve base pushes keep the log in step with the versions, so a mask under them is still undone", async () => {
    renderPanel();
    await loadBaseImage();
    paintOnCanvas();
    await waitFor(() => expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled());
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.restore"] }));

    for (let push = 0; push < MAX_BASE_VERSIONS + 2; push += 1) {
      await restoreAndWait(10 + push);
    }
    expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(MAX_BASE_VERSIONS + 3);

    for (let undo = 0; undo < MAX_BASE_VERSIONS - 1; undo += 1) {
      pressUndo();
    }
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled();
    pressUndo();
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeDisabled();
  });

  it("clone shows the crosshair at the source right after Alt+click", async () => {
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.clone"] }));
    const canvas = screen.getByTestId("editor-mask-canvas");
    fireEvent.pointerDown(canvas, { clientX: 40, clientY: 40, pointerId: 1, altKey: true });
    fireEvent.pointerUp(canvas, { pointerId: 1 });
    expect(screen.getByTestId("retouch-crosshair")).toBeInTheDocument();
  });

  it("strokes are ignored while an apply is in flight", async () => {
    let finish: (blob: Blob) => void = () => {};
    vi.mocked(retouchApply.applyRetouch).mockImplementationOnce(
      () => new Promise<Blob>((resolve) => { finish = resolve; }),
    );
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.restore"] }));
    await restoreStroke(1);
    expect(applying()).toBeInTheDocument();

    await restoreStroke(2);
    expect(retouchApply.applyRetouch).toHaveBeenCalledTimes(1);

    await act(async () => finish(new Blob(["retouched"], { type: "image/png" })));
    await waitFor(() => expect(applying()).not.toBeInTheDocument());
    expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(2);
  });

  it("submit stays disabled while a retouch is applying, even with a mask and a model", async () => {
    let finish: (blob: Blob) => void = () => {};
    vi.mocked(retouchApply.applyRetouch).mockImplementationOnce(
      () => new Promise<Blob>((resolve) => { finish = resolve; }),
    );
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), { target: { value: "gen--sdxl" } });
    paintOnCanvas();
    await waitFor(() => expect(screen.getByRole("button", { name: en["editor.submit.erase"] })).toBeEnabled());
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.restore"] }));

    await restoreStroke(2);

    expect(applying()).toBeInTheDocument();
    expect(screen.getByRole("button", { name: en["editor.submit.erase"] })).toBeDisabled();
    await act(async () => finish(new Blob(["retouched"], { type: "image/png" })));
    await waitFor(() => expect(applying()).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: en["editor.submit.erase"] })).toBeEnabled();
  });

  it("Ctrl+Z waits for an apply in flight, then undoes the base before the mask", async () => {
    let finish: (blob: Blob) => void = () => {};
    renderPanel();
    await loadBaseImage();
    paintOnCanvas();
    await waitFor(() => expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled());
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.restore"] }));
    vi.mocked(retouchApply.applyRetouch).mockImplementationOnce(
      () => new Promise<Blob>((resolve) => { finish = resolve; }),
    );
    await restoreStroke(2);

    pressUndo();
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled();

    await act(async () => finish(new Blob(["retouched"], { type: "image/png" })));
    await waitFor(() => expect(applying()).not.toBeInTheDocument());
    pressUndo();
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled();
    pressUndo();
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeDisabled();
  });

  it("an apply error keeps the previous base and shows the error", async () => {
    vi.mocked(retouchApply.applyRetouch).mockRejectedValueOnce(new Error("no se pudo aplicar"));
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.restore"] }));
    await restoreAndWait(1);

    expect(screen.getByText("no se pudo aplicar")).toBeInTheDocument();
    expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: en["editor.undo"] })).toBeDisabled();
  });
});

describe("EditorPanel: corrector puntual", () => {
  const HEALED: InitImageResponse = { initImageToken: "healed1", originalFilename: "foto.png", width: 640, height: 480 };
  const healing = () => screen.queryByText(en["editor.heal.healing"]);

  beforeEach(() => {
    vi.stubGlobal("PointerEvent", TestPointerEvent);
    withLamaCapabilities();
    vi.mocked(editorService.healEditorImage).mockResolvedValue(HEALED);
    vi.mocked(editorService.fetchInitImage).mockResolvedValue(new Blob(["healed"], { type: "image/png" }));
  });

  function healStroke(pointerId: number, to = { x: 30, y: 30 }) {
    const canvas = screen.getByTestId("editor-mask-canvas");
    fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, pointerId });
    fireEvent.pointerMove(canvas, { clientX: to.x, clientY: to.y, pointerId });
    fireEvent.pointerUp(canvas, { pointerId });
  }

  async function openHeal() {
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.heal"] }));
  }

  it("a small stroke heals the base and the next heal starts from the new image", async () => {
    await openHeal();
    vi.mocked(generationService.uploadGenerationInitImage).mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "mask01" });

    healStroke(1);

    await waitFor(() => expect(editorService.healEditorImage).toHaveBeenCalledWith("aabbcc", "mask01"));
    await waitFor(() => expect(healing()).not.toBeInTheDocument());
    expect(editorService.fetchInitImage).toHaveBeenCalledWith("healed1");
    healStroke(2);
    await waitFor(() => expect(editorService.healEditorImage).toHaveBeenCalledTimes(2));
    expect(editorService.healEditorImage).toHaveBeenLastCalledWith("healed1", "aabbcc");
  });

  it("the heal stroke stays out of the mask history", async () => {
    await openHeal();

    healStroke(1);

    await waitFor(() => expect(healing()).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeDisabled();
  });

  it("shows healing on the stage and ignores further strokes while it runs", async () => {
    let finish!: (value: InitImageResponse) => void;
    vi.mocked(editorService.healEditorImage).mockImplementationOnce(
      () => new Promise((resolve) => { finish = resolve; }),
    );
    await openHeal();

    healStroke(1);
    await screen.findByText(en["editor.heal.healing"]);
    await waitFor(() => expect(editorService.healEditorImage).toHaveBeenCalledTimes(1));
    healStroke(2);
    await act(async () => finish(HEALED));

    await waitFor(() => expect(healing()).not.toBeInTheDocument());
    expect(editorService.healEditorImage).toHaveBeenCalledTimes(1);
    expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(2);
  });

  function holdHeal(): () => Promise<void> {
    let finish!: (value: InitImageResponse) => void;
    vi.mocked(editorService.healEditorImage).mockImplementationOnce(
      () => new Promise((resolve) => { finish = resolve; }),
    );
    return async () => { await act(async () => finish(HEALED)); };
  }

  it("submit stays disabled while a heal runs", async () => {
    const finish = holdHeal();
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), { target: { value: "gen--sdxl" } });
    paintOnCanvas();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.heal"] }));

    healStroke(2);

    await screen.findByText(en["editor.heal.healing"]);
    expect(screen.getByRole("button", { name: en["editor.submit.erase"] })).toBeDisabled();
    await finish();
    await waitFor(() => expect(healing()).not.toBeInTheDocument());
  });

  it("ctrl+z while healing does nothing", async () => {
    const finish = holdHeal();
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.brush"] }));
    paintOnCanvas();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.heal"] }));

    healStroke(2);
    await screen.findByText(en["editor.heal.healing"]);
    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    await finish();

    await waitFor(() => expect(healing()).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: en["editor.clear"] })).toBeEnabled();
  });

  it("ignores restore and clone strokes while a heal runs", async () => {
    vi.mocked(retouchApply.applyRetouch).mockClear();
    const finish = holdHeal();
    await openHeal();

    healStroke(1);
    await screen.findByText(en["editor.heal.healing"]);
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.restore"] }));
    healStroke(2);
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.clone"] }));
    const canvas = screen.getByTestId("editor-mask-canvas");
    fireEvent.pointerDown(canvas, { clientX: 5, clientY: 5, pointerId: 3, altKey: true });
    fireEvent.pointerUp(canvas, { pointerId: 3 });
    expect(screen.getByText(en["editor.retouch.cloneNoSource"])).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.heal"] }));
    healStroke(4);
    await finish();

    await waitFor(() => expect(healing()).not.toBeInTheDocument());
    expect(retouchApply.applyRetouch).not.toHaveBeenCalled();
    expect(editorService.healEditorImage).toHaveBeenCalledTimes(1);
  });

  it("a large stroke shows the notice and does not call the service", async () => {
    vi.mocked(generationService.uploadGenerationInitImage).mockResolvedValueOnce({ ...BASE_IMAGE, width: 1200, height: 900 });
    const rect = { left: 0, top: 0, width: 1200, height: 900 } as DOMRect;
    const spy = vi.spyOn(HTMLCanvasElement.prototype, "getBoundingClientRect").mockReturnValue(rect);
    await openHeal();

    healStroke(1, { x: 800, y: 10 });

    expect(await screen.findByText(en["editor.heal.tooLarge"])).toBeInTheDocument();
    expect(editorService.healEditorImage).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  it("an endpoint error is shown and the base stays the same", async () => {
    vi.mocked(editorService.healEditorImage).mockRejectedValueOnce(new api.ApiError(503, "Falta el pack LaMa"));
    await openHeal();

    healStroke(1);

    expect(await screen.findByText("Falta el pack LaMa")).toBeInTheDocument();
    expect(editorService.fetchInitImage).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: en["editor.undo"] })).toBeDisabled();
  });

  it("undo after a heal restores the previous base", async () => {
    await openHeal();
    healStroke(1);
    await waitFor(() => expect(healing()).not.toBeInTheDocument());

    fireEvent.keyDown(window, { key: "z", ctrlKey: true });
    healStroke(2);

    await waitFor(() => expect(editorService.healEditorImage).toHaveBeenCalledTimes(2));
    expect(editorService.healEditorImage).toHaveBeenLastCalledWith("aabbcc", "aabbcc");
  });
});
