import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import type { InitImageResponse } from "../../lib/apiTypes";
import * as editorService from "../../services/editor";
import * as generationService from "../../services/generation";
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

describe("EditorPanel protection and automatic erase", () => {
  beforeEach(() => {
    enableMaskPixels();
    vi.mocked(editorService.segmentEditorObject).mockResolvedValue(new Blob(["png"], { type: "image/png" }));
  });

  it("protects a segmented object separately and supports undoing protection and its clear action", async () => {
    renderPanel();
    await loadBaseImage();

    await protectObject();

    expect(editorService.segmentEditorObject).toHaveBeenCalledWith("aabbcc", expect.any(Number), expect.any(Number));
    expect(screen.getByRole("button", { name: en["editor.submit.erase"] })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: en["editor.undo"] }));
    expect(screen.getByRole("button", { name: en["editor.protect.clear"] })).toBeDisabled();

    await protectObject();
    fireEvent.click(screen.getByRole("button", { name: en["editor.protect.clear"] }));
    expect(screen.getByRole("button", { name: en["editor.protect.clear"] })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: en["editor.undo"] }));
    expect(screen.getByRole("button", { name: en["editor.protect.clear"] })).toBeEnabled();
  });

  it("disables protection when object segmentation is unavailable", async () => {
    vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: false });
    renderPanel();
    await loadBaseImage();

    expect(screen.getByRole("radio", { name: en["editor.tool.protect"] })).toBeDisabled();
  });

  it("discards a protected segment that finishes after closing the base image", async () => {
    let finishSegment!: (blob: Blob) => void;
    vi.mocked(editorService.segmentEditorObject).mockReturnValue(new Promise((resolve) => { finishSegment = resolve; }));
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.protect"] }));
    fireEvent.click(screen.getByTestId("editor-mask-canvas"), { clientX: 5, clientY: 5 });
    await waitFor(() => expect(editorService.segmentEditorObject).toHaveBeenCalledOnce());

    fireEvent.click(screen.getByRole("button", { name: en["editor.newImage"] }));
    await act(async () => { finishSegment(new Blob(["png"], { type: "image/png" })); });
    await loadBaseImage();

    expect(screen.getByRole("button", { name: en["editor.protect.clear"] })).toBeDisabled();
    expect(screen.getByRole("button", { name: en["editor.undo"] })).toBeDisabled();
  });

  it("discards an edit upload that finishes after another base image is selected", async () => {
    let finishMaskUpload!: (image: InitImageResponse) => void;
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(BASE_IMAGE)
      .mockReturnValueOnce(new Promise((resolve) => { finishMaskUpload = resolve; }))
      .mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "new-base", originalFilename: "new.png" });
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), { target: { value: "gen--sdxl" } });
    paintOnCanvas();
    fireEvent.click(screen.getByRole("button", { name: en["editor.submit.erase"] }));
    await waitFor(() => expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(2));

    fireEvent.change(screen.getByLabelText(en["editor.changeImage"]), {
      target: { files: [new File(["new"], "new.png", { type: "image/png" })] },
    });
    await screen.findByAltText("new.png");
    await act(async () => { finishMaskUpload({ ...BASE_IMAGE, initImageToken: "stale-mask" }); });

    expect(generationService.createGenerationJob).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: en["editor.submit.erase"] })).toBeDisabled();
  });

  it("uploads protection as a separate PNG and submits its token with automatic erase", async () => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue({
      ...CAPABILITIES,
      models: [
        { ...CAPABILITIES.models[0], inpaintOnly: true },
        { id: "eraser-lama", name: "LaMa", status: "installed", eraseOnly: true },
      ],
    });
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(BASE_IMAGE)
      .mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "edit-token" })
      .mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "protect-token" });
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ id: "protected-job", status: "queued" } as never);
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), { target: { value: "gen--sdxl" } });
    expect(screen.getByText(en["editor.erase.autoHint"])).toBeInTheDocument();
    await protectObject();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.brush"] }));
    paintOnCanvas();

    fireEvent.click(screen.getByRole("button", { name: en["editor.submit.erase"] }));

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    expect(vi.mocked(generationService.createGenerationJob).mock.calls[0][0]).toEqual(expect.objectContaining({
      eraseStrategy: "auto", maskImageToken: "edit-token", protectMaskToken: "protect-token",
    }));
    const files = vi.mocked(generationService.uploadGenerationInitImage).mock.calls.map(([file]) => file);
    expect(files.map((file) => file.name)).toEqual(["foto.png", "mask.png", "protect-mask.png"]);
    expect(files[2].type).toBe("image/png");
  });

  it("hides automatic erase when switching to replacement", async () => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue({
      ...CAPABILITIES,
      models: [
        { ...CAPABILITIES.models[0], inpaintOnly: true },
        { id: "eraser-lama", name: "LaMa", status: "installed", eraseOnly: true },
      ],
    });
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), { target: { value: "gen--sdxl" } });
    expect(screen.getByText(en["editor.erase.autoHint"])).toBeInTheDocument();

    fireEvent.click(screen.getByRole("radio", { name: en["editor.mode.replace"] }));

    expect(screen.queryByText(en["editor.erase.autoHint"])).not.toBeInTheDocument();
  });
});
