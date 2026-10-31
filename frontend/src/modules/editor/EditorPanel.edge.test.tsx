import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import type { GenerationCapabilities } from "../../lib/apiTypes";
import * as generationService from "../../services/generation";
import {
  BASE_IMAGE,
  CAPABILITIES,
  installEditorPanelHooks,
  loadBaseImage,
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
vi.mock("../../services/editor", () => ({
  fetchEditorCapabilities: vi.fn(),
  detectPeople: vi.fn(),
  segmentEditorObject: vi.fn(),
  segmentEditorBox: vi.fn(),
  healEditorImage: vi.fn(),
  fetchInitImage: vi.fn(),
}));

installEditorPanelHooks();

describe("EditorPanel: ajustar borde", () => {
  const ERASER_CAPABILITIES: GenerationCapabilities = {
    ...CAPABILITIES,
    models: [
      { id: "eraser-auto", name: "Auto", status: "installed", eraseOnly: true, supportsInpaint: true },
      CAPABILITIES.models[0],
    ],
  };

  beforeEach(() => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue(ERASER_CAPABILITIES);
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ id: "edge-job", status: "queued" } as never);
  });

  async function openEraseWithAuto() {
    renderPanel();
    await loadBaseImage();
    await waitFor(() => expect(screen.getByLabelText(en["editor.model.label"])).toHaveValue("eraser-auto"));
  }

  async function submitErase() {
    paintOnCanvas();
    fireEvent.click(screen.getByRole("button", { name: en["editor.submit.erase"] }));
    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    return vi.mocked(generationService.createGenerationJob).mock.calls[0][0];
  }

  it("moving Expand to 12 makes the erase job carry maskGrowPx 12", async () => {
    await openEraseWithAuto();
    fireEvent.change(screen.getByLabelText(en["editor.edge.expand"]), { target: { value: "12" } });

    expect(await submitErase()).toEqual(expect.objectContaining({ maskGrowPx: 12 }));
  });

  it("moving Soften to 5 makes the erase job carry maskFeatherPx 5", async () => {
    await openEraseWithAuto();
    fireEvent.change(screen.getByLabelText(en["editor.edge.soften"]), { target: { value: "5" } });

    expect(await submitErase()).toEqual(expect.objectContaining({ maskFeatherPx: 5 }));
  });

  it("announces each slider value with its unit", async () => {
    await openEraseWithAuto();
    fireEvent.change(screen.getByLabelText(en["editor.edge.expand"]), { target: { value: "12" } });
    fireEvent.change(screen.getByLabelText(en["editor.edge.soften"]), { target: { value: "5" } });

    expect(screen.getByLabelText(en["editor.edge.expand"])).toHaveAttribute("aria-valuetext", "12 px");
    expect(screen.getByLabelText(en["editor.edge.soften"])).toHaveAttribute("aria-valuetext", "5 px");
  });

  it("resets Expand and Soften to 0 when another image is loaded", async () => {
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(BASE_IMAGE)
      .mockResolvedValueOnce({ ...BASE_IMAGE, originalFilename: "new.png", initImageToken: "new-base" });
    await openEraseWithAuto();
    fireEvent.change(screen.getByLabelText(en["editor.edge.expand"]), { target: { value: "12" } });
    fireEvent.change(screen.getByLabelText(en["editor.edge.soften"]), { target: { value: "5" } });

    fireEvent.change(screen.getByLabelText(en["editor.changeImage"]), {
      target: { files: [new File(["new"], "new.png", { type: "image/png" })] },
    });
    await screen.findByAltText("new.png");

    expect((screen.getByLabelText(en["editor.edge.expand"]) as HTMLInputElement).value).toBe("0");
    expect((screen.getByLabelText(en["editor.edge.soften"]) as HTMLInputElement).value).toBe("0");
  });

  it("replace mode does not send the edge values", async () => {
    await openEraseWithAuto();
    fireEvent.change(screen.getByLabelText(en["editor.edge.expand"]), { target: { value: "12" } });

    fireEvent.click(screen.getByRole("radio", { name: en["editor.mode.replace"] }));
    expect(screen.queryByLabelText(en["editor.edge.expand"])).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), { target: { value: "gen--sdxl" } });
    fireEvent.change(screen.getByLabelText(en["editor.prompt.label"]), { target: { value: "a red vase" } });
    paintOnCanvas();
    fireEvent.click(screen.getByRole("button", { name: en["editor.submit.replace"] }));

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    const params = vi.mocked(generationService.createGenerationJob).mock.calls[0][0];
    expect(params).not.toHaveProperty("maskGrowPx");
    expect(params).not.toHaveProperty("maskFeatherPx");
  });
});
