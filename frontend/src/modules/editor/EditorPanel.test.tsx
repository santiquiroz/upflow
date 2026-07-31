import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { en } from "../../i18n/en";
import type { GenerationCapabilities, InitImageResponse } from "../../lib/apiTypes";
import * as api from "../../lib/api";
import * as editorService from "../../services/editor";
import * as generationService from "../../services/generation";
import { EditorPanel } from "./EditorPanel";

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
  segmentEditorObject: vi.fn(),
}));

const CAPABILITIES: GenerationCapabilities = {
  available: true,
  reason: null,
  devices: ["cpu", "dml:0"],
  cpuOnly: false,
  models: [
    { id: "gen--sdxl", name: "epicrealism", status: "installed", supportsInpaint: true },
    { id: "gen--lcm", name: "lcm-fast", status: "installed", supportsInpaint: false },
    { id: "gen--busy", name: "converting-one", status: "converting", supportsInpaint: true },
  ],
};

const BASE_IMAGE: InitImageResponse = {
  initImageToken: "aabbcc",
  originalFilename: "foto.png",
  width: 640,
  height: 480,
};

function fakeContext() {
  return {
    clearRect: vi.fn(),
    drawImage: vi.fn(),
    getImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4) })),
    putImageData: vi.fn(),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    lineWidth: 0,
    lineCap: "",
    lineJoin: "",
    strokeStyle: "",
    fillStyle: "",
    globalCompositeOperation: "source-over",
  };
}

beforeEach(() => {
  HTMLCanvasElement.prototype.getContext = vi.fn(() => fakeContext()) as never;
  HTMLCanvasElement.prototype.toBlob = vi.fn((callback: BlobCallback) =>
    callback(new Blob(["mask"], { type: "image/png" })),
  );
  (URL as { createObjectURL?: unknown }).createObjectURL = vi.fn(() => "blob:fake");
  (URL as { revokeObjectURL?: unknown }).revokeObjectURL = vi.fn();

  vi.mocked(api.getDevices).mockResolvedValue({
    devices: [
      { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" },
      { id: "dml:0", kind: "gpu", name: "RX 7800 XT", backend: "directml" },
    ],
    defaultDeviceId: "dml:0",
  });
  vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue(CAPABILITIES);
  vi.mocked(generationService.uploadGenerationInitImage).mockResolvedValue(BASE_IMAGE);
  vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: true });
});

afterEach(() => {
  vi.mocked(generationService.uploadGenerationInitImage).mockReset();
  vi.mocked(generationService.createGenerationJob).mockReset();
  vi.mocked(generationService.fetchGenerationCapabilities).mockReset();
  vi.mocked(editorService.fetchEditorCapabilities).mockReset();
  vi.mocked(editorService.segmentEditorObject).mockReset();
});

function renderPanel() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <LocaleProvider initialLocale="en">
          <MemoryRouter>{children}</MemoryRouter>
        </LocaleProvider>
      </QueryClientProvider>
    );
  }
  return render(<EditorPanel />, { wrapper: Wrapper });
}

async function loadBaseImage() {
  fireEvent.change(screen.getByLabelText(en["editor.dropzone.inputLabel"]), {
    target: { files: [new File(["img"], "foto.png", { type: "image/png" })] },
  });
  await screen.findByTestId("editor-mask-canvas");
}

function paintOnCanvas() {
  const canvas = screen.getByTestId("editor-mask-canvas");
  fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, pointerId: 1 });
  fireEvent.pointerMove(canvas, { clientX: 60, clientY: 60, pointerId: 1 });
  fireEvent.pointerUp(canvas, { pointerId: 1 });
}

describe("EditorPanel", () => {
  it("only offers installed models with real inpaint support", async () => {
    renderPanel();
    await loadBaseImage();

    const select = screen.getByLabelText(en["editor.model.label"]);
    const labels = Array.from(select.querySelectorAll("option")).map((option) => option.textContent);
    expect(labels).toContain("epicrealism");
    expect(labels).not.toContain("lcm-fast");
    expect(labels).not.toContain("converting-one");
  });

  it("keeps the submit button disabled until there is a mask", async () => {
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });

    const submit = screen.getByRole("button", { name: en["editor.submit.erase"] });
    expect(submit).toBeDisabled();

    paintOnCanvas();
    await waitFor(() => expect(submit).toBeEnabled());
  });

  it("submits an inpaint job with base and mask tokens", async () => {
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(BASE_IMAGE)
      .mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "mask-token" });
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({
      id: "job-1",
      status: "queued",
    } as never);
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });
    paintOnCanvas();

    fireEvent.click(await screen.findByRole("button", { name: en["editor.submit.erase"] }));

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    const params = vi.mocked(generationService.createGenerationJob).mock.calls[0][0];
    expect(params.initImageToken).toBe("aabbcc");
    expect(params.maskImageToken).toBe("mask-token");
    expect(params.width % 64).toBe(0);
    expect(params.height % 64).toBe(0);
    expect(params.prompt.length).toBeGreaterThan(0);
  });

  it("replace mode requires a prompt before submitting", async () => {
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });
    paintOnCanvas();

    fireEvent.click(screen.getByRole("radio", { name: en["editor.mode.replace"] }));
    const submit = screen.getByRole("button", { name: en["editor.submit.replace"] });
    expect(submit).toBeDisabled();

    fireEvent.change(screen.getByLabelText(en["editor.prompt.label"]), {
      target: { value: "a wooden bench" },
    });
    await waitFor(() => expect(submit).toBeEnabled());
  });

  it("tap tool asks the backend for the object mask", async () => {
    vi.mocked(editorService.segmentEditorObject).mockResolvedValue(
      new Blob(["png"], { type: "image/png" }),
    );
    renderPanel();
    await loadBaseImage();

    fireEvent.click(await screen.findByRole("radio", { name: en["editor.tool.tap"] }));
    fireEvent.click(screen.getByTestId("editor-mask-canvas"), { clientX: 5, clientY: 5 });

    await waitFor(() => expect(editorService.segmentEditorObject).toHaveBeenCalledOnce());
    expect(vi.mocked(editorService.segmentEditorObject).mock.calls[0][0]).toBe("aabbcc");
  });

  it("disables the tap tool when the segmenter is not installed", async () => {
    vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: false });
    renderPanel();
    await loadBaseImage();

    await waitFor(() =>
      expect(screen.getByRole("radio", { name: en["editor.tool.tap"] })).toBeDisabled(),
    );
  });
});
