import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
  vi.unstubAllGlobals();
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

function enableMaskPixels() {
  const contexts = new WeakMap<HTMLCanvasElement, ReturnType<typeof fakeContext>>();
  HTMLCanvasElement.prototype.getContext = vi.fn(function (this: HTMLCanvasElement) {
    let context = contexts.get(this);
    if (context) return context;
    const pixels = new Uint8ClampedArray(4);
    context = fakeContext();
    context.clearRect.mockImplementation(() => { pixels.fill(0); });
    context.drawImage.mockImplementation((...args: unknown[]) => {
      const source = args[0];
      if (source instanceof HTMLCanvasElement) {
        pixels.set(source.getContext("2d")!.getImageData(0, 0, 1, 1).data);
      } else {
        pixels.fill(255);
      }
    });
    context.getImageData.mockImplementation(() => ({ data: pixels.slice() }));
    context.putImageData.mockImplementation((...args: unknown[]) => {
      pixels.set((args[0] as ImageData).data);
    });
    context.stroke.mockImplementation(() => { pixels.fill(255); });
    context.fill.mockImplementation(() => { pixels.fill(255); });
    contexts.set(this, context);
    return context;
  }) as never;
  vi.stubGlobal("Image", class {
    onload: (() => void) | null = null;
    set src(_value: string) { queueMicrotask(() => this.onload?.()); }
  });
}

async function protectObject() {
  fireEvent.click(await screen.findByRole("radio", { name: en["editor.tool.protect"] }));
  fireEvent.click(screen.getByTestId("editor-mask-canvas"), { clientX: 5, clientY: 5 });
  await waitFor(() => expect(screen.getByRole("button", { name: en["editor.protect.clear"] })).toBeEnabled());
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
    expect(params.protectMaskToken).toBeUndefined();
    expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(2);
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

describe("EditorPanel vista avanzada", () => {
  async function setupReadyToSubmit() {
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(BASE_IMAGE)
      .mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "mask-token" });
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({
      id: "job-2",
      status: "queued",
    } as never);
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });
    paintOnCanvas();
    fireEvent.click(screen.getByRole("button", { name: (n) => n.startsWith(en["editor.advanced.title"]) }));
  }

  it("advanced params flow into the job payload", async () => {
    await setupReadyToSubmit();

    fireEvent.change(screen.getByLabelText(en["editor.advanced.steps"]), { target: { value: "45" } });
    fireEvent.change(screen.getByLabelText(en["editor.advanced.negativePrompt"]), {
      target: { value: "blurry, low quality" },
    });
    fireEvent.change(screen.getByLabelText(en["editor.advanced.seed"]), { target: { value: "1234" } });

    fireEvent.click(screen.getByRole("button", { name: en["editor.submit.erase"] }));
    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());

    const params = vi.mocked(generationService.createGenerationJob).mock.calls[0][0];
    expect(params.steps).toBe(45);
    expect(params.negativePrompt).toBe("blurry, low quality");
    expect(params.seed).toBe(1234);
    expect(params.strength).toBe(1);
  });

  it("the erase fill prompt is editable and used as the job prompt", async () => {
    await setupReadyToSubmit();

    fireEvent.change(screen.getByLabelText(/Fill prompt/), {
      target: { value: "clean sand, beach" },
    });

    fireEvent.click(screen.getByRole("button", { name: en["editor.submit.erase"] }));
    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    expect(vi.mocked(generationService.createGenerationJob).mock.calls[0][0].prompt).toBe(
      "clean sand, beach",
    );
  });

  it("clearing the erase fill prompt blocks submission", async () => {
    await setupReadyToSubmit();

    fireEvent.change(screen.getByLabelText(/Fill prompt/), { target: { value: "  " } });

    expect(screen.getByRole("button", { name: en["editor.submit.erase"] })).toBeDisabled();
  });
});

describe("EditorPanel zoom y desplazamiento", () => {
  it("zooms in and out from the toolbar and can reset", async () => {
    renderPanel();
    await loadBaseImage();

    expect(screen.getByText("100%")).toBeInTheDocument();
    const reset = screen.getByRole("button", { name: en["editor.zoom.reset"] });
    expect(reset).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: en["editor.zoom.in"] }));
    expect(screen.getByText("150%")).toBeInTheDocument();
    expect(reset).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: en["editor.zoom.reset"] }));
    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("never zooms out past the fitted view", async () => {
    renderPanel();
    await loadBaseImage();

    fireEvent.click(screen.getByRole("button", { name: en["editor.zoom.out"] }));

    expect(screen.getByText("100%")).toBeInTheDocument();
  });

  it("the hand tool drags the view instead of painting", async () => {
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.pan"] }));

    const canvas = screen.getByTestId("editor-mask-canvas");
    fireEvent.pointerDown(canvas, { clientX: 100, clientY: 100, pointerId: 1, button: 0 });
    fireEvent.pointerMove(canvas, { clientX: 40, clientY: 40, pointerId: 1 });
    fireEvent.pointerUp(canvas, { pointerId: 1 });

    // Sin trazos: el submit sigue bloqueado por falta de máscara.
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });
    expect(screen.getByRole("button", { name: en["editor.submit.erase"] })).toBeDisabled();
  });

  it("resets the view when a new image is loaded", async () => {
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("button", { name: en["editor.zoom.in"] }));
    expect(screen.getByText("150%")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Image to edit|Use another image/), {
      target: { files: [new File(["img2"], "otra.png", { type: "image/png" })] },
    });

    await waitFor(() => expect(screen.getByText("100%")).toBeInTheDocument());
  });
});

describe("EditorPanel: la vista sobrevive a un cambio de tamaño", () => {
  it("re-acota el desplazamiento cuando el lienzo encoge", async () => {
    // Regresión (review adversarial): con la ventana grande se podía arrastrar
    // hasta un desplazamiento legal que, al achicar la ventana, dejaba la foto
    // entera fuera de la vista y el editor en blanco.
    const observers: Array<() => void> = [];
    class FakeResizeObserver {
      constructor(private readonly cb: () => void) {
        observers.push(() => this.cb());
      }
      observe() {}
      disconnect() {}
    }
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);

    let clientWidth = 1200;
    Object.defineProperty(HTMLDivElement.prototype, "clientWidth", {
      configurable: true,
      get() {
        return this.dataset?.testid === "editor-stage" ? clientWidth : 0;
      },
    });
    Object.defineProperty(HTMLDivElement.prototype, "clientHeight", {
      configurable: true,
      get() {
        return this.dataset?.testid === "editor-stage" ? 800 : 0;
      },
    });

    try {
      renderPanel();
      await loadBaseImage();

      fireEvent.click(screen.getByRole("button", { name: en["editor.zoom.in"] }));
      const stage = screen.getByTestId("editor-stage");
      const inner = stage.firstElementChild as HTMLElement;
      const panned = inner.style.transform;
      expect(panned).toContain("scale(1.5)");

      clientWidth = 300;
      act(() => {
        observers.forEach((fire) => fire());
      });

      const clamped = inner.style.transform;
      const panX = Number(clamped.match(/translate\((-?[\d.]+)px/)?.[1] ?? "0");
      // Con 300 px de ancho a 1.5x el mínimo legal es -150: nunca menos.
      expect(panX).toBeGreaterThanOrEqual(-150);
    } finally {
      vi.unstubAllGlobals();
      delete (HTMLDivElement.prototype as never as Record<string, unknown>).clientWidth;
      delete (HTMLDivElement.prototype as never as Record<string, unknown>).clientHeight;
    }
  });
});
