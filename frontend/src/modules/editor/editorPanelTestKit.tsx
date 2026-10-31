import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, afterEach, expect, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { en } from "../../i18n/en";
import type { GenerationCapabilities, GenerationModelSummary, InitImageResponse } from "../../lib/apiTypes";
import * as api from "../../lib/api";
import * as editorService from "../../services/editor";
import * as generationService from "../../services/generation";
import { EditorPanel } from "./EditorPanel";

export const CAPABILITIES: GenerationCapabilities = {
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

// Con LaMa instalado el Corrector y el borrado automático están disponibles.
export const LAMA_MODEL: GenerationModelSummary = { id: "eraser-lama", name: "LaMa", status: "installed", eraseOnly: true };

export const BASE_IMAGE: InitImageResponse = {
  initImageToken: "aabbcc",
  originalFilename: "foto.png",
  width: 640,
  height: 480,
};

export function fakeContext() {
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
    setTransform: vi.fn(),
    lineWidth: 0,
    lineCap: "",
    lineJoin: "",
    strokeStyle: "",
    fillStyle: "",
    globalCompositeOperation: "source-over",
  };
}

// Setup común de la suite del Editor: cada archivo de test lo llama una vez al tope.
export function installEditorPanelHooks() {
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
    vi.mocked(editorService.detectPeople).mockResolvedValue([]);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.mocked(generationService.uploadGenerationInitImage).mockReset();
    vi.mocked(generationService.createGenerationJob).mockReset();
    vi.mocked(generationService.fetchGenerationCapabilities).mockReset();
    vi.mocked(editorService.fetchEditorCapabilities).mockReset();
    vi.mocked(editorService.detectPeople).mockReset();
    vi.mocked(editorService.segmentEditorObject).mockReset();
    vi.mocked(editorService.segmentEditorBox).mockReset();
    vi.mocked(editorService.healEditorImage).mockReset();
    vi.mocked(editorService.fetchInitImage).mockReset();
  });
}

export function renderPanel() {
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

export async function loadBaseImage() {
  fireEvent.change(screen.getByLabelText(en["editor.dropzone.inputLabel"]), {
    target: { files: [new File(["img"], "foto.png", { type: "image/png" })] },
  });
  await screen.findByTestId("editor-mask-canvas");
}

// jsdom no trae PointerEvent: sin esto fireEvent pierde altKey (Alt+clic de Clonar).
export class TestPointerEvent extends MouseEvent {
  readonly pointerId: number;

  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 0;
  }
}

export function paintOnCanvas() {
  const canvas = screen.getByTestId("editor-mask-canvas");
  fireEvent.pointerDown(canvas, { clientX: 10, clientY: 10, pointerId: 1 });
  fireEvent.pointerMove(canvas, { clientX: 60, clientY: 60, pointerId: 1 });
  fireEvent.pointerUp(canvas, { pointerId: 1 });
}

export function enableMaskPixels() {
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

export async function openPeopleWithTwoToRemove() {
  vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: true, people: true });
  const people: editorService.EditorPerson[] = [
    { id: 1, box: [10, 10, 120, 400], score: 0.9, role: "subject", face: [30, 30, 60, 70], anchor: [60, 200] },
    { id: 2, box: [200, 30, 300, 420], score: 0.8, role: "background", face: null, anchor: [250, 220] },
    { id: 3, box: [320, 30, 400, 420], score: 0.7, role: "background", face: null, anchor: [360, 220] },
  ];
  vi.mocked(editorService.detectPeople).mockResolvedValue(people);
  enableMaskPixels();
  renderPanel();
  await loadBaseImage();
  fireEvent.click(await screen.findByRole("radio", { name: "People" }));
  await screen.findByRole("button", { name: "Add 2 silhouettes" });
  return { boxes: people.map((person) => person.box) };
}

export async function protectObject() {
  fireEvent.click(await screen.findByRole("radio", { name: en["editor.tool.protect"] }));
  fireEvent.click(screen.getByTestId("editor-mask-canvas"), { clientX: 5, clientY: 5 });
  await waitFor(() => expect(screen.getByRole("button", { name: en["editor.protect.clear"] })).toBeEnabled());
}

// Con LaMa instalado: las capabilities de generación incluyen el borrado LaMa listo.
export function withLamaCapabilities() {
  vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue({
    ...CAPABILITIES,
    models: [...CAPABILITIES.models, LAMA_MODEL],
  });
}
