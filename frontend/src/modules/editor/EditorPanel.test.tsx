import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
import * as zones from "./zones";

vi.mock("./zones", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./zones")>();
  return { ...actual, clipMaskToBox: vi.fn(actual.clipMaskToBox) };
});

// jsdom no decodifica imágenes: el retoque real se prueba en retouch.test.ts.
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

// jsdom no trae PointerEvent: sin esto fireEvent pierde altKey (Alt+clic de Clonar).
class TestPointerEvent extends MouseEvent {
  readonly pointerId: number;

  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 0;
  }
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

async function openPeopleWithTwoToRemove() {
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

async function protectObject() {
  fireEvent.click(await screen.findByRole("radio", { name: en["editor.tool.protect"] }));
  fireEvent.click(screen.getByTestId("editor-mask-canvas"), { clientX: 5, clientY: 5 });
  await waitFor(() => expect(screen.getByRole("button", { name: en["editor.protect.clear"] })).toBeEnabled());
}

describe("EditorPanel", () => {
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
    fireEvent.click(screen.getByRole("button", { name: "Add 1 silhouettes" }));
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
    expect(await screen.findByText("1 silhouettes in the erase")).toBeInTheDocument();
    expect(screen.getByText("I couldn't cut out 1 people; mark them with the brush.")).toHaveAttribute("aria-live", "polite");
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

  it("disables People when detection is unavailable", async () => {
    renderPanel();
    await loadBaseImage();

    expect(screen.getByRole("radio", { name: "People" })).toBeDisabled();
  });

  it("defaults to eraser-auto and visibly disables replacement without a fill model", async () => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue({
      ...CAPABILITIES,
      models: [
        {
          id: "eraser-auto",
          name: "Borrado automático (LaMa + OSOR)",
          status: "installed",
          eraseOnly: true,
          supportsInpaint: true,
        },
      ],
    });
    renderPanel();
    await loadBaseImage();

    expect(screen.getByLabelText(en["editor.model.label"])).toHaveValue("eraser-auto");
    const replaceRadio = screen.getByRole("radio", { name: en["editor.mode.replace"] });
    expect(replaceRadio).toBeDisabled();
    expect(replaceRadio).toHaveAccessibleDescription("Needs a fill model (SDXL)");
    expect(screen.getByText(en["editor.replace.needsSdxl"])).toBeVisible();
  });

  it("does not show replacement as unavailable while capabilities are loading", async () => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockReturnValue(new Promise(() => {}));
    renderPanel();
    await loadBaseImage();

    expect(screen.getByRole("radio", { name: en["editor.mode.replace"] })).toBeEnabled();
    expect(screen.queryByText(en["editor.replace.needsSdxl"])).not.toBeInTheDocument();
  });

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

  it("keeps the remove-selection controls visible after a job completes", async () => {
    enableMaskPixels();
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(BASE_IMAGE)
      .mockResolvedValueOnce({ ...BASE_IMAGE, initImageToken: "mask-token" });
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({
      id: "job-review",
      status: "queued",
    } as never);
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({
      id: "job-review",
      status: "completed",
      prompt: "background",
      negativePrompt: null,
      modelId: "gen--sdxl",
      steps: 30,
      guidance: 7,
      width: 640,
      height: 480,
      seed: null,
      device: "dml:0",
      autoUpscale: false,
      createdAt: "2026-10-30T00:00:00Z",
      startedAt: "2026-10-30T00:00:01Z",
      finishedAt: "2026-10-30T00:00:02Z",
      progressPct: 100,
      stages: [],
      error: null,
      ownerId: null,
      downloadUrl: "/api/v1/generation/jobs/job-review/download",
    });
    renderPanel();
    await loadBaseImage();
    fireEvent.change(screen.getByLabelText(en["editor.model.label"]), {
      target: { value: "gen--sdxl" },
    });
    paintOnCanvas();

    fireEvent.click(screen.getByRole("button", { name: en["editor.submit.erase"] }));

    await waitFor(() => expect(generationService.getGenerationJob).toHaveBeenCalledWith("job-review"));
    expect(await screen.findByRole("button", { name: en["editor.useAsBase"] })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove selection" })).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole("button", { name: en["editor.tool.pan"] }));

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

describe("EditorPanel: diseño de tres columnas", () => {
  it("marca la herramienta activa sin usar el relleno ámbar", async () => {
    renderPanel();
    await loadBaseImage();

    const activeTool = screen.getByRole("radio", { name: en["editor.tool.brush"] });
    expect(activeTool).toHaveAttribute("aria-checked", "true");
    expect(activeTool.className).not.toContain("bg-accent");
    expect(activeTool).toHaveStyle({ boxShadow: "inset 3px 0 0 var(--accent)" });
  });

  it("mueve el foco y selecciona herramientas con las flechas del radiogroup", async () => {
    renderPanel();
    await loadBaseImage();

    const brush = screen.getByRole("radio", { name: en["editor.tool.brush"] });
    fireEvent.keyDown(brush, { key: "ArrowDown" });

    const eraser = screen.getByRole("radio", { name: en["editor.tool.eraser"] });
    expect(eraser).toHaveAttribute("aria-checked", "true");
    expect(eraser).toHaveFocus();
  });

  it("muestra la acción dentro del panel complementario", async () => {
    renderPanel();
    await loadBaseImage();

    const sidePanel = screen.getByRole("complementary");
    expect(
      within(sidePanel).getByRole("button", { name: en["editor.submit.erase"] }),
    ).toBeInTheDocument();
  });

  it("cambia el tamaño del pincel con los atajos y respeta los límites", async () => {
    renderPanel();
    await loadBaseImage();

    const brushSize = screen.getByLabelText(en["editor.brushSize"]);
    fireEvent.keyDown(window, { key: "]" });
    expect(brushSize).toHaveValue("36");

    for (let step = 0; step < 30; step += 1) {
      fireEvent.keyDown(window, { key: "]" });
    }
    expect(brushSize).toHaveValue("128");

    for (let step = 0; step < 40; step += 1) {
      fireEvent.keyDown(window, { key: "[" });
    }
    expect(brushSize).toHaveValue("8");
  });

  it("clone without a source shows the Alt+click notice until a source is set", async () => {
    vi.stubGlobal("PointerEvent", TestPointerEvent);
    renderPanel();
    await loadBaseImage();
    fireEvent.click(screen.getByRole("radio", { name: en["editor.tool.clone"] }));
    expect(screen.getByText(en["editor.retouch.cloneNoSource"])).toBeInTheDocument();

    const canvas = screen.getByTestId("editor-mask-canvas");
    fireEvent.pointerDown(canvas, { clientX: 40, clientY: 40, pointerId: 1, altKey: true });
    fireEvent.pointerUp(canvas, { pointerId: 1 });
    expect(screen.queryByText(en["editor.retouch.cloneNoSource"])).not.toBeInTheDocument();
  });

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
