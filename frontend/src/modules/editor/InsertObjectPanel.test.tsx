import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { en } from "../../i18n/en";
import type { GenerationCapabilities, InitImageResponse } from "../../lib/apiTypes";
import * as api from "../../lib/api";
import * as editorService from "../../services/editor";
import * as generationService from "../../services/generation";
import { EditorPanel } from "./EditorPanel";
import { InsertObjectPanel } from "./InsertObjectPanel";

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
  insertObject: vi.fn(),
}));

const CAPABILITIES: GenerationCapabilities = {
  available: true,
  reason: null,
  devices: ["cpu"],
  cpuOnly: true,
  models: [
    { id: "gen--sdxl", name: "epicrealism", status: "installed", supportsInpaint: true },
  ],
};

const CAPABILITIES_WITH_INPAINT: GenerationCapabilities = {
  ...CAPABILITIES,
  models: [
    ...CAPABILITIES.models,
    {
      id: "gen--inpaint",
      name: "sd15-inpaint",
      status: "installed",
      supportsInpaint: true,
      inpaintOnly: true,
    },
  ],
};

const TARGET: InitImageResponse = {
  initImageToken: "target-token",
  originalFilename: "base.png",
  width: 640,
  height: 480,
};

const SOURCE: InitImageResponse = {
  initImageToken: "source-token",
  originalFilename: "obj.png",
  width: 400,
  height: 200,
};

const MASK: InitImageResponse = {
  initImageToken: "mask-token",
  originalFilename: "mask.png",
  width: 400,
  height: 200,
};

const TARGET_MASK: InitImageResponse = {
  initImageToken: "target-mask-token",
  originalFilename: "target-mask.png",
  width: 640,
  height: 480,
};

// Escala por defecto al subir SOURCE: el objeto entra ocupando 1/3 del ancho
// destino → clamp((640/3)/400*100) = 53%, o sea 212x106 px sobre el destino.
const DEFAULT_SCALED = { width: 212, height: 106 };

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
  (URL as { createObjectURL?: unknown }).createObjectURL = vi.fn(() => "blob:fake");
  (URL as { revokeObjectURL?: unknown }).revokeObjectURL = vi.fn();

  vi.mocked(api.getDevices).mockResolvedValue({
    devices: [{ id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" }],
    defaultDeviceId: "cpu",
  });
  vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue(CAPABILITIES);
  vi.mocked(generationService.uploadGenerationInitImage).mockResolvedValue(TARGET);
  vi.mocked(editorService.fetchEditorCapabilities).mockResolvedValue({ tapSelect: true });
});

afterEach(() => {
  vi.mocked(generationService.uploadGenerationInitImage).mockReset();
  vi.mocked(generationService.fetchGenerationCapabilities).mockReset();
  vi.mocked(editorService.fetchEditorCapabilities).mockReset();
  vi.mocked(editorService.segmentEditorObject).mockReset();
  vi.mocked(editorService.insertObject).mockReset();
});

function renderWithProviders(ui: ReactElement) {
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
  return render(ui, { wrapper: Wrapper });
}

function openPanel() {
  fireEvent.click(
    screen.getByRole("button", { name: (name) => name.startsWith(en["editor.insert.title"]) }),
  );
}

async function uploadSource() {
  fireEvent.change(screen.getByLabelText(en["editor.insert.sourceInputLabel"]), {
    target: { files: [new File(["obj"], "obj.png", { type: "image/png" })] },
  });
  return await screen.findByTestId("insert-source-image");
}

async function selectObjectOnPreview() {
  const preview = await uploadSource();
  fireEvent.click(preview, { clientX: 5, clientY: 5 });
  await waitFor(() =>
    expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(2),
  );
  return preview;
}

// El mapa se dibuja a 320x240: exactamente la mitad del destino 640x480, así
// cada click del test se escala x2 a coordenadas de imagen sin decimales.
function mockPlacementMapRect() {
  const map = screen.getByTestId("insert-placement-map");
  vi.spyOn(map, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    width: 320,
    height: 240,
    right: 320,
    bottom: 240,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  } as DOMRect);
  return map;
}

async function clickInsert() {
  const insert = screen.getByRole("button", { name: en["editor.insert.insert"] });
  await waitFor(() => expect(insert).toBeEnabled());
  fireEvent.click(insert);
  await waitFor(() => expect(editorService.insertObject).toHaveBeenCalledOnce());
}

describe("InsertObjectPanel", () => {
  it("only appears once a base image is loaded", async () => {
    renderWithProviders(<EditorPanel />);
    expect(screen.queryByText(en["editor.insert.title"])).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(en["editor.dropzone.inputLabel"]), {
      target: { files: [new File(["img"], "foto.png", { type: "image/png" })] },
    });
    await screen.findByTestId("editor-mask-canvas");

    expect(screen.getByText(en["editor.insert.title"])).toBeInTheDocument();
  });

  it("uploads the source image and shows a preview", async () => {
    vi.mocked(generationService.uploadGenerationInitImage).mockResolvedValueOnce(SOURCE);
    renderWithProviders(<InsertObjectPanel targetInfo={TARGET} />);
    openPanel();

    const preview = await uploadSource();

    expect(generationService.uploadGenerationInitImage).toHaveBeenCalledOnce();
    expect(preview).toHaveAttribute("src", "blob:fake");
  });

  it("clicking the preview segments with image-mapped coordinates", async () => {
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(SOURCE)
      .mockResolvedValueOnce(MASK);
    vi.mocked(editorService.segmentEditorObject).mockResolvedValue(
      new Blob(["png"], { type: "image/png" }),
    );
    renderWithProviders(<InsertObjectPanel targetInfo={TARGET} />);
    openPanel();
    const preview = await uploadSource();
    vi.spyOn(preview, "getBoundingClientRect").mockReturnValue({
      left: 0,
      top: 0,
      width: 200,
      height: 100,
      right: 200,
      bottom: 100,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect);

    fireEvent.click(preview, { clientX: 50, clientY: 25 });

    // El preview mide 200x100 y la imagen origen 400x200: el click se escala x2.
    await waitFor(() =>
      expect(editorService.segmentEditorObject).toHaveBeenCalledWith("source-token", 100, 50),
    );
    await waitFor(() =>
      expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(2),
    );
    expect(await screen.findByTestId("insert-mask-overlay")).toBeInTheDocument();
  });

  it("insert sends the three tokens and shows the composite with a download link", async () => {
    // Ajuste: los inputs X/Y no existen más. Sin click en el mapa, la posición
    // por defecto es el CENTRO del destino (320,240); el body lleva la esquina
    // derivada centro→esquina con la escala default 53%: (320-106, 240-53).
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(SOURCE)
      .mockResolvedValueOnce(MASK);
    vi.mocked(editorService.segmentEditorObject).mockResolvedValue(
      new Blob(["png"], { type: "image/png" }),
    );
    vi.mocked(editorService.insertObject).mockResolvedValue({
      compositeToken: "comp-1",
      compositePngBase64: "QUJD",
      jobId: null,
    });
    renderWithProviders(<InsertObjectPanel targetInfo={TARGET} />);
    openPanel();
    await selectObjectOnPreview();

    await clickInsert();

    expect(editorService.insertObject).toHaveBeenCalledWith(
      expect.objectContaining({
        targetToken: "target-token",
        sourceToken: "source-token",
        sourceMaskToken: "mask-token",
        x: 320 - DEFAULT_SCALED.width / 2,
        y: 240 - DEFAULT_SCALED.height / 2,
        width: DEFAULT_SCALED.width,
        height: DEFAULT_SCALED.height,
        matchColor: true,
        harmonize: false,
      }),
    );
    const composite = await screen.findByTestId("insert-composite");
    expect(composite).toHaveAttribute("src", "data:image/png;base64,QUJD");
    expect(screen.getByRole("link", { name: en["editor.insert.download"] })).toHaveAttribute(
      "href",
      "data:image/png;base64,QUJD",
    );
  });

  it("clicking the placement map sets the object's center and insert derives the corner", async () => {
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(SOURCE)
      .mockResolvedValueOnce(MASK);
    vi.mocked(editorService.segmentEditorObject).mockResolvedValue(
      new Blob(["png"], { type: "image/png" }),
    );
    vi.mocked(editorService.insertObject).mockResolvedValue({
      compositeToken: "comp-1",
      compositePngBase64: "QUJD",
      jobId: null,
    });
    renderWithProviders(<InsertObjectPanel targetInfo={TARGET} />);
    openPanel();
    await selectObjectOnPreview();
    const map = mockPlacementMapRect();

    // Click en (80,60) del mapa → centro (160,120) en el destino → esquina
    // (160-106, 120-53) con el objeto escalado a 212x106.
    fireEvent.click(map, { clientX: 80, clientY: 60 });
    await clickInsert();

    expect(editorService.insertObject).toHaveBeenCalledWith(
      expect.objectContaining({
        x: 160 - DEFAULT_SCALED.width / 2,
        y: 120 - DEFAULT_SCALED.height / 2,
        width: DEFAULT_SCALED.width,
        height: DEFAULT_SCALED.height,
      }),
    );
  });

  it("replace mode segments the target at map-scaled coordinates and sends targetMaskToken", async () => {
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(SOURCE)
      .mockResolvedValueOnce(MASK)
      .mockResolvedValueOnce(TARGET_MASK);
    vi.mocked(editorService.segmentEditorObject).mockResolvedValue(
      new Blob(["png"], { type: "image/png" }),
    );
    vi.mocked(editorService.insertObject).mockResolvedValue({
      compositeToken: "comp-1",
      compositePngBase64: "QUJD",
      jobId: null,
    });
    renderWithProviders(<InsertObjectPanel targetInfo={TARGET} />);
    openPanel();
    await selectObjectOnPreview();

    fireEvent.click(screen.getByLabelText(en["editor.insert.modeReplace"]));
    // En modo reemplazo el tamaño sale del objeto reemplazado: sin escala.
    expect(screen.queryByLabelText(en["editor.insert.scale"])).not.toBeInTheDocument();

    const map = mockPlacementMapRect();
    fireEvent.click(map, { clientX: 80, clientY: 60 });

    await waitFor(() =>
      expect(editorService.segmentEditorObject).toHaveBeenLastCalledWith("target-token", 160, 120),
    );
    await waitFor(() =>
      expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(3),
    );
    expect(await screen.findByTestId("insert-target-mask-overlay")).toBeInTheDocument();

    await clickInsert();

    // Con targetMaskToken el backend ignora la geometría: van los defaults del schema.
    expect(editorService.insertObject).toHaveBeenCalledWith(
      expect.objectContaining({
        targetMaskToken: "target-mask-token",
        x: 0,
        y: 0,
        width: 8,
        height: 8,
      }),
    );
  });

  it("shows the integration prompt only with harmonize on and sends its value", async () => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue(
      CAPABILITIES_WITH_INPAINT,
    );
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(SOURCE)
      .mockResolvedValueOnce(MASK);
    vi.mocked(editorService.segmentEditorObject).mockResolvedValue(
      new Blob(["png"], { type: "image/png" }),
    );
    vi.mocked(editorService.insertObject).mockResolvedValue({
      compositeToken: "comp-1",
      compositePngBase64: "QUJD",
      jobId: "job-1",
    });
    renderWithProviders(<InsertObjectPanel targetInfo={TARGET} />);
    openPanel();
    await selectObjectOnPreview();

    expect(
      screen.queryByLabelText(en["editor.insert.integrationPrompt"]),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText(en["editor.insert.harmonize"]));
    const promptInput = await screen.findByLabelText(en["editor.insert.integrationPrompt"]);
    expect(promptInput).toHaveAttribute(
      "placeholder",
      en["editor.insert.integrationPromptPlaceholder"],
    );
    fireEvent.change(promptInput, { target: { value: "night scene, warm lighting" } });

    await clickInsert();

    expect(editorService.insertObject).toHaveBeenCalledWith(
      expect.objectContaining({
        harmonize: true,
        modelId: "gen--inpaint",
        prompt: "night scene, warm lighting",
      }),
    );
  });

  it("draws the ghost on the placement map once an object is segmented", async () => {
    vi.mocked(generationService.uploadGenerationInitImage)
      .mockResolvedValueOnce(SOURCE)
      .mockResolvedValueOnce(MASK);
    vi.mocked(editorService.segmentEditorObject).mockResolvedValue(
      new Blob(["png"], { type: "image/png" }),
    );
    renderWithProviders(<InsertObjectPanel targetInfo={TARGET} />);
    openPanel();
    await uploadSource();

    // Sin objeto segmentado todavía no hay nada que colocar: sin fantasma.
    expect(screen.queryByTestId("insert-ghost")).not.toBeInTheDocument();

    fireEvent.click(screen.getByTestId("insert-source-image"), { clientX: 5, clientY: 5 });
    await waitFor(() =>
      expect(generationService.uploadGenerationInitImage).toHaveBeenCalledTimes(2),
    );

    const ghost = await screen.findByTestId("insert-ghost");
    expect(screen.getByTestId("insert-placement-map")).toContainElement(ghost);
  });

  it("shows a notice when harmonize is on but no inpaint-only model is installed", async () => {
    renderWithProviders(<InsertObjectPanel targetInfo={TARGET} />);
    openPanel();

    fireEvent.click(screen.getByLabelText(en["editor.insert.harmonize"]));

    expect(await screen.findByText(en["editor.insert.needsInpaintModel"])).toBeInTheDocument();
  });

  it("offers only inpaint-only models in the harmonize selector", async () => {
    vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue(
      CAPABILITIES_WITH_INPAINT,
    );
    renderWithProviders(<InsertObjectPanel targetInfo={TARGET} />);
    openPanel();

    fireEvent.click(screen.getByLabelText(en["editor.insert.harmonize"]));

    const select = await screen.findByLabelText(en["editor.insert.model"]);
    const labels = Array.from(select.querySelectorAll("option")).map((option) => option.textContent);
    expect(labels).toEqual(["sd15-inpaint"]);
  });
});
