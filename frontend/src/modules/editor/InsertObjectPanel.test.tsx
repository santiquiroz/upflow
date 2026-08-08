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

    const insert = screen.getByRole("button", { name: en["editor.insert.insert"] });
    await waitFor(() => expect(insert).toBeEnabled());
    fireEvent.click(insert);

    await waitFor(() => expect(editorService.insertObject).toHaveBeenCalledOnce());
    expect(editorService.insertObject).toHaveBeenCalledWith(
      expect.objectContaining({
        targetToken: "target-token",
        sourceToken: "source-token",
        sourceMaskToken: "mask-token",
        x: 320,
        y: 240,
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
