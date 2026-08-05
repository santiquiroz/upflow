import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type {
  CreateJobResponse,
  DevicesResponse,
  EngineInfoResponse,
  JobResponse,
  ModelsResponse,
} from "../../lib/apiTypes";
import { ImagePanel } from "./ImagePanel";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return {
    ...actual,
    getModels: vi.fn(),
    getDevices: vi.fn(),
    getEngineInfo: vi.fn(),
    createImageJob: vi.fn(),
    getJob: vi.fn(),
  };
});

const MODELS: ModelsResponse = {
  models: [
    {
      id: "realesrgan-x4plus",
      name: "RealESRGAN x4plus",
      kind: "builtin-ncnn",
      source: "builtin",
      scale: 4,
      arch: "esrgan",
      sizeBytes: 0,
      status: "installed",
      error: null,
    },
  ],
};

const ANIME_X2_MODELS: ModelsResponse = {
  models: [
    {
      id: "realesr-animevideov3-x2",
      name: "RealESR AnimeVideo v3 x2",
      kind: "builtin-ncnn",
      source: "builtin",
      scale: 2,
      arch: "esrgan",
      sizeBytes: 0,
      status: "installed",
      error: null,
    },
  ],
};

const DEVICES: DevicesResponse = {
  devices: [
    { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" },
    { id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900", backend: "directml" },
  ],
  defaultDeviceId: "dml:0",
};

const ENGINE_INFO: EngineInfoResponse = {
  engine: "realesrgan-ncnn",
  configuredBinary: "vendor/realesrgan/realesrgan-ncnn-vulkan.exe",
  configuredModelsDir: "vendor/realesrgan/models",
  available: true,
  defaultModel: "realesrgan-x4plus",
  allowedScales: [2, 3, 4],
  supportedModels: [
    {
      key: "realesrgan-x4plus",
      label: "RealESRGAN x4plus",
      category: "general",
      description: "",
      scales: [2, 3, 4],
    },
    {
      key: "realesr-animevideov3-x2",
      label: "RealESR AnimeVideo v3 x2",
      category: "anime",
      description: "",
      scales: [2],
    },
  ],
  maxUploadMb: 50,
  maxVideoUploadMb: 2048,
  videoProfiles: [],
  ffmpegAvailable: true,
};

const CPU_ONLY_DEVICES: DevicesResponse = {
  devices: [{ id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" }],
  defaultDeviceId: "cpu",
};

function renderPanel(devices: DevicesResponse = DEVICES, models: ModelsResponse = MODELS) {
  vi.mocked(api.getModels).mockResolvedValue(models);
  vi.mocked(api.getDevices).mockResolvedValue(devices);
  vi.mocked(api.getEngineInfo).mockResolvedValue(ENGINE_INFO);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<ImagePanel />, { wrapper: Wrapper });
}

function makeFile(): File {
  return new File(["binary"], "photo.png", { type: "image/png" });
}

async function selectFileAndModel() {
  const fileInput = document.getElementById("image-file-input") as HTMLInputElement;
  fireEvent.change(fileInput, { target: { files: [makeFile()] } });

  const modelRadio = await screen.findByRole("radio", { name: /RealESRGAN x4plus/ });
  fireEvent.click(modelRadio);
}

afterEach(() => {
  vi.mocked(api.getModels).mockReset();
  vi.mocked(api.getDevices).mockReset();
  vi.mocked(api.getEngineInfo).mockReset();
  vi.mocked(api.createImageJob).mockReset();
  vi.mocked(api.getJob).mockReset();
});

describe("ImagePanel en lote", () => {
  async function selectFilesAndModel(nombres: string[]) {
    const fileInput = document.getElementById("image-file-input") as HTMLInputElement;
    fireEvent.change(fileInput, {
      target: {
        files: nombres.map((n) => new File(["binary"], n, { type: "image/png" })),
      },
    });
    const modelRadio = await screen.findByRole("radio", { name: /RealESRGAN x4plus/ });
    fireEvent.click(modelRadio);
  }

  it("creates one job per file instead of only the first", async () => {
    renderPanel();
    await selectFilesAndModel(["uno.png", "dos.png", "tres.png"]);
    fireEvent.click(await screen.findByRole("button", { name: /upscale/i }));

    await waitFor(() =>
      expect(vi.mocked(api.createImageJob)).toHaveBeenCalledTimes(3),
    );
    const enviados = vi
      .mocked(api.createImageJob)
      .mock.calls.map((call) => call[0].file.name);
    expect(enviados).toEqual(["uno.png", "dos.png", "tres.png"]);
  });

  it("applies the same settings to every file", async () => {
    renderPanel();
    await selectFilesAndModel(["uno.png", "dos.png"]);
    fireEvent.click(await screen.findByRole("button", { name: /upscale/i }));

    await waitFor(() =>
      expect(vi.mocked(api.createImageJob)).toHaveBeenCalledTimes(2),
    );
    const [primero, segundo] = vi
      .mocked(api.createImageJob)
      .mock.calls.map((call) => call[0]);
    expect(segundo.modelId).toBe(primero.modelId);
    expect(segundo.scale).toBe(primero.scale);
  });

  it("says how many files are queued up", async () => {
    renderPanel();
    await selectFilesAndModel(["uno.png", "dos.png", "tres.png"]);

    expect(screen.getByText("3 files selected")).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Upscale 3 files" }),
    ).toBeInTheDocument();
  });

  it("says how many of the batch never made it", async () => {
    // Que el servidor rechace tres de cinco y la pantalla no diga nada es la
    // misma falla silenciosa que los avisos del trabajo vienen a tapar.
    vi.mocked(api.createImageJob).mockImplementation(async (params) => {
      if (params.file.name === "dos.png") {
        throw new Error("Job queue is full");
      }
      return {
        jobId: `job-${params.file.name}`,
        status: "queued" as const,
        statusUrl: `/api/v1/jobs/job-${params.file.name}`,
        downloadUrl: null,
      };
    });
    renderPanel();
    await selectFilesAndModel(["uno.png", "dos.png", "tres.png"]);
    fireEvent.click(await screen.findByRole("button", { name: /upscale/i }));

    expect(await screen.findByText("1 file could not be sent")).toBeInTheDocument();
  });

  it("still works with a single file", async () => {
    renderPanel();
    await selectFilesAndModel(["solo.png"]);
    fireEvent.click(await screen.findByRole("button", { name: /upscale/i }));

    await waitFor(() =>
      expect(vi.mocked(api.createImageJob)).toHaveBeenCalledTimes(1),
    );
  });
});

describe("ImagePanel", () => {
  it("keeps the Model section expanded by default while Device and Scale & format start collapsed", async () => {
    renderPanel();

    expect(await screen.findByRole("button", { name: /^Model/ })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("button", { name: /^Device/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: /^Scale & format/ })).toHaveAttribute("aria-expanded", "false");
  });

  it("shows a placeholder summary until a model is picked, then reflects the selection", async () => {
    renderPanel();

    expect(await screen.findByRole("button", { name: /^Model/ })).toHaveTextContent("Select a model…");

    await selectFileAndModel();

    expect(screen.getByRole("button", { name: /^Model/ })).toHaveTextContent("RealESRGAN x4plus");
  });

  it("auto-fills the Device summary once a preferred device is resolved, without expanding it", async () => {
    renderPanel();
    await selectFileAndModel();

    expect(await screen.findByRole("button", { name: /^Device/ })).toHaveTextContent("AMD Radeon RX 7900");
  });

  it("shows the default scale and format in the Scale & format summary without expanding it", async () => {
    renderPanel();

    const scaleFormatToggle = await screen.findByRole("button", { name: /^Scale & format/ });
    await waitFor(() => expect(scaleFormatToggle).toHaveTextContent("4x"));
    expect(scaleFormatToggle).toHaveTextContent("PNG");
  });

  it("hides the Device options from the accessibility tree until the section is expanded", async () => {
    renderPanel();
    await selectFileAndModel();
    await screen.findByRole("button", { name: /^Device/ });

    expect(screen.queryByRole("radio", { name: /AMD Radeon RX 7900/ })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^Device/ }));

    expect(await screen.findByRole("radio", { name: /AMD Radeon RX 7900/ })).toBeInTheDocument();
  });

  it("disables the Upscale CTA until a file and a model are selected", async () => {
    renderPanel();

    const submitButton = screen.getByRole("button", { name: /upscale/i });
    expect(submitButton).toBeDisabled();

    await selectFileAndModel();

    await waitFor(() => expect(submitButton).not.toBeDisabled());
  });

  it("keeps the CTA disabled with a clear hint when a builtin model needs a GPU but only cpu exists", async () => {
    renderPanel(CPU_ONLY_DEVICES);
    await selectFileAndModel();

    const submitButton = screen.getByRole("button", { name: /upscale/i });
    expect(await screen.findByRole("status")).toHaveTextContent(/requires a Vulkan GPU/i);
    expect(submitButton).toBeDisabled();
    expect(vi.mocked(api.createImageJob)).not.toHaveBeenCalled();
  });

  it("submits the job and shows the completed preview with a download link", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "job-1",
      status: "queued",
      statusUrl: "/api/v1/jobs/job-1",
      downloadUrl: null,
    };
    const completedJob: JobResponse = {
      ownerId: null,
      jobId: "job-1",
      status: "completed",
      originalFilename: "photo.png",
      modelName: "realesrgan-x4plus",
      scale: 4,
      outputFormat: "png",
      modelId: "realesrgan-x4plus",
      device: "dml:0",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:01Z",
      finishedAt: "2026-01-01T00:00:02Z",
      error: null,
      metadata: {},
      progressPct: null,
      downloadUrl: "/api/v1/jobs/job-1/download",
    };
    vi.mocked(api.createImageJob).mockResolvedValue(createResponse);
    vi.mocked(api.getJob).mockResolvedValue(completedJob);

    renderPanel();
    await selectFileAndModel();

    const submitButton = await screen.findByRole("button", { name: /upscale/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    expect(await screen.findByRole("link", { name: /download/i })).toHaveAttribute(
      "href",
      "/api/v1/jobs/job-1/download",
    );
    expect(vi.mocked(api.createImageJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ modelId: "realesrgan-x4plus", device: "dml:0", scale: 4, outputFormat: "png" }),
    );
  });

  it("submits device='auto' when the Auto device option is selected", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "job-1",
      status: "queued",
      statusUrl: "/api/v1/jobs/job-1",
      downloadUrl: null,
    };
    vi.mocked(api.createImageJob).mockResolvedValue(createResponse);
    vi.mocked(api.getJob).mockResolvedValue({
      ownerId: null,
      jobId: "job-1",
      status: "queued",
      originalFilename: "photo.png",
      modelName: "realesrgan-x4plus",
      scale: 4,
      outputFormat: "png",
      modelId: "realesrgan-x4plus",
      device: "auto",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: null,
      finishedAt: null,
      error: null,
      metadata: {},
      progressPct: null,
      downloadUrl: null,
    });

    renderPanel();
    await selectFileAndModel();

    fireEvent.click(await screen.findByRole("button", { name: /^Device/ }));
    const autoRadio = await screen.findByRole("radio", { name: /Auto/ });
    fireEvent.click(autoRadio);

    const submitButton = await screen.findByRole("button", { name: /upscale/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(vi.mocked(api.createImageJob)).toHaveBeenCalled());
    expect(vi.mocked(api.createImageJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ device: "auto" }),
    );
  });

  it("shows an inline error message when the server rejects the upload", async () => {
    vi.mocked(api.createImageJob).mockRejectedValue(new Error("Job queue is full; try again later"));

    renderPanel();
    await selectFileAndModel();

    const submitButton = await screen.findByRole("button", { name: /upscale/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    expect(await screen.findByRole("alert")).toHaveTextContent("Job queue is full; try again later");
  });

  // El backend rechaza esta combinacion despues de subir el archivo entero
  // ("Model ... supports only scales [2]"). La pantalla ya tiene el dato para
  // no ofrecerla nunca.
  it("does not offer a scale the chosen model cannot do", async () => {
    renderPanel(DEVICES, ANIME_X2_MODELS);

    fireEvent.click(await screen.findByRole("radio", { name: /AnimeVideo/ }));
    const [scaleSection] = screen.getAllByRole("button", { name: /Scale & format/i });
    fireEvent.click(scaleSection);

    expect(await screen.findByRole("button", { name: "2x" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "4x" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "3x" })).not.toBeInTheDocument();
  });

  // El servidor corta la subida mientras la recibe, asi que sin este chequeo un
  // archivo demasiado grande se sube ENTERO antes de que alguien avise.
  it("refuses a file over the published limit without uploading it", async () => {
    renderPanel();
    await screen.findByRole("radio", { name: /RealESRGAN x4plus/ });

    const huge = new File(["x"], "enorme.png", { type: "image/png" });
    Object.defineProperty(huge, "size", { value: 3 * 1024 * 1024 * 1024 });
    const fileInput = document.getElementById("image-file-input") as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [huge] } });

    expect(await screen.findByRole("alert")).toHaveTextContent(/50 MB/);
    expect(vi.mocked(api.createImageJob)).not.toHaveBeenCalled();
  });
});
