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
  supportedModels: [],
  videoProfiles: [],
  ffmpegAvailable: true,
};

const CPU_ONLY_DEVICES: DevicesResponse = {
  devices: [{ id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" }],
  defaultDeviceId: "cpu",
};

function renderPanel(devices: DevicesResponse = DEVICES) {
  vi.mocked(api.getModels).mockResolvedValue(MODELS);
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
});
