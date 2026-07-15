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

function renderPanel() {
  vi.mocked(api.getModels).mockResolvedValue(MODELS);
  vi.mocked(api.getDevices).mockResolvedValue(DEVICES);
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
  it("disables the Upscale CTA until a file and a model are selected", async () => {
    renderPanel();

    const submitButton = screen.getByRole("button", { name: /upscale/i });
    expect(submitButton).toBeDisabled();

    await selectFileAndModel();

    await waitFor(() => expect(submitButton).not.toBeDisabled());
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
