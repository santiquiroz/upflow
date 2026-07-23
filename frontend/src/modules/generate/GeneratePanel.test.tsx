import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type { DevicesResponse, GenerationCapabilities, GenerationJob, ModelsResponse } from "../../lib/apiTypes";
import * as generationService from "../../services/generation";
import { GeneratePanel } from "./GeneratePanel";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getDevices: vi.fn(), getModels: vi.fn() };
});

vi.mock("../../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/generation")>();
  return {
    ...actual,
    createGenerationJob: vi.fn(),
    getGenerationJob: vi.fn(),
    fetchGenerationCapabilities: vi.fn(),
  };
});

const DEVICES: DevicesResponse = {
  devices: [{ id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900", backend: "directml" }],
  defaultDeviceId: "dml:0",
};

const CPU_ONLY_DEVICES: DevicesResponse = {
  devices: [{ id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" }],
  defaultDeviceId: "cpu",
};

const UPSCALE_MODELS: ModelsResponse = {
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

const AVAILABLE_CAPABILITIES: GenerationCapabilities = {
  available: true,
  reason: null,
  models: [{ id: "sd15-onnx", name: "SD 1.5 (ONNX)" }],
  devices: ["dml:0"],
  cpuOnly: false,
};

const CPU_ONLY_CAPABILITIES: GenerationCapabilities = {
  available: true,
  reason: null,
  models: [{ id: "sd15-onnx", name: "SD 1.5 (ONNX)" }],
  devices: ["cpu"],
  cpuOnly: true,
};

const BASE_JOB: GenerationJob = {
  id: "gen-1",
  status: "completed",
  prompt: "a red fox in the snow",
  negativePrompt: null,
  modelId: "sd15-onnx",
  steps: 25,
  guidance: 7.5,
  width: 512,
  height: 512,
  seed: null,
  device: null,
  autoUpscale: false,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: "2026-01-01T00:00:01Z",
  finishedAt: "2026-01-01T00:00:02Z",
  progressPct: null,
  stages: null,
  error: null,
  downloadUrl: "/api/v1/generation/jobs/gen-1/download",
};

function renderPanel(
  capabilities: GenerationCapabilities = AVAILABLE_CAPABILITIES,
  devices: DevicesResponse = DEVICES,
) {
  vi.mocked(api.getDevices).mockResolvedValue(devices);
  vi.mocked(api.getModels).mockResolvedValue(UPSCALE_MODELS);
  vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue(capabilities);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>{children}</MemoryRouter>
      </QueryClientProvider>
    );
  }
  return render(<GeneratePanel />, { wrapper: Wrapper });
}

async function fillPromptAndModel() {
  const promptField = await screen.findByLabelText(/^prompt$/i);
  fireEvent.change(promptField, { target: { value: "a red fox in the snow" } });
  const modelSelect = screen.getByLabelText(/^model$/i);
  fireEvent.change(modelSelect, { target: { value: "sd15-onnx" } });
}

afterEach(() => {
  vi.mocked(api.getDevices).mockReset();
  vi.mocked(api.getModels).mockReset();
  vi.mocked(generationService.fetchGenerationCapabilities).mockReset();
  vi.mocked(generationService.createGenerationJob).mockReset();
  vi.mocked(generationService.getGenerationJob).mockReset();
});

describe("GeneratePanel", () => {
  it("shows the unavailable banner with the capabilities reason and hides the form", async () => {
    renderPanel({ available: false, reason: "No compatible ONNX runtime found.", models: [], devices: [], cpuOnly: false });

    expect(await screen.findByRole("alert")).toHaveTextContent("No compatible ONNX runtime found.");
    expect(screen.queryByLabelText(/prompt/i)).not.toBeInTheDocument();
  });

  it("populates the model select from capabilities and submits with the chosen params", async () => {
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ ...BASE_JOB });
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({ ...BASE_JOB });

    renderPanel();
    await fillPromptAndModel();

    const submitButton = await screen.findByRole("button", { name: /^generate$/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    expect(vi.mocked(generationService.createGenerationJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({
        prompt: "a red fox in the snow",
        modelId: "sd15-onnx",
        steps: 25,
        guidance: 7.5,
        width: 512,
        height: 512,
        autoUpscale: false,
        upscaleModelName: null,
        upscaleModelId: null,
        upscaleScale: null,
      }),
    );
  });

  it("blocks the first Generate click on a CPU-only machine and submits after confirming", async () => {
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ ...BASE_JOB });
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({ ...BASE_JOB });

    renderPanel(CPU_ONLY_CAPABILITIES, CPU_ONLY_DEVICES);
    await fillPromptAndModel();

    const submitButton = await screen.findByRole("button", { name: /^generate$/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    expect(generationService.createGenerationJob).not.toHaveBeenCalled();
    expect(
      await screen.findByText("No se detectó GPU compatible (DirectX 12). Generar en CPU tarda varios minutos por imagen. ¿Continuar igual?"),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /continuar igual/i }));

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
  });

  it("hides upscale params when auto-upscale is off and includes them when on", async () => {
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ ...BASE_JOB });
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({ ...BASE_JOB });

    renderPanel();
    await fillPromptAndModel();

    const submitButton = await screen.findByRole("button", { name: /^generate$/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalledTimes(1));
    expect(vi.mocked(generationService.createGenerationJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ upscaleModelName: null, upscaleModelId: null, upscaleScale: null }),
    );

    expect(screen.queryByRole("radio", { name: /RealESRGAN x4plus/ })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("checkbox", { name: /escalar automáticamente/i }));

    const upscaleRadio = await screen.findByRole("radio", { name: /RealESRGAN x4plus/ });
    fireEvent.click(upscaleRadio);
    fireEvent.change(screen.getByLabelText(/scale/i), { target: { value: "3" } });

    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalledTimes(2));
    expect(vi.mocked(generationService.createGenerationJob).mock.calls[1][0]).toEqual(
      expect.objectContaining({ upscaleModelName: "RealESRGAN x4plus", upscaleModelId: null, upscaleScale: 3 }),
    );
  });
});
