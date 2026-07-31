import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import * as api from "../../lib/api";
import type {
  DevicesResponse,
  GenerationCapabilities,
  GenerationJob,
  InitImageResponse,
  ModelsResponse,
} from "../../lib/apiTypes";
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
    uploadGenerationInitImage: vi.fn(),
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

const MIXED_DEVICES: DevicesResponse = {
  devices: [
    { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" },
    { id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900", backend: "directml" },
  ],
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
  models: [{ id: "sd15-onnx", name: "SD 1.5 (ONNX)", status: "installed" }],
  devices: ["dml:0"],
  cpuOnly: false,
};

const CPU_ONLY_CAPABILITIES: GenerationCapabilities = {
  available: true,
  reason: null,
  models: [{ id: "sd15-onnx", name: "SD 1.5 (ONNX)", status: "installed" }],
  devices: ["cpu"],
  cpuOnly: true,
};

const BASE_JOB: GenerationJob = {
  ownerId: null,
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

const INIT_IMAGE: InitImageResponse = {
  initImageToken: "a1b2c3",
  originalFilename: "starting-photo.png",
  width: 1024,
  height: 768,
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
        <LocaleProvider initialLocale="en">
          <MemoryRouter>{children}</MemoryRouter>
        </LocaleProvider>
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

async function switchToImageMode() {
  fireEvent.click(
    await screen.findByRole("radio", { name: en["generation.mode.imageToImage"] }),
  );
}

async function uploadInitImage(file = new File(["pixels"], "starting-photo.png", { type: "image/png" })) {
  vi.mocked(generationService.uploadGenerationInitImage).mockResolvedValue({ ...INIT_IMAGE });
  await switchToImageMode();
  fireEvent.change(screen.getByLabelText(en["generation.initImage.inputLabel"]), {
    target: { files: [file] },
  });
  await screen.findByText(INIT_IMAGE.originalFilename);
}

afterEach(() => {
  vi.mocked(api.getDevices).mockReset();
  vi.mocked(api.getModels).mockReset();
  vi.mocked(generationService.fetchGenerationCapabilities).mockReset();
  vi.mocked(generationService.createGenerationJob).mockReset();
  vi.mocked(generationService.getGenerationJob).mockReset();
  vi.mocked(generationService.uploadGenerationInitImage).mockReset();
});

describe("GeneratePanel", () => {
  it("shows the unavailable banner with the capabilities reason and hides the form", async () => {
    renderPanel({ available: false, reason: "No compatible ONNX runtime found.", models: [], devices: [], cpuOnly: false });

    expect(await screen.findByRole("alert")).toHaveTextContent("No compatible ONNX runtime found.");
    expect(screen.queryByLabelText(/prompt/i)).not.toBeInTheDocument();
  });

  it("defaults to text-to-image mode and keeps the existing panel workflow", async () => {
    renderPanel();

    expect(
      await screen.findByRole("radio", { name: en["generation.mode.textToImage"] }),
    ).toBeChecked();
    expect(
      screen.getByRole("radio", { name: en["generation.mode.imageToImage"] }),
    ).not.toBeChecked();
    expect(
      screen.queryByLabelText(en["generation.initImage.inputLabel"]),
    ).not.toBeInTheDocument();

    await fillPromptAndModel();
    expect(screen.getByRole("button", { name: /^generate$/i })).not.toBeDisabled();
  });

  it("preserves the prompt when switching to image-to-image mode and back", async () => {
    renderPanel();
    const promptField = await screen.findByLabelText(/^prompt$/i);
    fireEvent.change(promptField, { target: { value: "keep this prompt" } });

    await switchToImageMode();
    fireEvent.click(
      screen.getByRole("radio", { name: en["generation.mode.textToImage"] }),
    );

    expect(promptField).toHaveValue("keep this prompt");
  });

  it("uploads a starting image, shows its real metadata, and sends its token in the job", async () => {
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ ...BASE_JOB });
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({ ...BASE_JOB });

    renderPanel();
    await fillPromptAndModel();
    await uploadInitImage();

    expect(generationService.uploadGenerationInitImage).toHaveBeenCalledWith(
      expect.objectContaining({ name: "starting-photo.png" }),
    );
    expect(
      screen.getByText(
        en["generation.initImage.dimensions"]
          .replace("{{width}}", String(INIT_IMAGE.width))
          .replace("{{height}}", String(INIT_IMAGE.height)),
      ),
    ).toBeInTheDocument();

    const submitButton = screen.getByRole("button", { name: /^generate$/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    expect(vi.mocked(generationService.createGenerationJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ initImageToken: INIT_IMAGE.initImageToken }),
    );
  });

  it("blocks image-to-image generation without an uploaded image and explains why", async () => {
    renderPanel();
    await fillPromptAndModel();
    await switchToImageMode();

    expect(screen.getByRole("button", { name: /^generate$/i })).toBeDisabled();
    expect(screen.getByText(en["generation.initImage.required"])).toBeInTheDocument();
  });

  it("omits initImageToken and strength from a text-to-image job", async () => {
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ ...BASE_JOB });
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({ ...BASE_JOB });

    renderPanel();
    await fillPromptAndModel();
    fireEvent.click(screen.getByRole("button", { name: /^generate$/i }));

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    const params = vi.mocked(generationService.createGenerationJob).mock.calls[0][0];
    expect(params).not.toHaveProperty("initImageToken");
    expect(params).not.toHaveProperty("strength");
  });

  it("sends the selected strength in an image-to-image job", async () => {
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ ...BASE_JOB });
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({ ...BASE_JOB });

    renderPanel();
    await fillPromptAndModel();
    await uploadInitImage();
    fireEvent.change(screen.getByLabelText(en["generation.strength.label"]), {
      target: { value: "0.35" },
    });
    fireEvent.click(screen.getByRole("button", { name: /^generate$/i }));

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    expect(vi.mocked(generationService.createGenerationJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ strength: 0.35 }),
    );
  });

  it("shows the backend detail when the selected model rejects image-to-image", async () => {
    const detail = "The selected model does not support image-to-image generation.";
    vi.mocked(generationService.createGenerationJob).mockRejectedValue(
      new api.ApiError(400, detail),
    );

    renderPanel();
    await fillPromptAndModel();
    await uploadInitImage();
    fireEvent.click(screen.getByRole("button", { name: /^generate$/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(detail);
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

  it("does not offer the Auto device option (the generation backend rejects device='auto')", async () => {
    renderPanel();
    await fillPromptAndModel();

    expect(await screen.findByRole("radio", { name: /AMD Radeon RX 7900/ })).toBeInTheDocument();
    const deviceOptionValues = screen.getAllByRole("radio").map((radio) => radio.getAttribute("value"));
    expect(deviceOptionValues).not.toContain("auto");
  });

  it("clears the stale CPU-only warning once the device changes away from CPU, and submits directly", async () => {
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ ...BASE_JOB });
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({ ...BASE_JOB });

    renderPanel(CPU_ONLY_CAPABILITIES, MIXED_DEVICES);
    await fillPromptAndModel();

    const submitButton = await screen.findByRole("button", { name: /^generate$/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    expect(
      await screen.findByText("No se detectó GPU compatible (DirectX 12). Generar en CPU tarda varios minutos por imagen. ¿Continuar igual?"),
    ).toBeInTheDocument();
    expect(generationService.createGenerationJob).not.toHaveBeenCalled();

    fireEvent.click(await screen.findByRole("radio", { name: /AMD Radeon RX 7900/ }));

    expect(
      screen.queryByText("No se detectó GPU compatible (DirectX 12). Generar en CPU tarda varios minutos por imagen. ¿Continuar igual?"),
    ).not.toBeInTheDocument();

    fireEvent.click(submitButton);

    await waitFor(() => expect(generationService.createGenerationJob).toHaveBeenCalled());
    expect(vi.mocked(generationService.createGenerationJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ device: "dml:0" }),
    );
  });
});

describe("GeneratePanel — conversiones visibles en el dropdown", () => {
  it("un modelo convirtiéndose aparece deshabilitado con el aviso de demora", async () => {
    renderPanel({
      available: true,
      reason: null,
      models: [
        { id: "sd15-onnx", name: "SD 1.5 (ONNX)", status: "installed" },
        { id: "gen--john--anime", name: "john/anime", status: "converting" },
      ],
      devices: [{ id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900", backend: "directml" }],
      cpuOnly: false,
    });

    const option = (await screen.findByRole("option", {
      name: /john\/anime.*convirtiendo/i,
    })) as HTMLOptionElement;
    expect(option.disabled).toBe(true);
  });

  it("con solo conversiones en curso no se muestra el 'no hay modelos'", async () => {
    renderPanel({
      available: true,
      reason: null,
      models: [{ id: "gen--john--anime", name: "john/anime", status: "converting" }],
      devices: [{ id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900", backend: "directml" }],
      cpuOnly: false,
    });

    expect(await screen.findByRole("option", { name: /convirtiendo/i })).toBeInTheDocument();
    expect(screen.queryByText(/No generation models installed/i)).not.toBeInTheDocument();
  });
});
