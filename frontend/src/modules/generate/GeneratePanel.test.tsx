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
  VideoGenerationCapabilities,
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
    fetchVideoGenerationCapabilities: vi.fn(),
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

const VIDEO_CAPABILITIES: VideoGenerationCapabilities = {
  available: true,
  models: [
    {
      id: "sdcppvid:Wan2_2-TI2V-5B-Turbo-Q8_0",
      name: "Wan2_2-TI2V-5B-Turbo-Q8_0 (Vulkan)",
      fast: true,
      defaultSteps: 4,
      defaultGuidance: 1.0,
    },
    {
      id: "sdcppvid:Wan2.2-TI2V-5B-Q8_0",
      name: "Wan2.2-TI2V-5B-Q8_0 (Vulkan)",
      fast: false,
      defaultSteps: 20,
      defaultGuidance: 5.0,
    },
  ],
  defaultFrames: 33,
  defaultFps: 16,
  maxFrames: 81,
};

const NO_VIDEO_CAPABILITIES: VideoGenerationCapabilities = {
  available: false,
  models: [],
  defaultFrames: 33,
  defaultFps: 16,
  maxFrames: 81,
};

function renderPanel(
  capabilities: GenerationCapabilities = AVAILABLE_CAPABILITIES,
  devices: DevicesResponse = DEVICES,
  videoCapabilities: VideoGenerationCapabilities = VIDEO_CAPABILITIES,
) {
  vi.mocked(api.getDevices).mockResolvedValue(devices);
  vi.mocked(api.getModels).mockResolvedValue(UPSCALE_MODELS);
  vi.mocked(generationService.fetchGenerationCapabilities).mockResolvedValue(capabilities);
  vi.mocked(generationService.fetchVideoGenerationCapabilities).mockResolvedValue(videoCapabilities);
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
  vi.mocked(generationService.fetchVideoGenerationCapabilities).mockReset();
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
      devices: ["dml:0"],
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
      devices: ["dml:0"],
      cpuOnly: false,
    });

    expect(await screen.findByRole("option", { name: /convirtiendo/i })).toBeInTheDocument();
    expect(screen.queryByText(/No generation models installed/i)).not.toBeInTheDocument();
  });
});


describe("GeneratePanel — modo video", () => {

  it("switches to the video catalogue, which is a different one from the image models", async () => {
    renderPanel();
    await screen.findByRole("option", { name: "SD 1.5 (ONNX)" });
    fireEvent.change(screen.getByLabelText(/^model$/i), { target: { value: "sd15-onnx" } });
    fireEvent.change(screen.getByLabelText(/^prompt$/i), { target: { value: "un zorro" } });

    fireEvent.click(screen.getByRole("radio", { name: /^video$/i }));

    expect(
      await screen.findByRole("option", { name: /Wan2_2-TI2V-5B-Turbo-Q8_0 \(Vulkan\) - rapido, 4 pasos/i }),
    ).toBeInTheDocument();
    // Un modelo de imagen mandado al lane de video es un 400 seguro.
    expect(screen.queryByRole("option", { name: "SD 1.5 (ONNX)" })).not.toBeInTheDocument();
    // El id del modelo de imagen no puede sobrevivir al cruce: si sobrevive, el
    // boton queda habilitado y se manda a la API un modelo que ese lane no tiene.
    // Un <select> ya reporta "" cuando su valor no esta entre las opciones, asi
    // que lo unico que distingue de verdad es el estado del boton.
    expect(screen.getByRole("button", { name: /^generate$/i })).toBeDisabled();
  });

  it("turns auto upscale off when entering video, even if it was on", async () => {
    const createSpy = vi.mocked(generationService.createGenerationJob);
    createSpy.mockResolvedValue({ ...BASE_JOB, id: "gen-video" } as GenerationJob);
    renderPanel();
    await screen.findByRole("option", { name: "SD 1.5 (ONNX)" });

    fireEvent.click(screen.getByRole("checkbox", { name: /escalar autom/i }));
    fireEvent.click(screen.getByRole("radio", { name: /^video$/i }));
    await screen.findByRole("option", { name: /Turbo/i });
    fireEvent.change(screen.getByLabelText(/^model$/i), {
      target: { value: "sdcppvid:Wan2_2-TI2V-5B-Turbo-Q8_0" },
    });
    fireEvent.change(screen.getByLabelText(/^prompt$/i), { target: { value: "un zorro" } });
    fireEvent.click(screen.getByRole("button", { name: /^generate$/i }));

    await waitFor(() => expect(createSpy).toHaveBeenCalled());
    expect(createSpy.mock.calls[0][0].autoUpscale).toBe(false);
  });

  it("adopts the sampling defaults of the chosen video model instead of the image ones", async () => {
    renderPanel();
    await screen.findByRole("option", { name: "SD 1.5 (ONNX)" });
    fireEvent.click(screen.getByRole("radio", { name: /^video$/i }));
    await screen.findByRole("option", { name: /Turbo/i });

    fireEvent.change(screen.getByLabelText(/^model$/i), {
      target: { value: "sdcppvid:Wan2_2-TI2V-5B-Turbo-Q8_0" },
    });

    // El destilado se entreno con 4 pasos y CFG 1: con 25 y 7.5 sale quemado.
    expect(screen.getByLabelText(/^steps$/i)).toHaveValue(4);
    expect(screen.getByLabelText(/^guidance$/i)).toHaveValue(1);

    fireEvent.change(screen.getByLabelText(/^model$/i), {
      target: { value: "sdcppvid:Wan2.2-TI2V-5B-Q8_0" },
    });
    expect(screen.getByLabelText(/^steps$/i)).toHaveValue(20);
    expect(screen.getByLabelText(/^guidance$/i)).toHaveValue(5);
  });

  it("sends frames, fps and the cinematic size, and never auto-upscales a clip", async () => {
    const createSpy = vi.mocked(generationService.createGenerationJob);
    createSpy.mockResolvedValue({ ...BASE_JOB, id: "gen-video" } as GenerationJob);
    renderPanel();
    await screen.findByRole("option", { name: "SD 1.5 (ONNX)" });
    fireEvent.click(screen.getByRole("radio", { name: /^video$/i }));
    await screen.findByRole("option", { name: /Turbo/i });

    fireEvent.change(screen.getByLabelText(/^model$/i), {
      target: { value: "sdcppvid:Wan2_2-TI2V-5B-Turbo-Q8_0" },
    });
    fireEvent.change(screen.getByLabelText(/^prompt$/i), { target: { value: "un zorro corriendo" } });
    fireEvent.change(screen.getByLabelText(/cuadros por segundo/i), { target: { value: "24" } });
    fireEvent.change(screen.getByLabelText(/^cuadros$/i), { target: { value: "49" } });
    fireEvent.click(screen.getByRole("button", { name: /^generate$/i }));

    await waitFor(() => expect(createSpy).toHaveBeenCalled());
    const params = createSpy.mock.calls[0][0];
    expect(params.modelId).toBe("sdcppvid:Wan2_2-TI2V-5B-Turbo-Q8_0");
    expect(params.frames).toBe(49);
    expect(params.fps).toBe(24);
    // 832x480 es la relacion con la que Wan fue entrenado, y 480 no es multiplo de 64.
    expect(params.width).toBe(832);
    expect(params.height).toBe(480);
  });

  it("tells the user how long the clip will be", async () => {
    renderPanel();
    await screen.findByRole("option", { name: "SD 1.5 (ONNX)" });
    fireEvent.click(screen.getByRole("radio", { name: /^video$/i }));

    expect(await screen.findByRole("status")).toHaveTextContent("1.1 s");
  });

  it("warns that longer clips fall apart, only when the clip is actually longer", async () => {
    renderPanel();
    await screen.findByRole("option", { name: "SD 1.5 (ONNX)" });
    fireEvent.click(screen.getByRole("radio", { name: /^video$/i }));

    // Medido con el mismo prompt y el mismo seed: a 17 cuadros el sujeto aguanta
    // hasta el ultimo frame, a 33 se deshace pasada la mitad.
    expect(await screen.findByRole("status")).not.toHaveTextContent(/deformando/i);
    fireEvent.change(screen.getByLabelText(/^cuadros$/i), { target: { value: "33" } });
    expect(screen.getByRole("status")).toHaveTextContent(/deformando/i);
  });

  it("points at the installer when the video pack is not downloaded", async () => {
    renderPanel(AVAILABLE_CAPABILITIES, DEVICES, NO_VIDEO_CAPABILITIES);
    await screen.findByRole("option", { name: "SD 1.5 (ONNX)" });

    fireEvent.click(screen.getByRole("radio", { name: /^video$/i }));

    expect(await screen.findByText(/download-wan-video/i)).toBeInTheDocument();
  });

  it("keeps the chosen model when switching between the two image modes", async () => {
    renderPanel();
    await screen.findByRole("option", { name: "SD 1.5 (ONNX)" });
    fireEvent.change(screen.getByLabelText(/^model$/i), { target: { value: "sd15-onnx" } });

    fireEvent.click(screen.getByRole("radio", { name: /image to image/i }));

    expect(screen.getByLabelText(/^model$/i)).toHaveValue("sd15-onnx");
  });
});
