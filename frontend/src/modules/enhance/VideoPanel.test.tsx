import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type {
  CreateJobResponse,
  DevicesResponse,
  EngineInfoResponse,
  ModelsResponse,
  VideoJobResponse,
  VideoProfileResponse,
} from "../../lib/apiTypes";
import * as audioService from "../../services/audio";
import { VideoPanel } from "./VideoPanel";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return {
    ...actual,
    getModels: vi.fn(),
    getDevices: vi.fn(),
    getEngineInfo: vi.fn(),
    getVideoCapabilities: vi.fn(),
    createVideoJob: vi.fn(),
    getVideoJob: vi.fn(),
  };
});

vi.mock("../../services/audio", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/audio")>();
  return { ...actual, fetchAudioCapabilities: vi.fn() };
});

const GENERAL_PROFILE: VideoProfileResponse = {
  key: "general-balanced-4x",
  label: "General Balanced 4x",
  category: "general",
  description: "Good default for long videos.",
  modelKey: "realesrgan-x4plus",
  scale: 4,
  videoCodec: "libx264",
  videoPreset: "medium",
  crf: 18,
  keepAudio: true,
};

const ANIME_PROFILE: VideoProfileResponse = {
  key: "anime-balanced-2x",
  label: "Anime Balanced 2x",
  category: "anime",
  description: "Best starting point for anime episodes.",
  modelKey: "realesr-animevideov3-x2",
  scale: 2,
  videoCodec: "libx265",
  videoPreset: "slow",
  crf: 16,
  keepAudio: true,
};

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
    {
      id: "realesr-animevideov3-x2",
      name: "RealESR AnimeVideoV3 x2",
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

const CPU_ONLY_DEVICES: DevicesResponse = {
  devices: [{ id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" }],
  defaultDeviceId: "cpu",
};

const ENGINE_INFO: EngineInfoResponse = {
  engine: "realesrgan-ncnn",
  configuredBinary: "vendor/realesrgan/realesrgan-ncnn-vulkan.exe",
  configuredModelsDir: "vendor/realesrgan/models",
  available: true,
  defaultModel: "realesrgan-x4plus",
  allowedScales: [2, 3, 4],
  supportedModels: [],
  videoProfiles: [GENERAL_PROFILE, ANIME_PROFILE],
  ffmpegAvailable: true,
};

function renderPanel(
  devices: DevicesResponse = DEVICES,
  restoreAvailable = false,
  interpEngines: string[] = ["rife"],
) {
  vi.mocked(api.getModels).mockResolvedValue(MODELS);
  vi.mocked(api.getDevices).mockResolvedValue(devices);
  vi.mocked(api.getEngineInfo).mockResolvedValue(ENGINE_INFO);
  vi.mocked(api.getVideoCapabilities).mockResolvedValue({ interpEngines });
  vi.mocked(audioService.fetchAudioCapabilities).mockResolvedValue({
    denoiseModes: ["deepfilter", "rnnoise"],
    restoreAvailable,
    restoreModes: restoreAvailable ? ["apollo", "audiosr"] : [],
  });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<VideoPanel />, { wrapper: Wrapper });
}

function makeFile(): File {
  return new File(["binary"], "clip.mp4", { type: "video/mp4" });
}

async function selectFile() {
  const fileInput = document.getElementById("video-file-input") as HTMLInputElement;
  fireEvent.change(fileInput, { target: { files: [makeFile()] } });
  await screen.findByRole("radio", { name: /General Balanced 4x/ });
}

function openSection(title: string) {
  const toggle = screen.getByRole("button", { name: new RegExp(`^${title}`) });
  fireEvent.click(toggle);
}

afterEach(() => {
  vi.mocked(api.getModels).mockReset();
  vi.mocked(api.getDevices).mockReset();
  vi.mocked(api.getEngineInfo).mockReset();
  vi.mocked(api.getVideoCapabilities).mockReset();
  vi.mocked(api.createVideoJob).mockReset();
  vi.mocked(api.getVideoJob).mockReset();
  vi.mocked(audioService.fetchAudioCapabilities).mockReset();
});

describe("VideoPanel", () => {
  it("keeps the Profile section expanded by default while the rest start collapsed", async () => {
    renderPanel();

    expect(await screen.findByRole("button", { name: /^Profile/ })).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("button", { name: /^Model/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: /^Device/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: /^Runtime/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: /^FPS boost/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: /^Audio/ })).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("button", { name: /^Advanced/ })).toHaveAttribute("aria-expanded", "false");
  });

  it("shows a placeholder Profile summary until one is picked, then reflects the selection", async () => {
    renderPanel();

    expect(await screen.findByRole("button", { name: /^Profile/ })).toHaveTextContent("Select a profile…");

    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));

    expect(screen.getByRole("button", { name: /^Profile/ })).toHaveTextContent("General Balanced 4x");
  });

  it("reflects the profile-driven model, FPS boost and audio choices in their collapsed summaries", async () => {
    renderPanel();
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /Anime Balanced 2x/ }));

    expect(await screen.findByRole("button", { name: /^Model/ })).toHaveTextContent("RealESR AnimeVideoV3 x2");
    expect(screen.getByRole("button", { name: /^FPS boost/ })).toHaveTextContent("Off");
    expect(screen.getByRole("button", { name: /^Audio/ })).toHaveTextContent("Kept");

    openSection("FPS boost");
    fireEvent.click(screen.getByRole("button", { name: "3×" }));
    expect(screen.getByRole("button", { name: /^FPS boost/ })).toHaveTextContent("3×");
  });

  it("hides the Advanced options from the accessibility tree until the section is expanded", async () => {
    renderPanel();
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));

    expect(screen.queryByRole("spinbutton", { name: /crf/i })).not.toBeInTheDocument();

    openSection("Advanced");

    expect(await screen.findByRole("spinbutton", { name: /crf/i })).toBeInTheDocument();
  });

  it("disables the Upscale CTA until a file and a profile are selected", async () => {
    renderPanel();

    const submitButton = screen.getByRole("button", { name: /upscale video/i });
    expect(submitButton).toBeDisabled();

    await selectFile();
    const profileRadio = await screen.findByRole("radio", { name: /General Balanced 4x/ });
    fireEvent.click(profileRadio);

    await waitFor(() => expect(submitButton).not.toBeDisabled());
  });

  it("auto-selects the profile's model and applies its advanced defaults", async () => {
    renderPanel();
    await selectFile();

    const profileRadio = await screen.findByRole("radio", { name: /Anime Balanced 2x/ });
    fireEvent.click(profileRadio);

    openSection("Model");
    const modelRadio = await screen.findByRole("radio", { name: /RealESR AnimeVideoV3 x2/ });
    await waitFor(() => expect(modelRadio).toBeChecked());

    openSection("Advanced");
    expect(await screen.findByRole("spinbutton", { name: /crf/i })).toHaveValue(16);
  });

  it("keeps a manual ModelPicker override after the profile's default model was auto-applied", async () => {
    renderPanel();
    await selectFile();

    fireEvent.click(await screen.findByRole("radio", { name: /Anime Balanced 2x/ }));
    openSection("Model");
    const animeModelRadio = await screen.findByRole("radio", { name: /RealESR AnimeVideoV3 x2/ });
    await waitFor(() => expect(animeModelRadio).toBeChecked());

    const generalModelRadio = await screen.findByRole("radio", { name: /RealESRGAN x4plus/ });
    fireEvent.click(generalModelRadio);

    await waitFor(() => expect(generalModelRadio).toBeChecked());
    expect(animeModelRadio).not.toBeChecked();
  });

  it("keeps the CTA disabled with a clear hint when the profile's model needs a GPU but only cpu exists", async () => {
    renderPanel(CPU_ONLY_DEVICES);
    await selectFile();

    const profileRadio = await screen.findByRole("radio", { name: /General Balanced 4x/ });
    fireEvent.click(profileRadio);

    const submitButton = screen.getByRole("button", { name: /upscale video/i });
    expect(await screen.findByRole("status")).toHaveTextContent(/requires a Vulkan GPU/i);
    expect(submitButton).toBeDisabled();
    expect(vi.mocked(api.createVideoJob)).not.toHaveBeenCalled();
  });

  it("disables audio enhance until keep_audio is on, and re-enables it once toggled on", async () => {
    renderPanel();
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));
    openSection("Audio");

    const keepAudioToggle = await screen.findByRole("checkbox", { name: /keep original audio/i });
    expect(keepAudioToggle).toBeChecked();
    expect(screen.getByRole("button", { name: "RNNoise" })).not.toBeDisabled();

    fireEvent.click(keepAudioToggle);
    expect(screen.getByRole("button", { name: "RNNoise" })).toBeDisabled();

    fireEvent.click(keepAudioToggle);
    expect(screen.getByRole("button", { name: "RNNoise" })).not.toBeDisabled();
  });

  it("submits the job with the mutually-exclusive FPS boost mode and shows the normalized outputFps on completion", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "vid-1",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-1",
      downloadUrl: null,
    };
    const completedJob: VideoJobResponse = {
      jobId: "vid-1",
      status: "completed",
      originalFilename: "clip.mp4",
      modelName: "realesrgan-x4plus",
      scale: 4,
      outputContainer: "mp4",
      videoCodec: "libx264",
      videoPreset: "medium",
      crf: 18,
      keepAudio: true,
      fpsMultiplier: 1,
      targetFps: "60000/1001",
      audioEnhance: null,
      audioRestore: null,
      interpEngine: "rife",
      modelId: "realesrgan-x4plus",
      device: "dml:0",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:01Z",
      finishedAt: "2026-01-01T00:00:05Z",
      error: null,
      metadata: { outputFps: "24000/1001" },
      progressPct: null,
      downloadUrl: "/api/v1/video/jobs/vid-1/download",
    };
    vi.mocked(api.createVideoJob).mockResolvedValue(createResponse);
    vi.mocked(api.getVideoJob).mockResolvedValue(completedJob);

    renderPanel();
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));
    openSection("FPS boost");

    fireEvent.click(screen.getByRole("button", { name: "59.94 fps" }));
    expect(screen.getByRole("button", { name: "2×" })).toBeDisabled();

    const submitButton = await screen.findByRole("button", { name: /upscale video/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    expect(await screen.findByRole("link", { name: /download/i })).toHaveAttribute(
      "href",
      "/api/v1/video/jobs/vid-1/download",
    );
    expect(screen.getByText("23.98")).toBeInTheDocument();
    expect(vi.mocked(api.createVideoJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ fpsMultiplier: 1, targetFps: "60000/1001" }),
    );
  });

  it("submits device='auto' when the Auto device option is selected", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "vid-2",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-2",
      downloadUrl: null,
    };
    vi.mocked(api.createVideoJob).mockResolvedValue(createResponse);
    vi.mocked(api.getVideoJob).mockResolvedValue({
      jobId: "vid-2",
      status: "queued",
      originalFilename: "clip.mp4",
      modelName: "realesrgan-x4plus",
      scale: 4,
      outputContainer: "mp4",
      videoCodec: "libx264",
      videoPreset: "medium",
      crf: 18,
      keepAudio: true,
      fpsMultiplier: 1,
      targetFps: null,
      audioEnhance: null,
      audioRestore: null,
      interpEngine: "rife",
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
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));

    openSection("Device");
    const autoRadio = await screen.findByRole("radio", { name: /Auto/ });
    fireEvent.click(autoRadio);

    const submitButton = await screen.findByRole("button", { name: /upscale video/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(vi.mocked(api.createVideoJob)).toHaveBeenCalled());
    expect(vi.mocked(api.createVideoJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ device: "auto" }),
    );
  });

  it("hides the Apollo restore control when restore is unavailable", async () => {
    renderPanel(DEVICES, false);
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));
    openSection("Audio");

    await screen.findByRole("checkbox", { name: /keep original audio/i });
    expect(screen.queryByRole("checkbox", { name: /restore compression/i })).not.toBeInTheDocument();
  });

  it("shows the Apollo restore control with keep_audio + restore available and sends audio_restore on submit", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "vid-3",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-3",
      downloadUrl: null,
    };
    vi.mocked(api.createVideoJob).mockResolvedValue(createResponse);
    vi.mocked(api.getVideoJob).mockResolvedValue({
      jobId: "vid-3",
      status: "queued",
      originalFilename: "clip.mp4",
      modelName: "realesrgan-x4plus",
      scale: 4,
      outputContainer: "mp4",
      videoCodec: "libx264",
      videoPreset: "medium",
      crf: 18,
      keepAudio: true,
      fpsMultiplier: 1,
      targetFps: null,
      audioEnhance: null,
      audioRestore: "apollo",
      interpEngine: "rife",
      modelId: "realesrgan-x4plus",
      device: "dml:0",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: null,
      finishedAt: null,
      error: null,
      metadata: {},
      progressPct: null,
      downloadUrl: null,
    });

    renderPanel(DEVICES, true);
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));
    openSection("Audio");

    const apolloRadio = await screen.findByRole("radio", { name: "Apollo" });
    fireEvent.click(apolloRadio);

    const submitButton = await screen.findByRole("button", { name: /upscale video/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(vi.mocked(api.createVideoJob)).toHaveBeenCalled());
    expect(vi.mocked(api.createVideoJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ audioRestore: "apollo" }),
    );
  });

  it("offers AudioSR as a restore mode and shows its cost hint", async () => {
    renderPanel(DEVICES, true);
    await selectFile();
    openSection("Audio");

    const audiosrRadio = await screen.findByRole("radio", { name: "AudioSR" });
    fireEvent.click(audiosrRadio);

    expect(await screen.findByText(/2 minutes of processing per minute/i)).toBeInTheDocument();
  });

  it("defaults the Runtime to Auto and lists the three backend options", async () => {
    renderPanel();
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));

    expect(screen.getByRole("button", { name: /^Runtime/ })).toHaveTextContent("Auto");

    openSection("Runtime");
    expect(await screen.findByRole("radio", { name: /Auto/ })).toBeChecked();
    expect(screen.getByRole("radio", { name: /NCNN Vulkan/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /ONNX DirectML/ })).toBeInTheDocument();
  });

  it("sends the chosen backend runtime and reflects it in the section summary", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "vid-rt",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-rt",
      downloadUrl: null,
    };
    vi.mocked(api.createVideoJob).mockResolvedValue(createResponse);
    vi.mocked(api.getVideoJob).mockResolvedValue({
      jobId: "vid-rt",
      status: "queued",
      originalFilename: "clip.mp4",
      modelName: "realesrgan-x4plus",
      scale: 4,
      outputContainer: "mp4",
      videoCodec: "libx264",
      videoPreset: "medium",
      crf: 18,
      keepAudio: true,
      fpsMultiplier: 1,
      targetFps: null,
      audioEnhance: null,
      audioRestore: null,
      interpEngine: "rife",
      backend: "onnx",
      modelId: "realesrgan-x4plus",
      device: "dml:0",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: null,
      finishedAt: null,
      error: null,
      metadata: {},
      progressPct: null,
      downloadUrl: null,
    });

    renderPanel();
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));

    openSection("Runtime");
    fireEvent.click(await screen.findByRole("radio", { name: /ONNX DirectML/ }));
    expect(screen.getByRole("button", { name: /^Runtime/ })).toHaveTextContent("ONNX DirectML");

    const submitButton = await screen.findByRole("button", { name: /upscale video/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(vi.mocked(api.createVideoJob)).toHaveBeenCalled());
    expect(vi.mocked(api.createVideoJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ backend: "onnx" }),
    );
  });

  it("hides the interpolation engine selector when only one engine is available", async () => {
    renderPanel(DEVICES, false, ["rife"]);
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));
    openSection("FPS boost");
    fireEvent.click(screen.getByRole("button", { name: "3×" }));

    expect(screen.queryByRole("group", { name: "Interpolation engine" })).not.toBeInTheDocument();
  });

  it("hides the interpolation engine selector until FPS boost is actually active", async () => {
    renderPanel(DEVICES, false, ["rife", "gmfss"]);
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));
    openSection("FPS boost");

    expect(screen.queryByRole("group", { name: "Interpolation engine" })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "3×" }));

    expect(await screen.findByRole("group", { name: "Interpolation engine" })).toBeInTheDocument();
  });

  it("submits the selected GMFSS engine and shows its very-slow cost hint", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "vid-gmfss",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-gmfss",
      downloadUrl: null,
    };
    vi.mocked(api.createVideoJob).mockResolvedValue(createResponse);
    vi.mocked(api.getVideoJob).mockResolvedValue({
      jobId: "vid-gmfss",
      status: "queued",
      originalFilename: "clip.mp4",
      modelName: "realesrgan-x4plus",
      scale: 4,
      outputContainer: "mp4",
      videoCodec: "libx264",
      videoPreset: "medium",
      crf: 18,
      keepAudio: true,
      fpsMultiplier: 2,
      targetFps: null,
      audioEnhance: null,
      audioRestore: null,
      interpEngine: "gmfss",
      modelId: "realesrgan-x4plus",
      device: "dml:0",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: null,
      finishedAt: null,
      error: null,
      metadata: {},
      progressPct: null,
      downloadUrl: null,
    });

    renderPanel(DEVICES, false, ["rife", "gmfss"]);
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));
    openSection("FPS boost");
    fireEvent.click(screen.getByRole("button", { name: "2×" }));

    fireEvent.click(await screen.findByRole("button", { name: /GMFSS/ }));
    expect(await screen.findByText(/10x or more/i)).toBeInTheDocument();

    const submitButton = await screen.findByRole("button", { name: /upscale video/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(vi.mocked(api.createVideoJob)).toHaveBeenCalled());
    expect(vi.mocked(api.createVideoJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ interpEngine: "gmfss" }),
    );
  });

  it("always submits rife when FPS boost is off, even with multiple engines available", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "vid-off",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-off",
      downloadUrl: null,
    };
    vi.mocked(api.createVideoJob).mockResolvedValue(createResponse);
    vi.mocked(api.getVideoJob).mockResolvedValue({
      jobId: "vid-off",
      status: "queued",
      originalFilename: "clip.mp4",
      modelName: "realesrgan-x4plus",
      scale: 4,
      outputContainer: "mp4",
      videoCodec: "libx264",
      videoPreset: "medium",
      crf: 18,
      keepAudio: true,
      fpsMultiplier: 1,
      targetFps: null,
      audioEnhance: null,
      audioRestore: null,
      interpEngine: "rife",
      modelId: "realesrgan-x4plus",
      device: "dml:0",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: null,
      finishedAt: null,
      error: null,
      metadata: {},
      progressPct: null,
      downloadUrl: null,
    });

    renderPanel(DEVICES, false, ["rife", "gmfss"]);
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));

    const submitButton = await screen.findByRole("button", { name: /upscale video/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    await waitFor(() => expect(vi.mocked(api.createVideoJob)).toHaveBeenCalled());
    expect(vi.mocked(api.createVideoJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({ interpEngine: "rife" }),
    );
  });

  it("shows an inline error message when the server rejects the upload", async () => {
    vi.mocked(api.createVideoJob).mockRejectedValue(new Error("Video queue is full; try again later"));

    renderPanel();
    await selectFile();
    fireEvent.click(await screen.findByRole("radio", { name: /General Balanced 4x/ }));

    const submitButton = await screen.findByRole("button", { name: /upscale video/i });
    await waitFor(() => expect(submitButton).not.toBeDisabled());
    fireEvent.click(submitButton);

    expect(await screen.findByRole("alert")).toHaveTextContent("Video queue is full; try again later");
  });
});
