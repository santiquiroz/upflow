import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render as rtlRender, screen } from "@testing-library/react";
import type { ReactElement, ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { JobQueueEntry } from "../hooks/useJobQueue";
import { LocaleProvider } from "../i18n/LocaleProvider";
import type {
  AudioJob,
  DownloadJob,
  GenerationJob,
  JobStage,
  Shape3dJob,
  TranscribeJob,
  VideoJobResponse,
} from "../lib/apiTypes";
import * as api from "../lib/api";
import * as audioService from "../services/audio";
import { JobDetailModal } from "./JobDetailModal";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, getDevices: vi.fn() };
});

vi.mock("../services/audio", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/audio")>();
  return { ...actual, fetchAudioCapabilities: vi.fn(), fetchVoiceCatalog: vi.fn() };
});

beforeEach(() => {
  // Sin datos (o sin match) el modal cae al id crudo del device, que es lo que
  // afirman los tests historicos.
  vi.mocked(api.getDevices).mockResolvedValue({ devices: [], defaultDeviceId: "dml:0" });
  vi.mocked(audioService.fetchAudioCapabilities).mockResolvedValue({
    denoiseModes: [],
    restoreAvailable: false,
    restoreModes: [],
  });
  vi.mocked(audioService.fetchVoiceCatalog).mockResolvedValue({ steps: [], deliveries: [] });
});

function render(ui: ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <LocaleProvider initialLocale="en">{children}</LocaleProvider>
      </QueryClientProvider>
    );
  }
  return rtlRender(ui, { wrapper: Wrapper });
}

function stage(key: string, label: string, status: JobStage["status"]): JobStage {
  return { key, label, weight: 0.25, status };
}

const BASE_VIDEO_JOB: VideoJobResponse = {
  ownerId: null,
  jobId: "vid-1",
  status: "running",
  originalFilename: "clip.mp4",
  modelName: "realesr-animevideov3-x2",
  scale: 2,
  outputContainer: "mp4",
  videoCodec: "libx264",
  videoPreset: "medium",
  crf: 17,
  keepAudio: true,
  fpsMultiplier: 1,
  targetFps: null,
  audioEnhance: null,
  audioRestore: null,
  interpEngine: "rife",
  modelId: "realesr-animevideov3-x2",
  device: "dml:0",
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  error: null,
  metadata: {},
  progressPct: null,
  downloadUrl: null,
};

function buildEntry(
  overrides: Partial<VideoJobResponse> = {},
  entryOverrides: Partial<JobQueueEntry> = {},
): JobQueueEntry {
  const job: VideoJobResponse = { ...BASE_VIDEO_JOB, ...overrides };
  return {
    id: job.jobId,
    kind: "video",
    fileName: job.originalFilename,
    createdAt: 1,
    status: job.status,
    downloadUrl: job.downloadUrl,
    errorMessage: null,
    job,
    ...entryOverrides,
  };
}

function entryFor(kind: JobQueueEntry["kind"], job: unknown, fileName: string): JobQueueEntry {
  const typed = job as { id: string; status: JobQueueEntry["status"] };
  return {
    id: typed.id,
    kind,
    fileName,
    createdAt: 1,
    status: typed.status,
    downloadUrl: null,
    errorMessage: null,
    job: job as JobQueueEntry["job"],
  };
}

describe("JobDetailModal", () => {
  it("renders the file name and the grouped sections", () => {
    render(<JobDetailModal entry={buildEntry()} onClose={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "clip.mp4" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Parameters" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Progress", level: 3 })).toBeInTheDocument();
    expect(screen.getByText("Video")).toBeInTheDocument();
    expect(screen.getByText("realesr-animevideov3-x2")).toBeInTheDocument();
    expect(screen.getByText("dml:0")).toBeInTheDocument();
    expect(screen.getByText("2x")).toBeInTheDocument();
  });

  it("renders numeric details as tabular mono but leaves text details unstyled", () => {
    render(<JobDetailModal entry={buildEntry()} onClose={vi.fn()} />);

    expect(screen.getByText("2x")).toHaveClass("font-mono-tabular");
    expect(screen.getByText("realesr-animevideov3-x2")).not.toHaveClass("font-mono-tabular");
  });

  it("shows the elapsed duration once the job has finished", () => {
    const entry = buildEntry({
      status: "completed",
      startedAt: "2026-01-01T00:00:00Z",
      finishedAt: "2026-01-01T00:03:12Z",
    });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText("Duration")).toBeInTheDocument();
    expect(screen.getByText("3m 12s")).toBeInTheDocument();
    expect(screen.queryByText("Running for")).not.toBeInTheDocument();
  });

  it("shows how long a running job has been going, and keeps it ticking", () => {
    vi.useFakeTimers();
    vi.setSystemTime(Date.parse("2026-01-01T00:00:10Z"));
    const entry = buildEntry({ status: "running", startedAt: "2026-01-01T00:00:00Z" });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);
    expect(screen.getByText("Running for")).toBeInTheDocument();
    expect(screen.getByText("10s")).toBeInTheDocument();

    // advanceTimersByTime mueve tambien el reloj falso: 10s + 5s = 15s.
    act(() => {
      vi.advanceTimersByTime(5000);
    });

    expect(screen.getByText("15s")).toBeInTheDocument();
    vi.useRealTimers();
  });

  it("renders a vertical stepper translated from the stage keys", () => {
    const entry = buildEntry({
      metadata: {
        stages: [
          stage("probing", "Probing video", "done"),
          stage("upscaling_frames", "Upscaling frames", "active"),
          stage("encoding_video", "Encoding video", "pending"),
        ],
      },
    });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText("Probing video")).toBeInTheDocument();
    expect(screen.getByText("Upscaling frames")).toBeInTheDocument();
    expect(screen.getByText("Encoding video")).toBeInTheDocument();
  });

  it("falls back to the backend label for a stage the catalog does not know", () => {
    const entry = buildEntry({
      metadata: { stages: [stage("quantizing", "Quantizing weights", "active")] },
    });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText("Quantizing weights")).toBeInTheDocument();
    expect(screen.queryByText("job.stage.quantizing")).not.toBeInTheDocument();
  });

  it("shows a determinate progress bar with the percentage when progressPct is available", () => {
    const entry = buildEntry({ progressPct: 42 });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    const bar = screen.getByRole("progressbar", { name: "Progress" });
    expect(bar).toHaveAttribute("aria-valuenow", "42");
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("shows an indeterminate progress bar when progress is not yet available", () => {
    const entry = buildEntry({ progressPct: null });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    const bar = screen.getByRole("progressbar", { name: "Progress" });
    expect(bar).toHaveAttribute("aria-busy", "true");
    expect(bar).not.toHaveAttribute("aria-valuenow");
  });

  it("shows frames X / Y as tabular numbers when frame counts are present", () => {
    const entry = buildEntry({ progressPct: 20, metadata: { framesDone: 120, framesTotal: 600 } });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText(/frames/)).toHaveTextContent("120 / 600 frames");
    expect(screen.getByText("120")).toHaveClass("font-mono-tabular");
    expect(screen.getByText("600")).toHaveClass("font-mono-tabular");
  });

  it("uses interpFramesTotal as the denominator during interpolation so the ratio stays valid", () => {
    const entry = buildEntry({
      progressPct: 90,
      metadata: { stage: "interpolating_frames", framesDone: 800, framesTotal: 400, interpFramesTotal: 800 },
    });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText(/frames/)).toHaveTextContent("800 / 800 frames");
  });

  it("omits the frames readout when framesTotal is unknown (VFR source)", () => {
    const entry = buildEntry({ progressPct: 20, metadata: { framesDone: 120, framesTotal: null } });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.queryByText(/frames/)).not.toBeInTheDocument();
  });

  it("shows the audio enhancement mode when configured", () => {
    const entry = buildEntry({ keepAudio: true, audioEnhance: "deepfilter" });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText("DeepFilterNet")).toBeInTheDocument();
  });

  it("says the audio was dropped when the job did not keep it", () => {
    const entry = buildEntry({ keepAudio: false });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText("Dropped")).toBeInTheDocument();
  });

  it("shows the failure message and hides the progress bar when the job failed", () => {
    const entry = buildEntry(
      { status: "failed", error: "Model crashed" },
      { status: "failed", errorMessage: "Model crashed" },
    );

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Model crashed");
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("keeps a long error inside the modal instead of stretching it", () => {
    const long = "ffmpeg: ".concat("x".repeat(400));
    const entry = buildEntry({ status: "failed" }, { status: "failed", errorMessage: long });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByRole("alert").querySelector("span")).toHaveClass("break-words");
  });

  it("scrolls its body instead of growing past the screen", () => {
    // Un trabajo de video llega a veinte filas: en un celular el modal tiene que
    // scrollear por dentro y dejar el titulo y los botones alcanzables.
    render(<JobDetailModal entry={buildEntry()} onClose={vi.fn()} />);

    const dialog = screen.getByRole("dialog");
    expect(dialog.className).toContain("max-h-[90vh]");
    expect(dialog.querySelector(".overflow-y-auto")).not.toBeNull();
  });

  it("does not blow up when the job has not been fetched yet", () => {
    const entry = buildEntry({}, { job: undefined });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "clip.mp4" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Parameters" })).not.toBeInTheDocument();
  });

  describe("ETA", () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(0);
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it("hides the ETA until a second poll establishes a rate", () => {
      const entry = buildEntry({ progressPct: 20 });

      render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

      expect(screen.queryByText(/ETA/)).not.toBeInTheDocument();
    });

    it("shows the ETA once a steady rate is established across polls", () => {
      const entry = buildEntry({ progressPct: 20 });
      const { rerender } = render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

      vi.setSystemTime(10_000);
      rerender(<JobDetailModal entry={buildEntry({ progressPct: 40 })} onClose={vi.fn()} />);

      expect(screen.getByText(/ETA/)).toBeInTheDocument();
    });

    it("resets its sample buffer when a different job is shown", () => {
      const firstEntry = buildEntry({ progressPct: 20 }, { id: "vid-1" });
      const { rerender } = render(<JobDetailModal entry={firstEntry} onClose={vi.fn()} />);
      vi.setSystemTime(10_000);
      rerender(<JobDetailModal entry={buildEntry({ progressPct: 40 }, { id: "vid-1" })} onClose={vi.fn()} />);
      expect(screen.getByText(/ETA/)).toBeInTheDocument();

      const otherJobEntry = buildEntry({ jobId: "vid-2", progressPct: 5 }, { id: "vid-2" });
      rerender(<JobDetailModal entry={otherJobEntry} onClose={vi.fn()} />);

      expect(screen.queryByText(/ETA/)).not.toBeInTheDocument();
    });
  });

  it("closes when Escape is pressed", () => {
    const onClose = vi.fn();
    render(<JobDetailModal entry={buildEntry()} onClose={onClose} />);

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("closes when the Close button is clicked", () => {
    const onClose = vi.fn();
    render(<JobDetailModal entry={buildEntry()} onClose={onClose} />);

    fireEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("shows a Cancel action for a running job and calls onCancel with the job id", () => {
    const onCancel = vi.fn();
    render(<JobDetailModal entry={buildEntry()} onClose={vi.fn()} onCancel={onCancel} />);

    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));

    expect(onCancel).toHaveBeenCalledWith("vid-1");
  });

  it("hides the Cancel action once the job is terminal", () => {
    const entry = buildEntry({ status: "completed", downloadUrl: "/download" }, { status: "completed" });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} onCancel={vi.fn()} />);

    expect(screen.queryByRole("button", { name: /cancel/i })).not.toBeInTheDocument();
  });

  it("renders the cancelled status and hides the progress bar", () => {
    const entry = buildEntry({ status: "cancelled" }, { status: "cancelled" });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} onCancel={vi.fn()} />);

    expect(screen.getByText("Cancelled")).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /cancel/i })).not.toBeInTheDocument();
  });
});

describe("generation job details", () => {
  it("shows acceleration, resolved seed, device name and pace", async () => {
    vi.mocked(api.getDevices).mockResolvedValue({
      devices: [{ id: "dml:0", kind: "gpu", name: "AMD Radeon RX 7900 XT", backend: "directml" }],
      defaultDeviceId: "dml:0",
    });
    const job: GenerationJob = {
      id: "gen-1",
      status: "completed",
      prompt: "a red apple",
      negativePrompt: null,
      modelId: "gen--m",
      steps: 30,
      guidance: 7,
      width: 1024,
      height: 704,
      seed: 123456,
      seedWasRandom: true,
      device: "dml:0",
      executionProvider: "CPU (fallback)",
      strength: 1,
      autoUpscale: false,
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:00Z",
      finishedAt: "2026-01-01T00:01:00Z",
      progressPct: 100,
      stages: null,
      error: null,
      ownerId: null,
      downloadUrl: null,
    };

    render(<JobDetailModal entry={entryFor("generation", job, "a red apple")} onClose={vi.fn()} />);

    expect(await screen.findByText("AMD Radeon RX 7900 XT (dml:0)")).toBeInTheDocument();
    expect(screen.getByText("CPU (fallback)")).toBeInTheDocument();
    expect(screen.getByText("123456 (random)")).toBeInTheDocument();
    expect(screen.getByText("~2.0 s/step")).toBeInTheDocument();
    expect(screen.getByText("Strength")).toBeInTheDocument();
  });
});

describe("audio job details", () => {
  const AUDIO_JOB: AudioJob = {
    id: "aud-1",
    status: "completed",
    originalFilename: "cancion.mp3",
    denoise: null,
    restore: "audiosr",
    device: null,
    outputFormat: "flac",
    master: "streaming",
    cleanupSteps: ["denoise"],
    createdAt: "2026-01-01T00:00:00Z",
    startedAt: "2026-01-01T00:00:00Z",
    finishedAt: "2026-01-01T00:04:14Z",
    progressPct: 100,
    stages: [
      stage("decoding", "Decoding audio", "done"),
      stage("restoring", "Restoring", "done"),
      stage("mastering", "Mastering", "done"),
      stage("finalizing", "Writing output", "done"),
    ],
    metadata: { loudnessBefore: -21.53, loudnessTarget: -14 },
    error: null,
    ownerId: null,
    downloadUrl: "/download",
  };

  it("shows the chain the user picked, the finishing preset and the loudness move", async () => {
    render(<JobDetailModal entry={entryFor("audio", AUDIO_JOB, "cancion.mp3")} onClose={vi.fn()} />);

    expect(await screen.findByText("AudioSR")).toBeInTheDocument();
    expect(screen.getByText("Cleanup chain")).toBeInTheDocument();
    expect(screen.getByText("Mastering", { selector: "dt" })).toBeInTheDocument();
    expect(screen.getByText("FLAC")).toBeInTheDocument();
    expect(screen.getByText("-21.5 → -14.0 LUFS")).toBeInTheDocument();
  });

  it("shows the effective device even though the job did not pin one", async () => {
    render(<JobDetailModal entry={entryFor("audio", AUDIO_JOB, "cancion.mp3")} onClose={vi.fn()} />);

    expect(await screen.findByText("dml:0 (by default)")).toBeInTheDocument();
  });

  it("hides the video-only rows", () => {
    render(<JobDetailModal entry={entryFor("audio", AUDIO_JOB, "cancion.mp3")} onClose={vi.fn()} />);

    expect(screen.queryByText("Container")).not.toBeInTheDocument();
    expect(screen.queryByText("Scale")).not.toBeInTheDocument();
  });

  it("translates a cleanup pass keeping the model's proper name", () => {
    const job: AudioJob = {
      ...AUDIO_JOB,
      stages: [stage("cleanup_denoise", "Cleaning up: UVR DeNoise by FoxJoy", "active")],
    };

    render(<JobDetailModal entry={entryFor("audio", job, "cancion.mp3")} onClose={vi.fn()} />);

    expect(screen.getByText("Cleaning up: UVR DeNoise by FoxJoy")).toBeInTheDocument();
  });
});

describe("transcribe job details", () => {
  const TRANSCRIBE_JOB: TranscribeJob = {
    id: "tr-1",
    status: "completed",
    originalFilename: "charla.mp4",
    modelId: "whisper-small",
    language: null,
    device: null,
    outputMode: "dubbed_video",
    targetLanguage: "en",
    createdAt: "2026-01-01T00:00:00Z",
    startedAt: "2026-01-01T00:00:00Z",
    finishedAt: "2026-01-01T00:02:00Z",
    progressPct: 100,
    text: "hola",
    error: null,
    ownerId: null,
    downloadUrl: null,
  };

  it("shows the model, the output mode and the dubbing language", () => {
    render(<JobDetailModal entry={entryFor("transcribe", TRANSCRIBE_JOB, "charla.mp4")} onClose={vi.fn()} />);

    expect(screen.getByText("Transcription")).toBeInTheDocument();
    expect(screen.getByText("whisper-small")).toBeInTheDocument();
    expect(screen.getByText("Dubbed video")).toBeInTheDocument();
    expect(screen.getByText("Dubbing language")).toBeInTheDocument();
  });

  it("hides the dubbing language on a plain transcription", () => {
    const job: TranscribeJob = { ...TRANSCRIBE_JOB, outputMode: "text", targetLanguage: null };

    render(<JobDetailModal entry={entryFor("transcribe", job, "charla.mp4")} onClose={vi.fn()} />);

    expect(screen.queryByText("Dubbing language")).not.toBeInTheDocument();
  });
});

describe("download job details", () => {
  const DOWNLOAD_JOB: DownloadJob = {
    id: "dl-1",
    status: "running",
    url: "https://example.com/watch?v=1",
    maxHeight: 1080,
    audioOnly: false,
    audioFormat: "mp3",
    audioBitrateKbps: null,
    videoContainer: "mp4",
    mediaTitle: "Un video",
    mediaUploader: "Alguien",
    extractor: "youtube",
    createdAt: "2026-01-01T00:00:00Z",
    startedAt: "2026-01-01T00:00:00Z",
    finishedAt: null,
    progressPct: 25,
    downloadedBytes: 1024 * 1024,
    totalBytes: 4 * 1024 * 1024,
    outputFiles: [],
    outputDirectory: "",
    error: null,
    ownerId: null,
    thenSeparate: false,
    followupJobIds: [],
    followupError: null,
  };

  it("shows where the media came from and how much has arrived", () => {
    render(<JobDetailModal entry={entryFor("download", DOWNLOAD_JOB, "Un video")} onClose={vi.fn()} />);

    expect(screen.getByText("Download")).toBeInTheDocument();
    // El titulo es tambien el nombre visible en la cola: aparece en el encabezado
    // y en la fila de Titulo.
    expect(screen.getByText("Un video", { selector: "dd" })).toBeInTheDocument();
    expect(screen.getByText("youtube")).toBeInTheDocument();
    expect(screen.getByText("1.0 MB / 4.0 MB")).toBeInTheDocument();
  });

  it("hides the produced files until there are any", () => {
    render(<JobDetailModal entry={entryFor("download", DOWNLOAD_JOB, "Un video")} onClose={vi.fn()} />);

    expect(screen.queryByText("Files")).not.toBeInTheDocument();
  });
});

describe("3D job details", () => {
  const SHAPE3D_JOB: Shape3dJob = {
    id: "3d-1",
    status: "completed",
    prompt: "una maceta",
    printer: "ender-3",
    source: "mesh",
    code: null,
    retries: 0,
    targetMm: 80,
    targetMmSource: "estimate",
    targetMmReference: "una taza",
    createdAt: "2026-01-01T00:00:00Z",
    startedAt: "2026-01-01T00:00:00Z",
    finishedAt: "2026-01-01T00:03:00Z",
    canPrint: true,
    sizeMm: [80, 40, 40],
    triangleCount: 4200,
    blockers: [],
    advice: [],
    error: null,
    downloadUrl: "/download",
  };

  it("shows the printer, the measurement and where the measurement came from", () => {
    render(<JobDetailModal entry={entryFor("shape3d", SHAPE3D_JOB, "una maceta")} onClose={vi.fn()} />);

    expect(screen.getByText("3D model")).toBeInTheDocument();
    expect(screen.getByText("ender-3")).toBeInTheDocument();
    expect(
      screen.getByText("80 mm · the model suggested it from una taza and you confirmed it"),
    ).toBeInTheDocument();
  });

  it("shows the print verdict", () => {
    render(<JobDetailModal entry={entryFor("shape3d", SHAPE3D_JOB, "una maceta")} onClose={vi.fn()} />);

    expect(screen.getByText("Ready to print")).toBeInTheDocument();
  });

  it("renders without a progress percentage, since the 3D lane reports none", () => {
    render(<JobDetailModal entry={entryFor("shape3d", SHAPE3D_JOB, "una maceta")} onClose={vi.fn()} />);

    const bar = screen.getByRole("progressbar", { name: "Progress" });
    expect(bar).toHaveAttribute("aria-busy", "true");
  });

  it("hides the CAD code on the mesh lane and shows it on the CAD lane", () => {
    const { rerender } = render(
      <JobDetailModal entry={entryFor("shape3d", SHAPE3D_JOB, "una maceta")} onClose={vi.fn()} />,
    );
    expect(screen.queryByText("CAD code")).not.toBeInTheDocument();

    const cad: Shape3dJob = { ...SHAPE3D_JOB, source: "cad", code: "cube([10,10,10]);" };
    rerender(<JobDetailModal entry={entryFor("shape3d", cad, "una maceta")} onClose={vi.fn()} />);

    expect(screen.getByText("CAD code")).toBeInTheDocument();
    expect(screen.getByText("cube([10,10,10]);")).toBeInTheDocument();
  });
});
