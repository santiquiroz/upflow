import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import type { GenerationJob, JobResponse, VideoJobResponse } from "../lib/apiTypes";
import { createJobQueueStore } from "../lib/jobQueueStore";
import * as audioService from "../services/audio";
import * as downloadService from "../services/download";
import * as generationService from "../services/generation";
import * as printService from "../services/print";
import * as transcribeService from "../services/transcribe";
import { useJobQueue } from "./useJobQueue";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, getJob: vi.fn(), getVideoJob: vi.fn() };
});

vi.mock("../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/generation")>();
  return { ...actual, getGenerationJob: vi.fn(), cancelGenerationJob: vi.fn() };
});

vi.mock("../services/transcribe", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/transcribe")>();
  return { ...actual, getTranscribeJob: vi.fn(), cancelTranscribeJob: vi.fn() };
});

vi.mock("../services/audio", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/audio")>();
  return { ...actual, getAudioJob: vi.fn(), cancelAudioJob: vi.fn() };
});

vi.mock("../services/download", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/download")>();
  return { ...actual, getDownloadJob: vi.fn(), cancelDownloadJob: vi.fn() };
});

vi.mock("../services/print", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/print")>();
  return { ...actual, getShape3dJob: vi.fn(), cancelShape3dJob: vi.fn() };
});

const BASE_AUDIO_JOB = {
  id: "au-1",
  status: "queued" as const,
  originalFilename: "un-tema.mp3",
  denoise: null,
  restore: null,
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  progressPct: null,
  stages: null,
  error: null,
  ownerId: null,
  downloadUrl: null,
};

const DESCARGA_TERMINADA = {
  id: "dl-1",
  status: "completed" as const,
  url: "https://example.com/x",
  maxHeight: 1080,
  audioOnly: true,
  audioFormat: "mp3",
  audioBitrateKbps: null,
  videoContainer: "mp4",
  mediaTitle: "Un tema",
  mediaUploader: null,
  extractor: "youtube",
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: "2026-01-01T00:00:00Z",
  finishedAt: "2026-01-01T00:01:00Z",
  progressPct: 100,
  downloadedBytes: 100,
  totalBytes: 100,
  outputFiles: ["un-tema.mp3"],
  outputDirectory: "C:/uploads",
  error: null,
  ownerId: null,
  thenSeparate: true,
  followupJobIds: [] as string[],
  followupError: null as string | null,
};

const POLL_INTERVAL_MS = 10;

const BASE_IMAGE_JOB: JobResponse = {
  ownerId: null,
  jobId: "img-1",
  status: "queued",
  originalFilename: "photo.png",
  modelName: "realesrgan-x4plus",
  scale: 4,
  outputFormat: "png",
  modelId: "realesrgan-x4plus",
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  error: null,
  metadata: {},
  progressPct: null,
  downloadUrl: null,
};

const BASE_VIDEO_JOB: VideoJobResponse = {
  ownerId: null,
  jobId: "vid-1",
  status: "queued",
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
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  error: null,
  metadata: {},
  progressPct: null,
  downloadUrl: null,
};

const BASE_GENERATION_JOB: GenerationJob = {
  ownerId: null,
  id: "g1",
  status: "completed",
  prompt: "a red apple",
  negativePrompt: null,
  modelId: "gen--amd--sd15",
  steps: 25,
  guidance: 7.5,
  width: 512,
  height: 512,
  seed: 42,
  device: "dml:0",
  autoUpscale: false,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: "2026-01-01T00:00:01Z",
  finishedAt: "2026-01-01T00:00:05Z",
  progressPct: 100,
  stages: null,
  error: null,
  downloadUrl: "/api/v1/generation/jobs/g1/download",
};

function createWrapper() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
}

afterEach(() => {
  vi.mocked(api.getJob).mockReset();
  vi.mocked(api.getVideoJob).mockReset();
  vi.mocked(generationService.getGenerationJob).mockReset();
  vi.mocked(generationService.cancelGenerationJob).mockReset();
  vi.mocked(transcribeService.getTranscribeJob).mockReset();
  vi.mocked(downloadService.getDownloadJob).mockReset();
  vi.mocked(printService.getShape3dJob).mockReset();
});

describe("useJobQueue", () => {
  it("starts empty when no jobs were tracked", () => {
    const store = createJobQueueStore();

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });

    expect(result.current.entries).toEqual([]);
  });

  it("mete en la cola los trabajos que encadeno una descarga", async () => {
    const store = createJobQueueStore();
    store.addTrackedJob({ id: "dl-1", kind: "download", fileName: "un-tema", createdAt: 1 });
    vi.mocked(downloadService.getDownloadJob).mockResolvedValue({
      ...DESCARGA_TERMINADA,
      followupJobIds: ["au-1"],
    });
    vi.mocked(audioService.getAudioJob).mockResolvedValue(BASE_AUDIO_JOB);

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });

    // Nadie pidio ese trabajo por pantalla: sin registrarlo, las pistas salen en
    // silencio y no hay donde ver el progreso ni bajarlas.
    await waitFor(() => expect(result.current.entries).toHaveLength(2));
    const encadenado = result.current.entries.find((entry) => entry.id === "au-1");
    expect(encadenado?.kind).toBe("audio");
    // El nombre sale del archivo bajado, no del id.
    expect(encadenado?.fileName).toBe("un-tema.mp3");
  });

  it("no duplica el encadenado aunque el sondeo lo repita", async () => {
    const store = createJobQueueStore();
    store.addTrackedJob({ id: "dl-1", kind: "download", fileName: "un-tema", createdAt: 1 });
    vi.mocked(downloadService.getDownloadJob).mockResolvedValue({
      ...DESCARGA_TERMINADA,
      followupJobIds: ["au-1"],
    });
    vi.mocked(audioService.getAudioJob).mockResolvedValue({
      ...BASE_AUDIO_JOB,
      status: "running" as const,
    });

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });
    await waitFor(() => expect(result.current.entries).toHaveLength(2));
    // El id vuelve a llegar en CADA respuesta hasta que el usuario se va: la
    // guarda tiene que aguantar los sondeos, no solo la primera vez.
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS * 5));

    expect(result.current.entries.filter((entry) => entry.id === "au-1")).toHaveLength(1);
  });

  it("una descarga sin encadenado no agrega nada", async () => {
    const store = createJobQueueStore();
    store.addTrackedJob({ id: "dl-1", kind: "download", fileName: "un-tema", createdAt: 1 });
    vi.mocked(downloadService.getDownloadJob).mockResolvedValue(DESCARGA_TERMINADA);

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.entries[0]?.status).toBe("completed"));
    expect(result.current.entries).toHaveLength(1);
  });

  it("aggregates image and video jobs ordered newest first", async () => {
    const store = createJobQueueStore();
    vi.mocked(api.getJob).mockResolvedValue(BASE_IMAGE_JOB);
    vi.mocked(api.getVideoJob).mockResolvedValue(BASE_VIDEO_JOB);

    store.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });
    store.addTrackedJob({ id: "vid-1", kind: "video", fileName: "clip.mp4", createdAt: 2 });

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.entries).toHaveLength(2));
    expect(result.current.entries.map((entry) => entry.id)).toEqual(["vid-1", "img-1"]);
    expect(result.current.entries.map((entry) => entry.kind)).toEqual(["video", "image"]);
  });

  it("aggregates a completed generation job with its download URL", async () => {
    const store = createJobQueueStore();
    vi.mocked(generationService.getGenerationJob).mockResolvedValue(BASE_GENERATION_JOB);

    store.addTrackedJob({ id: "g1", kind: "generation", fileName: "a red apple", createdAt: 1 });

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.entries[0]?.status).toBe("completed"));
    expect(result.current.entries[0]).toMatchObject({
      id: "g1",
      kind: "generation",
      downloadUrl: "/api/v1/generation/jobs/g1/download",
    });
  });

  it("reports live status and stops polling a terminal job", async () => {
    const store = createJobQueueStore();
    vi.mocked(api.getJob)
      .mockResolvedValueOnce({ ...BASE_IMAGE_JOB, status: "running" })
      .mockResolvedValue({ ...BASE_IMAGE_JOB, status: "completed", downloadUrl: "/api/v1/jobs/img-1/download" });

    store.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.entries[0]?.status).toBe("completed"));
    expect(result.current.entries[0]?.downloadUrl).toBe("/api/v1/jobs/img-1/download");

    const callsAtCompletion = vi.mocked(api.getJob).mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS * 5));
    expect(vi.mocked(api.getJob).mock.calls.length).toBe(callsAtCompletion);
  });

  it("surfaces the job error message for a failed job", async () => {
    const store = createJobQueueStore();
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_IMAGE_JOB, status: "failed", error: "Model crashed" });

    store.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.entries[0]?.status).toBe("failed"));
    expect(result.current.entries[0]?.errorMessage).toBe("Model crashed");
  });

  it("dismisses a job by removing it from the store", async () => {
    const store = createJobQueueStore();
    vi.mocked(api.getJob).mockResolvedValue(BASE_IMAGE_JOB);
    store.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });
    await waitFor(() => expect(result.current.entries).toHaveLength(1));

    act(() => result.current.dismiss("img-1"));

    expect(result.current.entries).toHaveLength(0);
  });

  it("tracks the transcribe, download and 3D families too", async () => {
    const store = createJobQueueStore();
    vi.mocked(transcribeService.getTranscribeJob).mockResolvedValue({
      id: "tr-1",
      status: "running",
      originalFilename: "charla.mp4",
      modelId: "whisper-small",
      language: null,
      device: null,
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:00Z",
      finishedAt: null,
      progressPct: 10,
      text: null,
      error: null,
      ownerId: null,
      downloadUrl: null,
    });
    vi.mocked(downloadService.getDownloadJob).mockResolvedValue({
      id: "dl-1",
      status: "running",
      url: "https://example.com/x",
      maxHeight: 1080,
      audioOnly: false,
      audioFormat: "mp3",
      audioBitrateKbps: null,
      videoContainer: "mp4",
      mediaTitle: "Un video",
      mediaUploader: null,
      extractor: "youtube",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:00Z",
      finishedAt: null,
      progressPct: 25,
      downloadedBytes: 10,
      totalBytes: 100,
      outputFiles: [],
      outputDirectory: "",
      error: null,
      ownerId: null,
      thenSeparate: false,
      followupJobIds: [],
      followupError: null,
    });
    vi.mocked(printService.getShape3dJob).mockResolvedValue({
      id: "3d-1",
      status: "running",
      prompt: "una maceta",
      printer: "ender-3",
      source: "mesh",
      code: null,
      retries: 0,
      targetMm: null,
      targetMmSource: null,
      targetMmReference: null,
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:00Z",
      finishedAt: null,
      canPrint: null,
      sizeMm: null,
      triangleCount: null,
      blockers: [],
      advice: [],
      error: null,
      downloadUrl: null,
    });

    store.addTrackedJob({ id: "tr-1", kind: "transcribe", fileName: "charla.mp4", createdAt: 1 });
    store.addTrackedJob({ id: "dl-1", kind: "download", fileName: "Un video", createdAt: 2 });
    store.addTrackedJob({ id: "3d-1", kind: "shape3d", fileName: "una maceta", createdAt: 3 });

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });

    await waitFor(() =>
      expect(result.current.entries.every((entry) => entry.status === "running")).toBe(true),
    );
    expect(result.current.entries.map((entry) => entry.kind)).toEqual([
      "shape3d",
      "download",
      "transcribe",
    ]);
    // Una descarga deja los archivos en disco: no tiene URL de descarga y eso no
    // puede romper la entrada de la cola.
    expect(result.current.entries.find((entry) => entry.kind === "download")?.downloadUrl).toBeNull();
  });

  it("cancels a 3D job through its own endpoint", async () => {
    const store = createJobQueueStore();
    vi.mocked(printService.getShape3dJob).mockResolvedValue({
      id: "3d-1",
      status: "running",
      prompt: "una maceta",
      printer: "ender-3",
      source: "mesh",
      code: null,
      retries: 0,
      targetMm: null,
      targetMmSource: null,
      targetMmReference: null,
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: null,
      finishedAt: null,
      canPrint: null,
      sizeMm: null,
      triangleCount: null,
      blockers: [],
      advice: [],
      error: null,
      downloadUrl: null,
    });
    vi.mocked(printService.cancelShape3dJob).mockResolvedValue({} as never);
    store.addTrackedJob({ id: "3d-1", kind: "shape3d", fileName: "una maceta", createdAt: 1 });

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });
    await waitFor(() => expect(result.current.entries).toHaveLength(1));

    act(() => result.current.cancel("3d-1"));

    expect(printService.cancelShape3dJob).toHaveBeenCalledWith("3d-1");
  });

  it("clears only completed and failed jobs, keeping active ones", async () => {
    const store = createJobQueueStore();
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_IMAGE_JOB, status: "completed", downloadUrl: "/download" });
    vi.mocked(api.getVideoJob).mockResolvedValue({ ...BASE_VIDEO_JOB, status: "running" });
    store.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });
    store.addTrackedJob({ id: "vid-1", kind: "video", fileName: "clip.mp4", createdAt: 2 });

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });
    await waitFor(() => expect(result.current.entries.find((e) => e.id === "img-1")?.status).toBe("completed"));

    act(() => result.current.clearCompleted());

    await waitFor(() => expect(result.current.entries.map((entry) => entry.id)).toEqual(["vid-1"]));
  });
});
