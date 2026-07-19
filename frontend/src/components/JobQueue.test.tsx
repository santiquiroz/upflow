import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import type { AudioJob, JobResponse, VideoJobResponse } from "../lib/apiTypes";
import { createJobQueueStore, jobQueueStore } from "../lib/jobQueueStore";
import * as audioService from "../services/audio";
import { JobQueue } from "./JobQueue";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, getJob: vi.fn(), getVideoJob: vi.fn(), cancelJob: vi.fn(), cancelVideoJob: vi.fn() };
});

vi.mock("../services/audio", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/audio")>();
  return { ...actual, getAudioJob: vi.fn(), cancelAudioJob: vi.fn() };
});

const BASE_AUDIO_JOB: AudioJob = {
  id: "aud-1",
  status: "running",
  originalFilename: "voice.wav",
  denoise: "deepfilter",
  restore: null,
  device: null,
  progressPct: null,
  stages: null,
  error: null,
  downloadUrl: null,
};

const BASE_IMAGE_JOB: JobResponse = {
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
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  error: null,
  metadata: {},
  progressPct: null,
  downloadUrl: null,
};

function renderQueue() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<JobQueue />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(api.getJob).mockReset();
  vi.mocked(api.getVideoJob).mockReset();
  vi.mocked(api.cancelJob).mockReset();
  vi.mocked(api.cancelVideoJob).mockReset();
  vi.mocked(audioService.getAudioJob).mockReset();
  vi.mocked(audioService.cancelAudioJob).mockReset();
  // JobQueue always reads the singleton jobQueueStore, so each test must
  // clear it -- otherwise jobs tracked by an earlier test would still show
  // up here since the store is a module-level singleton shared across tests
  // in this file.
  jobQueueStore.getSnapshot().forEach((job) => jobQueueStore.removeTrackedJob(job.id));
});

// Tests that resolve a query print a benign "not wrapped in act(...)" warning
// from @tanstack/react-query's useQueries observer (its update scheduling
// differs from the single-query useQuery, which does not warn here) -- it
// does not fail the suite and is a known upstream interaction with RTL, not a
// bug in JobQueue/useJobQueue.
describe("JobQueue", () => {
  it("shows an empty state when no jobs are tracked", () => {
    renderQueue();

    expect(screen.getByText(/no active jobs/i)).toBeInTheDocument();
  });

  it("shows a live progress indicator for a running job", async () => {
    vi.mocked(api.getVideoJob).mockResolvedValue(BASE_VIDEO_JOB);
    jobQueueStore.addTrackedJob({ id: "vid-1", kind: "video", fileName: "clip.mp4", createdAt: 1 });

    renderQueue();

    expect(await screen.findByRole("progressbar")).toBeInTheDocument();
    expect(screen.getByText("clip.mp4")).toBeInTheDocument();
  });

  it("shows a download action once a job completes", async () => {
    vi.mocked(api.getJob).mockResolvedValue({
      ...BASE_IMAGE_JOB,
      status: "completed",
      downloadUrl: "/api/v1/jobs/img-1/download",
    });
    jobQueueStore.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });

    renderQueue();

    const link = await screen.findByRole("link", { name: /download photo\.png/i });
    expect(link).toHaveAttribute("href", "/api/v1/jobs/img-1/download");
  });

  it("shows the error message for a failed job", async () => {
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_IMAGE_JOB, status: "failed", error: "Model crashed" });
    jobQueueStore.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });

    renderQueue();

    expect(await screen.findByRole("alert")).toHaveTextContent("Model crashed");
  });

  it("removes a job when its dismiss button is clicked", async () => {
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_IMAGE_JOB, status: "completed", downloadUrl: "/download" });
    jobQueueStore.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });

    renderQueue();
    await screen.findByRole("link", { name: /download/i });

    fireEvent.click(screen.getByRole("button", { name: /dismiss photo\.png/i }));

    expect(screen.queryByText("photo.png")).not.toBeInTheDocument();
    expect(screen.getByText(/no active jobs/i)).toBeInTheDocument();
  });

  it("clears completed jobs but keeps active ones when Clear completed is clicked", async () => {
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_IMAGE_JOB, status: "completed", downloadUrl: "/download" });
    vi.mocked(api.getVideoJob).mockResolvedValue({ ...BASE_VIDEO_JOB, status: "running" });
    jobQueueStore.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });
    jobQueueStore.addTrackedJob({ id: "vid-1", kind: "video", fileName: "clip.mp4", createdAt: 2 });

    renderQueue();
    await waitFor(() => expect(screen.getByRole("link", { name: /download/i })).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /clear completed/i }));

    await waitFor(() => expect(screen.queryByText("photo.png")).not.toBeInTheDocument());
    expect(screen.getByText("clip.mp4")).toBeInTheDocument();
  });

  it("does not show Clear completed when there is nothing to clear", async () => {
    vi.mocked(api.getVideoJob).mockResolvedValue(BASE_VIDEO_JOB);
    jobQueueStore.addTrackedJob({ id: "vid-1", kind: "video", fileName: "clip.mp4", createdAt: 1 });

    renderQueue();
    await screen.findByText("clip.mp4");

    expect(screen.queryByRole("button", { name: /clear completed/i })).not.toBeInTheDocument();
  });

  it("shows a tabular job count badge", async () => {
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_IMAGE_JOB, status: "completed", downloadUrl: "/download" });
    jobQueueStore.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });

    renderQueue();
    await screen.findByRole("link", { name: /download/i });

    const badge = screen.getByText("1");
    expect(badge).toHaveClass("font-mono-tabular");
  });

  it("uses an isolated store instance without leaking into the shared singleton", () => {
    const isolatedStore = createJobQueueStore();
    isolatedStore.addTrackedJob({ id: "isolated-1", kind: "image", fileName: "isolated.png", createdAt: 1 });

    expect(jobQueueStore.getSnapshot()).toHaveLength(0);
  });

  it("opens the job detail modal when a queue item is clicked", async () => {
    vi.mocked(api.getVideoJob).mockResolvedValue(BASE_VIDEO_JOB);
    jobQueueStore.addTrackedJob({ id: "vid-1", kind: "video", fileName: "clip.mp4", createdAt: 1 });

    renderQueue();
    await screen.findByText("clip.mp4");

    fireEvent.click(screen.getByRole("button", { name: /view details for clip\.mp4/i }));

    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveTextContent("realesr-animevideov3-x2");
  });

  it("closes the job detail modal when Escape is pressed", async () => {
    vi.mocked(api.getVideoJob).mockResolvedValue(BASE_VIDEO_JOB);
    jobQueueStore.addTrackedJob({ id: "vid-1", kind: "video", fileName: "clip.mp4", createdAt: 1 });

    renderQueue();
    await screen.findByText("clip.mp4");
    fireEvent.click(screen.getByRole("button", { name: /view details for clip\.mp4/i }));
    const dialog = await screen.findByRole("dialog");

    fireEvent.keyDown(dialog, { key: "Escape" });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("cancels an active image job through the image cancel endpoint", async () => {
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_IMAGE_JOB, status: "running" });
    vi.mocked(api.cancelJob).mockResolvedValue({ ...BASE_IMAGE_JOB, status: "cancelled" });
    jobQueueStore.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });

    renderQueue();
    await screen.findByText("photo.png");

    fireEvent.click(await screen.findByRole("button", { name: /cancel photo\.png/i }));

    expect(api.cancelJob).toHaveBeenCalledWith("img-1");
    expect(api.cancelVideoJob).not.toHaveBeenCalled();
    expect(audioService.cancelAudioJob).not.toHaveBeenCalled();
  });

  it("cancels an active video job through the video cancel endpoint", async () => {
    vi.mocked(api.getVideoJob).mockResolvedValue(BASE_VIDEO_JOB);
    vi.mocked(api.cancelVideoJob).mockResolvedValue({ ...BASE_VIDEO_JOB, status: "cancelled" });
    jobQueueStore.addTrackedJob({ id: "vid-1", kind: "video", fileName: "clip.mp4", createdAt: 1 });

    renderQueue();
    await screen.findByText("clip.mp4");

    fireEvent.click(await screen.findByRole("button", { name: /cancel clip\.mp4/i }));

    expect(api.cancelVideoJob).toHaveBeenCalledWith("vid-1");
    expect(api.cancelJob).not.toHaveBeenCalled();
  });

  it("cancels an active audio job through the audio cancel endpoint", async () => {
    vi.mocked(audioService.getAudioJob).mockResolvedValue(BASE_AUDIO_JOB);
    vi.mocked(audioService.cancelAudioJob).mockResolvedValue({ ...BASE_AUDIO_JOB, status: "cancelled" });
    jobQueueStore.addTrackedJob({ id: "aud-1", kind: "audio", fileName: "voice.wav", createdAt: 1 });

    renderQueue();
    await screen.findByText("voice.wav");

    fireEvent.click(await screen.findByRole("button", { name: /cancel voice\.wav/i }));

    expect(audioService.cancelAudioJob).toHaveBeenCalledWith("aud-1");
    expect(api.cancelJob).not.toHaveBeenCalled();
  });

  it("renders a cancelled job with its status label and no cancel control", async () => {
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_IMAGE_JOB, status: "cancelled" });
    jobQueueStore.addTrackedJob({ id: "img-1", kind: "image", fileName: "photo.png", createdAt: 1 });

    renderQueue();

    expect(await screen.findByText("Cancelled")).toBeInTheDocument();
    await waitFor(() => expect(screen.queryByRole("button", { name: /cancel photo\.png/i })).not.toBeInTheDocument());
    expect(screen.getByRole("button", { name: /dismiss photo\.png/i })).toBeInTheDocument();
  });
});
