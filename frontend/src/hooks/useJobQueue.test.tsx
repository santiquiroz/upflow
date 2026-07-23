import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import type { GenerationJob, JobResponse, VideoJobResponse } from "../lib/apiTypes";
import { createJobQueueStore } from "../lib/jobQueueStore";
import * as generationService from "../services/generation";
import { useJobQueue } from "./useJobQueue";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, getJob: vi.fn(), getVideoJob: vi.fn() };
});

vi.mock("../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/generation")>();
  return { ...actual, getGenerationJob: vi.fn(), cancelGenerationJob: vi.fn() };
});

const POLL_INTERVAL_MS = 10;

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
});

describe("useJobQueue", () => {
  it("starts empty when no jobs were tracked", () => {
    const store = createJobQueueStore();

    const { result } = renderHook(() => useJobQueue(store, POLL_INTERVAL_MS), { wrapper: createWrapper() });

    expect(result.current.entries).toEqual([]);
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
