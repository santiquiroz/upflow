import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import type { CreateJobResponse, VideoJobResponse } from "../lib/apiTypes";
import { createJobQueueStore } from "../lib/jobQueueStore";
import { useVideoJob } from "./useVideoJob";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return {
    ...actual,
    createVideoJob: vi.fn(),
    getVideoJob: vi.fn(),
  };
});

const POLL_INTERVAL_MS = 10;

const BASE_JOB: VideoJobResponse = {
  jobId: "vid-1",
  status: "queued",
  originalFilename: "clip.mp4",
  modelName: "realesrgan-x4plus",
  scale: 2,
  outputContainer: "mp4",
  videoCodec: "libx264",
  videoPreset: "medium",
  crf: 17,
  keepAudio: true,
  fpsMultiplier: 1,
  targetFps: null,
  audioEnhance: null,
  modelId: "realesrgan-x4plus",
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  error: null,
  metadata: {},
  downloadUrl: null,
};

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
}

function makeFile(): File {
  return new File(["binary"], "clip.mp4", { type: "video/mp4" });
}

function submitParams() {
  return {
    file: makeFile(),
    profileKey: "anime-balanced-2x",
    modelId: "realesrgan-x4plus",
    device: null,
    scale: 2,
    outputContainer: "mp4",
    videoCodec: "libx264",
    videoPreset: "medium",
    crf: 17,
    keepAudio: true,
    fpsMultiplier: 1,
    targetFps: null,
    audioEnhance: null,
  };
}

afterEach(() => {
  vi.mocked(api.createVideoJob).mockReset();
  vi.mocked(api.getVideoJob).mockReset();
});

describe("useVideoJob", () => {
  it("starts idle with no job and no error", () => {
    const { result } = renderHook(() => useVideoJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    expect(result.current.phase).toBe("idle");
    expect(result.current.job).toBeUndefined();
    expect(result.current.errorMessage).toBeNull();
  });

  it("reports the uploading phase while the upload request is in flight", async () => {
    let resolveUpload: (value: CreateJobResponse) => void = () => {};
    vi.mocked(api.createVideoJob).mockReturnValue(
      new Promise<CreateJobResponse>((resolve) => {
        resolveUpload = resolve;
      }),
    );
    vi.mocked(api.getVideoJob).mockResolvedValue({ ...BASE_JOB, status: "completed" });

    const { result } = renderHook(() => useVideoJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("uploading"));

    act(() => {
      resolveUpload({ jobId: "vid-1", status: "queued", statusUrl: "/api/v1/video/jobs/vid-1", downloadUrl: null });
    });

    await waitFor(() => expect(result.current.phase).not.toBe("uploading"));
  });

  it("uploads, polls while non-terminal, and stops polling once the job completes", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "vid-1",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-1",
      downloadUrl: null,
    };
    vi.mocked(api.createVideoJob).mockResolvedValue(createResponse);
    vi.mocked(api.getVideoJob)
      .mockResolvedValueOnce({ ...BASE_JOB, status: "queued" })
      .mockResolvedValueOnce({ ...BASE_JOB, status: "running" })
      .mockResolvedValue({ ...BASE_JOB, status: "completed", downloadUrl: "/api/v1/video/jobs/vid-1/download" });

    const { result } = renderHook(() => useVideoJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("completed"));
    expect(result.current.job?.downloadUrl).toBe("/api/v1/video/jobs/vid-1/download");
    expect(vi.mocked(api.getVideoJob).mock.calls.length).toBeGreaterThanOrEqual(3);

    const callsAtCompletion = vi.mocked(api.getVideoJob).mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS * 5));
    expect(vi.mocked(api.getVideoJob).mock.calls.length).toBe(callsAtCompletion);
  });

  it("surfaces the upload error message when the request is rejected", async () => {
    vi.mocked(api.createVideoJob).mockRejectedValue(new Error("Queue is full"));

    const { result } = renderHook(() => useVideoJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.errorMessage).toBe("Queue is full"));
    expect(result.current.phase).toBe("idle");
  });

  it("surfaces the job error message when the job fails", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "vid-1",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-1",
      downloadUrl: null,
    };
    vi.mocked(api.createVideoJob).mockResolvedValue(createResponse);
    vi.mocked(api.getVideoJob).mockResolvedValue({ ...BASE_JOB, status: "failed", error: "ffmpeg crashed" });

    const { result } = renderHook(() => useVideoJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("failed"));
    expect(result.current.errorMessage).toBe("ffmpeg crashed");
  });

  it("resets back to idle", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "vid-1",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-1",
      downloadUrl: null,
    };
    vi.mocked(api.createVideoJob).mockResolvedValue(createResponse);
    vi.mocked(api.getVideoJob).mockResolvedValue({ ...BASE_JOB, status: "queued" });

    const { result } = renderHook(() => useVideoJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("queued"));

    act(() => {
      result.current.reset();
    });

    expect(result.current.phase).toBe("idle");
    expect(result.current.job).toBeUndefined();
  });

  it("tracks a submitted job in the shared job queue store with its file name", async () => {
    const queue = createJobQueueStore();
    const createResponse: CreateJobResponse = {
      jobId: "vid-1",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-1",
      downloadUrl: null,
    };
    vi.mocked(api.createVideoJob).mockResolvedValue(createResponse);
    vi.mocked(api.getVideoJob).mockResolvedValue({ ...BASE_JOB, status: "queued" });

    const { result } = renderHook(() => useVideoJob(POLL_INTERVAL_MS, queue), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(queue.getSnapshot()).toHaveLength(1));
    expect(queue.getSnapshot()[0]).toMatchObject({ id: "vid-1", kind: "video", fileName: "clip.mp4" });
  });
});
