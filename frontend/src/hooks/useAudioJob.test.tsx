import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AudioJob, CreateJobResponse } from "../lib/apiTypes";
import { createJobQueueStore } from "../lib/jobQueueStore";
import * as audioService from "../services/audio";
import { useAudioJob } from "./useAudioJob";

vi.mock("../services/audio", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/audio")>();
  return { ...actual, createAudioJob: vi.fn(), getAudioJob: vi.fn() };
});

const POLL_INTERVAL_MS = 10;

const BASE_JOB: AudioJob = {
  id: "aud-1",
  status: "queued",
  originalFilename: "voice.wav",
  denoise: "deepfilter",
  restore: null,
  device: null,
  progressPct: null,
  stages: null,
  error: null,
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

function submitParams() {
  return {
    file: new File(["binary"], "voice.wav", { type: "audio/wav" }),
    denoise: "deepfilter",
    restore: null,
    outputFormat: "flac",
    device: null,
  };
}

afterEach(() => {
  vi.mocked(audioService.createAudioJob).mockReset();
  vi.mocked(audioService.getAudioJob).mockReset();
});

describe("useAudioJob", () => {
  it("starts idle with no job and no error", () => {
    const { result } = renderHook(() => useAudioJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    expect(result.current.phase).toBe("idle");
    expect(result.current.job).toBeUndefined();
    expect(result.current.errorMessage).toBeNull();
  });

  it("uploads, polls while non-terminal, and stops polling once the job completes", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "aud-1",
      status: "queued",
      statusUrl: "/api/v1/audio/jobs/aud-1",
      downloadUrl: null,
    };
    vi.mocked(audioService.createAudioJob).mockResolvedValue(createResponse);
    vi.mocked(audioService.getAudioJob)
      .mockResolvedValueOnce({ ...BASE_JOB, status: "running" })
      .mockResolvedValue({ ...BASE_JOB, status: "completed", downloadUrl: "/api/v1/audio/jobs/aud-1/download" });

    const { result } = renderHook(() => useAudioJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("completed"));
    expect(result.current.job?.downloadUrl).toBe("/api/v1/audio/jobs/aud-1/download");

    const callsAtCompletion = vi.mocked(audioService.getAudioJob).mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS * 5));
    expect(vi.mocked(audioService.getAudioJob).mock.calls.length).toBe(callsAtCompletion);
  });

  it("surfaces the job error message when the job fails", async () => {
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "aud-1",
      status: "queued",
      statusUrl: "/api/v1/audio/jobs/aud-1",
      downloadUrl: null,
    });
    vi.mocked(audioService.getAudioJob).mockResolvedValue({ ...BASE_JOB, status: "failed", error: "model missing" });

    const { result } = renderHook(() => useAudioJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("failed"));
    expect(result.current.errorMessage).toBe("model missing");
  });

  it("tracks a submitted job in the shared queue store with the audio kind and file name", async () => {
    const queue = createJobQueueStore();
    vi.mocked(audioService.createAudioJob).mockResolvedValue({
      jobId: "aud-1",
      status: "queued",
      statusUrl: "/api/v1/audio/jobs/aud-1",
      downloadUrl: null,
    });
    vi.mocked(audioService.getAudioJob).mockResolvedValue({ ...BASE_JOB, status: "queued" });

    const { result } = renderHook(() => useAudioJob(POLL_INTERVAL_MS, queue), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(queue.getSnapshot()).toHaveLength(1));
    expect(queue.getSnapshot()[0]).toMatchObject({ id: "aud-1", kind: "audio", fileName: "voice.wav" });
  });
});
