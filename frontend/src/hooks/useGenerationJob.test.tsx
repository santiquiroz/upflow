import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { GenerationJob } from "../lib/apiTypes";
import { createJobQueueStore } from "../lib/jobQueueStore";
import * as generationService from "../services/generation";
import { useGenerationJob } from "./useGenerationJob";

vi.mock("../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/generation")>();
  return { ...actual, createGenerationJob: vi.fn(), getGenerationJob: vi.fn() };
});

const POLL_INTERVAL_MS = 10;

const BASE_JOB: GenerationJob = {
  id: "gen-1",
  status: "queued",
  prompt: "a red fox in the snow",
  negativePrompt: null,
  modelId: "sd15",
  steps: 25,
  guidance: 7.5,
  width: 512,
  height: 512,
  seed: null,
  device: null,
  autoUpscale: false,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
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
    prompt: "a red fox in the snow",
    negativePrompt: null,
    modelId: "sd15",
    steps: 25,
    guidance: 7.5,
    width: 512,
    height: 512,
    seed: null,
    device: null,
    autoUpscale: false,
    upscaleModelName: null,
    upscaleScale: null,
    upscaleModelId: null,
  };
}

afterEach(() => {
  vi.mocked(generationService.createGenerationJob).mockReset();
  vi.mocked(generationService.getGenerationJob).mockReset();
});

describe("useGenerationJob", () => {
  it("starts idle with no job and no error", () => {
    const { result } = renderHook(() => useGenerationJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    expect(result.current.phase).toBe("idle");
    expect(result.current.job).toBeUndefined();
    expect(result.current.errorMessage).toBeNull();
  });

  it("creates the job, polls while non-terminal, and stops polling once the job completes", async () => {
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ ...BASE_JOB, status: "queued" });
    vi.mocked(generationService.getGenerationJob)
      .mockResolvedValueOnce({ ...BASE_JOB, status: "running" })
      .mockResolvedValue({ ...BASE_JOB, status: "completed", downloadUrl: "/api/v1/generation/jobs/gen-1/download" });

    const { result } = renderHook(() => useGenerationJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("completed"));
    expect(result.current.job?.downloadUrl).toBe("/api/v1/generation/jobs/gen-1/download");

    const callsAtCompletion = vi.mocked(generationService.getGenerationJob).mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS * 5));
    expect(vi.mocked(generationService.getGenerationJob).mock.calls.length).toBe(callsAtCompletion);
  });

  it("surfaces the job error message when the job fails", async () => {
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ ...BASE_JOB, status: "queued" });
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({
      ...BASE_JOB,
      status: "failed",
      error: "model missing",
    });

    const { result } = renderHook(() => useGenerationJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("failed"));
    expect(result.current.errorMessage).toBe("model missing");
  });

  it("tracks a submitted job in the shared queue store with the generation kind and prompt as fileName", async () => {
    const queue = createJobQueueStore();
    vi.mocked(generationService.createGenerationJob).mockResolvedValue({ ...BASE_JOB, status: "queued" });
    vi.mocked(generationService.getGenerationJob).mockResolvedValue({ ...BASE_JOB, status: "queued" });

    const { result } = renderHook(() => useGenerationJob(POLL_INTERVAL_MS, queue), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(queue.getSnapshot()).toHaveLength(1));
    expect(queue.getSnapshot()[0]).toMatchObject({
      id: "gen-1",
      kind: "generation",
      fileName: "a red fox in the snow",
    });
  });
});
