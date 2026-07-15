import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import type { CreateJobResponse, JobResponse } from "../lib/apiTypes";
import { useImageJob } from "./useImageJob";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return {
    ...actual,
    createImageJob: vi.fn(),
    getJob: vi.fn(),
  };
});

const POLL_INTERVAL_MS = 10;

const BASE_JOB: JobResponse = {
  jobId: "job-1",
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
  return new File(["binary"], "photo.png", { type: "image/png" });
}

function submitParams() {
  return { file: makeFile(), modelId: "realesrgan-x4plus", device: null, scale: 4, outputFormat: "png" };
}

afterEach(() => {
  vi.mocked(api.createImageJob).mockReset();
  vi.mocked(api.getJob).mockReset();
});

describe("useImageJob", () => {
  it("starts idle with no job and no error", () => {
    const { result } = renderHook(() => useImageJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    expect(result.current.phase).toBe("idle");
    expect(result.current.job).toBeUndefined();
    expect(result.current.errorMessage).toBeNull();
  });

  it("reports the uploading phase while the upload request is in flight", async () => {
    let resolveUpload: (value: CreateJobResponse) => void = () => {};
    vi.mocked(api.createImageJob).mockReturnValue(
      new Promise<CreateJobResponse>((resolve) => {
        resolveUpload = resolve;
      }),
    );
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_JOB, status: "completed" });

    const { result } = renderHook(() => useImageJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("uploading"));

    act(() => {
      resolveUpload({ jobId: "job-1", status: "queued", statusUrl: "/api/v1/jobs/job-1", downloadUrl: null });
    });

    await waitFor(() => expect(result.current.phase).not.toBe("uploading"));
  });

  it("uploads, polls while non-terminal, and stops polling once the job completes", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "job-1",
      status: "queued",
      statusUrl: "/api/v1/jobs/job-1",
      downloadUrl: null,
    };
    vi.mocked(api.createImageJob).mockResolvedValue(createResponse);
    vi.mocked(api.getJob)
      .mockResolvedValueOnce({ ...BASE_JOB, status: "queued" })
      .mockResolvedValueOnce({ ...BASE_JOB, status: "running" })
      .mockResolvedValue({ ...BASE_JOB, status: "completed", downloadUrl: "/api/v1/jobs/job-1/download" });

    const { result } = renderHook(() => useImageJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("completed"));
    expect(result.current.job?.downloadUrl).toBe("/api/v1/jobs/job-1/download");
    expect(vi.mocked(api.getJob).mock.calls.length).toBeGreaterThanOrEqual(3);

    const callsAtCompletion = vi.mocked(api.getJob).mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS * 5));
    expect(vi.mocked(api.getJob).mock.calls.length).toBe(callsAtCompletion);
  });

  it("surfaces the upload error message when the request is rejected", async () => {
    vi.mocked(api.createImageJob).mockRejectedValue(new Error("Queue is full"));

    const { result } = renderHook(() => useImageJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.errorMessage).toBe("Queue is full"));
    expect(result.current.phase).toBe("idle");
  });

  it("surfaces the job error message when the job fails", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "job-1",
      status: "queued",
      statusUrl: "/api/v1/jobs/job-1",
      downloadUrl: null,
    };
    vi.mocked(api.createImageJob).mockResolvedValue(createResponse);
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_JOB, status: "failed", error: "Model crashed" });

    const { result } = renderHook(() => useImageJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

    act(() => {
      result.current.submit(submitParams());
    });

    await waitFor(() => expect(result.current.phase).toBe("failed"));
    expect(result.current.errorMessage).toBe("Model crashed");
  });

  it("resets back to idle", async () => {
    const createResponse: CreateJobResponse = {
      jobId: "job-1",
      status: "queued",
      statusUrl: "/api/v1/jobs/job-1",
      downloadUrl: null,
    };
    vi.mocked(api.createImageJob).mockResolvedValue(createResponse);
    vi.mocked(api.getJob).mockResolvedValue({ ...BASE_JOB, status: "queued" });

    const { result } = renderHook(() => useImageJob(POLL_INTERVAL_MS), { wrapper: createWrapper() });

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
});
