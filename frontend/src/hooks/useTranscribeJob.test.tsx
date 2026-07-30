import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { TranscribeJob } from "../lib/apiTypes";
import * as transcribeService from "../services/transcribe";
import {
  useAsrModelInstall,
  useTranscribeJob,
} from "./useTranscribeJob";

vi.mock("../services/transcribe", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../services/transcribe")>();
  return {
    ...actual,
    createTranscribeJob: vi.fn(),
    getTranscribeJob: vi.fn(),
    cancelTranscribeJob: vi.fn(),
    installAsrModel: vi.fn(),
    getAsrInstallStatus: vi.fn(),
  };
});

const POLL_INTERVAL_MS = 10;

const BASE_JOB: TranscribeJob = {
  id: "tr-1",
  status: "queued",
  originalFilename: "voice.wav",
  modelId: "asr-1",
  language: null,
  device: "cpu",
  createdAt: "2026-07-29T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  progressPct: null,
  text: null,
  error: null,
  ownerId: null,
  downloadUrl: null,
};

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };
}

function submitParams() {
  return {
    file: new File(["audio"], "voice.wav", { type: "audio/wav" }),
    modelId: "asr-1",
    device: "cpu",
  };
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("useTranscribeJob", () => {
  it("starts idle", () => {
    const { result } = renderHook(
      () => useTranscribeJob(POLL_INTERVAL_MS),
      { wrapper: createWrapper() },
    );

    expect(result.current.phase).toBe("idle");
    expect(result.current.job).toBeUndefined();
    expect(result.current.errorMessage).toBeNull();
  });

  it("polls until completion and exposes the inline text", async () => {
    vi.mocked(transcribeService.createTranscribeJob).mockResolvedValue({
      jobId: "tr-1",
      status: "queued",
      statusUrl: "/api/v1/transcribe/jobs/tr-1",
      downloadUrl: null,
    });
    vi.mocked(transcribeService.getTranscribeJob)
      .mockResolvedValueOnce({
        ...BASE_JOB,
        status: "running",
        progressPct: 30,
      })
      .mockResolvedValue({
        ...BASE_JOB,
        status: "completed",
        progressPct: 100,
        text: "Inline transcription",
        downloadUrl: "/api/v1/transcribe/jobs/tr-1/download",
      });

    const { result } = renderHook(
      () => useTranscribeJob(POLL_INTERVAL_MS),
      { wrapper: createWrapper() },
    );

    act(() => result.current.submit(submitParams()));

    await waitFor(() => expect(result.current.phase).toBe("completed"));
    expect(result.current.job?.text).toBe("Inline transcription");

    const callsAtCompletion = vi.mocked(transcribeService.getTranscribeJob).mock
      .calls.length;
    await new Promise((resolve) =>
      setTimeout(resolve, POLL_INTERVAL_MS * 5),
    );
    expect(transcribeService.getTranscribeJob).toHaveBeenCalledTimes(
      callsAtCompletion,
    );
  });

  it("cancels the active job and asks the poll to refresh", async () => {
    vi.mocked(transcribeService.createTranscribeJob).mockResolvedValue({
      jobId: "tr-1",
      status: "queued",
      statusUrl: "/api/v1/transcribe/jobs/tr-1",
      downloadUrl: null,
    });
    vi.mocked(transcribeService.getTranscribeJob).mockResolvedValue({
      ...BASE_JOB,
      status: "running",
      progressPct: 50,
    });
    vi.mocked(transcribeService.cancelTranscribeJob).mockResolvedValue({
      ...BASE_JOB,
      status: "cancelled",
    });

    const { result } = renderHook(
      () => useTranscribeJob(POLL_INTERVAL_MS),
      { wrapper: createWrapper() },
    );
    act(() => result.current.submit(submitParams()));
    await waitFor(() => expect(result.current.phase).toBe("running"));

    act(() => result.current.cancel());

    await waitFor(() =>
      expect(transcribeService.cancelTranscribeJob).toHaveBeenCalledWith(
        "tr-1",
      ),
    );
  });

  it("surfaces a failed job error", async () => {
    vi.mocked(transcribeService.createTranscribeJob).mockResolvedValue({
      jobId: "tr-1",
      status: "queued",
      statusUrl: "/api/v1/transcribe/jobs/tr-1",
      downloadUrl: null,
    });
    vi.mocked(transcribeService.getTranscribeJob).mockResolvedValue({
      ...BASE_JOB,
      status: "failed",
      error: "Decoder initialization failed",
    });

    const { result } = renderHook(
      () => useTranscribeJob(POLL_INTERVAL_MS),
      { wrapper: createWrapper() },
    );
    act(() => result.current.submit(submitParams()));

    await waitFor(() => expect(result.current.phase).toBe("failed"));
    expect(result.current.errorMessage).toBe(
      "Decoder initialization failed",
    );
  });
});

describe("useAsrModelInstall", () => {
  it("polls an ASR install and exposes download progress", async () => {
    vi.mocked(transcribeService.installAsrModel).mockResolvedValue({
      installId: "install-1",
      statusUrl: "/api/v1/asr/models/install/install-1",
    });
    vi.mocked(transcribeService.getAsrInstallStatus).mockResolvedValue({
      installId: "install-1",
      repoId: "owner/whisper",
      status: "downloading",
      progressPct: 42,
      modelId: null,
      error: null,
    });

    const { result } = renderHook(
      () => useAsrModelInstall(POLL_INTERVAL_MS),
      { wrapper: createWrapper() },
    );

    act(() => result.current.install("owner/whisper"));

    await waitFor(() => expect(result.current.phase).toBe("downloading"));
    expect(result.current.progressPct).toBe(42);
  });
});
