import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  AsrInstallStatusResponse,
  CreateJobResponse,
  DevicesResponse,
  ModelsResponse,
  ModelSearchResponse,
  TranscribeJob,
} from "../lib/apiTypes";
import {
  cancelTranscribeJob,
  createTranscribeJob,
  fetchInstalledAsrModels,
  fetchTranscribeDevices,
  getAsrInstallStatus,
  getTranscribeJob,
  installAsrModel,
  searchAsrModels,
} from "./transcribe";

function mockFetchOnce(body: unknown, init: ResponseInit = { status: 200 }) {
  const response = new Response(JSON.stringify(body), {
    ...init,
    headers: { "Content-Type": "application/json" },
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));
}

const COMPLETED_JOB: TranscribeJob = {
  id: "tr-1",
  status: "completed",
  originalFilename: "interview.wav",
  modelId: "asr-1",
  language: null,
  device: "cpu",
  createdAt: "2026-07-29T00:00:00Z",
  startedAt: "2026-07-29T00:00:01Z",
  finishedAt: "2026-07-29T00:00:20Z",
  progressPct: 100,
  text: "The response carries this text.",
  error: null,
  ownerId: null,
  downloadUrl: "/api/v1/transcribe/jobs/tr-1/download",
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("createTranscribeJob", () => {
  it("omits language when automatic detection is selected", async () => {
    const response: CreateJobResponse = {
      jobId: "tr-1",
      status: "queued",
      statusUrl: "/api/v1/transcribe/jobs/tr-1",
      downloadUrl: null,
    };
    mockFetchOnce(response, { status: 202 });
    const file = new File(["audio"], "interview.wav", {
      type: "audio/wav",
    });

    await createTranscribeJob({
      file,
      modelId: "asr-1",
      device: "cpu",
    });

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/transcribe/jobs",
      expect.objectContaining({ method: "POST" }),
    );
    const body = vi.mocked(fetch).mock.calls[0][1]?.body as FormData;
    expect(body.get("file")).toBe(file);
    expect(body.get("model_id")).toBe("asr-1");
    expect(body.has("language")).toBe(false);
    expect(body.get("device")).toBe("cpu");
  });

  it("sends a selected two-letter language code", async () => {
    mockFetchOnce(
      {
        jobId: "tr-2",
        status: "queued",
        statusUrl: "/api/v1/transcribe/jobs/tr-2",
        downloadUrl: null,
      },
      { status: 202 },
    );

    await createTranscribeJob({
      file: new File(["audio"], "spanish.mp3", { type: "audio/mpeg" }),
      modelId: "asr-2",
      language: "es",
      device: "dml:0",
    });

    const body = vi.mocked(fetch).mock.calls[0][1]?.body as FormData;
    expect(body.get("language")).toBe("es");
    expect(body.get("device")).toBe("dml:0");
  });
});

describe("transcribe job endpoints", () => {
  it("gets the job payload with its inline text", async () => {
    mockFetchOnce(COMPLETED_JOB);

    await expect(getTranscribeJob("tr-1")).resolves.toEqual(COMPLETED_JOB);
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/transcribe/jobs/tr-1",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("cancels the job with a POST", async () => {
    const cancelled = { ...COMPLETED_JOB, status: "cancelled" as const };
    mockFetchOnce(cancelled);

    await expect(cancelTranscribeJob("tr-1")).resolves.toEqual(cancelled);
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/transcribe/jobs/tr-1/cancel",
      expect.objectContaining({ method: "POST" }),
    );
  });
});

describe("ASR model endpoints", () => {
  it("searches with an encoded query", async () => {
    const response: ModelSearchResponse = { results: [] };
    mockFetchOnce(response);

    await expect(searchAsrModels("speech / spanish")).resolves.toEqual(
      response,
    );
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/asr/models/search?q=speech%20%2F%20spanish",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("starts an install with the repo id", async () => {
    mockFetchOnce(
      {
        installId: "install-1",
        statusUrl: "/api/v1/asr/models/install/install-1",
      },
      { status: 202 },
    );

    await installAsrModel("owner/whisper");

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/asr/models/install",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ repoId: "owner/whisper" }),
      }),
    );
  });

  it("polls install status", async () => {
    const response: AsrInstallStatusResponse = {
      installId: "install-1",
      repoId: "owner/whisper",
      status: "downloading",
      progressPct: 42,
      modelId: null,
      error: null,
    };
    mockFetchOnce(response);

    await expect(getAsrInstallStatus("install-1")).resolves.toEqual(response);
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/asr/models/install/install-1",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("keeps only installed asr-onnx models", async () => {
    const response: ModelsResponse = {
      models: [
        {
          id: "asr-1",
          name: "Whisper ONNX",
          kind: "asr-onnx",
          source: "hf",
          scale: null,
          arch: "whisper",
          sizeBytes: 10,
          status: "ready",
          error: null,
        },
        {
          id: "up-1",
          name: "Upscaler",
          kind: "onnx",
          source: "hf",
          scale: 4,
          arch: "rrdb",
          sizeBytes: 20,
          status: "ready",
          error: null,
        },
      ],
    };
    mockFetchOnce(response);

    await expect(fetchInstalledAsrModels()).resolves.toEqual([
      response.models[0],
    ]);
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/models",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("loads concrete devices for the selector", async () => {
    const response: DevicesResponse = {
      devices: [
        { id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" },
      ],
      defaultDeviceId: "cpu",
    };
    mockFetchOnce(response);

    await expect(fetchTranscribeDevices()).resolves.toEqual(response);
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/devices",
      expect.objectContaining({ method: "GET" }),
    );
  });
});
