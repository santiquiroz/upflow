import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, createImageJob, getDevices, getEngineInfo, getHealth, getJob, getModels } from "./api";
import type {
  CreateJobResponse,
  DevicesResponse,
  EngineInfoResponse,
  HealthResponse,
  JobResponse,
  ModelsResponse,
} from "./apiTypes";

function mockFetchOnce(body: unknown, init: ResponseInit = { status: 200 }) {
  const response = new Response(JSON.stringify(body), {
    ...init,
    headers: { "Content-Type": "application/json" },
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("getHealth", () => {
  it("issues a GET to /api/v1/health and returns the typed payload", async () => {
    const payload: HealthResponse = {
      status: "ok",
      engine: "realesrgan-ncnn",
      gpuConcurrency: 1,
      queueDepth: 0,
      videoQueueDepth: 0,
    };
    mockFetchOnce(payload);

    const result = await getHealth();

    expect(fetch).toHaveBeenCalledWith("/api/v1/health", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});

describe("getEngineInfo", () => {
  it("issues a GET to /api/v1/engine and returns the typed payload", async () => {
    const payload: EngineInfoResponse = {
      engine: "realesrgan-ncnn",
      configuredBinary: "vendor/realesrgan/realesrgan-ncnn-vulkan.exe",
      configuredModelsDir: "vendor/realesrgan/models",
      available: true,
      defaultModel: "realesrgan-x4plus",
      allowedScales: [2, 3, 4],
      supportedModels: [],
      videoProfiles: [],
      ffmpegAvailable: true,
    };
    mockFetchOnce(payload);

    const result = await getEngineInfo();

    expect(fetch).toHaveBeenCalledWith("/api/v1/engine", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});

describe("getDevices", () => {
  it("issues a GET to /api/v1/devices and returns the typed payload", async () => {
    const payload: DevicesResponse = {
      devices: [{ id: "dml:0", kind: "gpu", name: "AMD GPU", backend: "directml" }],
      defaultDeviceId: "dml:0",
    };
    mockFetchOnce(payload);

    const result = await getDevices();

    expect(fetch).toHaveBeenCalledWith("/api/v1/devices", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});

describe("getModels", () => {
  it("issues a GET to /api/v1/models and returns the typed payload", async () => {
    const payload: ModelsResponse = { models: [] };
    mockFetchOnce(payload);

    const result = await getModels();

    expect(fetch).toHaveBeenCalledWith("/api/v1/models", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});

describe("createImageJob", () => {
  it("issues a multipart POST to /api/v1/jobs with the resolved model id in both fields", async () => {
    const payload: CreateJobResponse = {
      jobId: "job-1",
      status: "queued",
      statusUrl: "/api/v1/jobs/job-1",
      downloadUrl: null,
    };
    mockFetchOnce(payload, { status: 202 });
    const file = new File(["binary"], "photo.png", { type: "image/png" });

    const result = await createImageJob({
      file,
      modelId: "realesrgan-x4plus",
      device: "dml:0",
      scale: 4,
      outputFormat: "png",
    });

    expect(fetch).toHaveBeenCalledWith("/api/v1/jobs", expect.objectContaining({ method: "POST" }));
    const call = vi.mocked(fetch).mock.calls[0];
    const body = call[1]?.body as FormData;
    expect(body.get("file")).toBe(file);
    expect(body.get("model_name")).toBe("realesrgan-x4plus");
    expect(body.get("model_id")).toBe("realesrgan-x4plus");
    expect(body.get("device")).toBe("dml:0");
    expect(body.get("scale")).toBe("4");
    expect(body.get("output_format")).toBe("png");
    expect(result).toEqual(payload);
  });

  it("omits the device field when no device is selected", async () => {
    mockFetchOnce(
      { jobId: "job-2", status: "queued", statusUrl: "/api/v1/jobs/job-2", downloadUrl: null },
      { status: 202 },
    );
    const file = new File(["binary"], "photo.png", { type: "image/png" });

    await createImageJob({ file, modelId: "realesrgan-x4plus", device: null, scale: 2, outputFormat: "webp" });

    const call = vi.mocked(fetch).mock.calls[0];
    const body = call[1]?.body as FormData;
    expect(body.has("device")).toBe(false);
  });
});

describe("getJob", () => {
  it("issues a GET to /api/v1/jobs/{id} and returns the typed payload", async () => {
    const payload: JobResponse = {
      jobId: "job-1",
      status: "running",
      originalFilename: "photo.png",
      modelName: "realesrgan-x4plus",
      scale: 4,
      outputFormat: "png",
      modelId: "realesrgan-x4plus",
      device: "dml:0",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:01Z",
      finishedAt: null,
      error: null,
      downloadUrl: null,
    };
    mockFetchOnce(payload);

    const result = await getJob("job-1");

    expect(fetch).toHaveBeenCalledWith("/api/v1/jobs/job-1", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});

describe("error handling", () => {
  it("throws an ApiError carrying the status and response body on a non-ok response", async () => {
    mockFetchOnce({ detail: "Job not found" }, { status: 404 });

    await expect(getHealth()).rejects.toMatchObject(
      new ApiError(404, "Job not found"),
    );
  });

  it("falls back to statusText when the error body has no detail field", async () => {
    const response = new Response("not json", { status: 500, statusText: "Internal Server Error" });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));

    await expect(getHealth()).rejects.toMatchObject(new ApiError(500, "Internal Server Error"));
  });
});
