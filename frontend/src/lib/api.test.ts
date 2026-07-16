import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ApiError,
  createImageJob,
  createVideoJob,
  deleteModel,
  getDevices,
  getEngineInfo,
  getHealth,
  getInstallStatus,
  getJob,
  getModels,
  getVideoJob,
  installModel,
  searchHfModels,
} from "./api";
import type {
  CreateInstallResponse,
  CreateJobResponse,
  DevicesResponse,
  EngineInfoResponse,
  HealthResponse,
  InstallStatusResponse,
  JobResponse,
  ModelSearchResponse,
  ModelsResponse,
  VideoJobResponse,
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
      metadata: {},
      progressPct: null,
      downloadUrl: null,
    };
    mockFetchOnce(payload);

    const result = await getJob("job-1");

    expect(fetch).toHaveBeenCalledWith("/api/v1/jobs/job-1", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});

describe("createVideoJob", () => {
  function videoParams(overrides: Partial<Parameters<typeof createVideoJob>[0]> = {}) {
    return {
      file: new File(["binary"], "clip.mp4", { type: "video/mp4" }),
      profileKey: "anime-balanced-2x",
      modelId: "realesrgan-x4plus",
      device: "dml:0",
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
      ...overrides,
    };
  }

  it("issues a multipart POST to /api/v1/video/jobs with every resolved field", async () => {
    const payload: CreateJobResponse = {
      jobId: "vid-1",
      status: "queued",
      statusUrl: "/api/v1/video/jobs/vid-1",
      downloadUrl: null,
    };
    mockFetchOnce(payload, { status: 202 });

    const result = await createVideoJob(videoParams());

    expect(fetch).toHaveBeenCalledWith("/api/v1/video/jobs", expect.objectContaining({ method: "POST" }));
    const call = vi.mocked(fetch).mock.calls[0];
    const body = call[1]?.body as FormData;
    expect(body.get("profile_key")).toBe("anime-balanced-2x");
    expect(body.get("model_name")).toBe("realesrgan-x4plus");
    expect(body.get("model_id")).toBe("realesrgan-x4plus");
    expect(body.get("device")).toBe("dml:0");
    expect(body.get("scale")).toBe("2");
    expect(body.get("output_container")).toBe("mp4");
    expect(body.get("video_codec")).toBe("libx264");
    expect(body.get("video_preset")).toBe("medium");
    expect(body.get("crf")).toBe("17");
    expect(body.get("keep_audio")).toBe("true");
    expect(body.get("fps_multiplier")).toBe("1");
    expect(body.has("target_fps")).toBe(false);
    expect(body.has("audio_enhance")).toBe(false);
    expect(result).toEqual(payload);
  });

  it("sends target_fps when set and omits model/device when absent", async () => {
    mockFetchOnce(
      { jobId: "vid-2", status: "queued", statusUrl: "/api/v1/video/jobs/vid-2", downloadUrl: null },
      { status: 202 },
    );

    await createVideoJob(videoParams({ modelId: null, device: null, targetFps: "60000/1001" }));

    const call = vi.mocked(fetch).mock.calls[0];
    const body = call[1]?.body as FormData;
    expect(body.has("model_name")).toBe(false);
    expect(body.has("model_id")).toBe(false);
    expect(body.has("device")).toBe(false);
    expect(body.get("target_fps")).toBe("60000/1001");
  });

  it("sends audio_enhance when set", async () => {
    mockFetchOnce(
      { jobId: "vid-3", status: "queued", statusUrl: "/api/v1/video/jobs/vid-3", downloadUrl: null },
      { status: 202 },
    );

    await createVideoJob(videoParams({ audioEnhance: "deepfilter" }));

    const call = vi.mocked(fetch).mock.calls[0];
    const body = call[1]?.body as FormData;
    expect(body.get("audio_enhance")).toBe("deepfilter");
  });
});

describe("getVideoJob", () => {
  it("issues a GET to /api/v1/video/jobs/{id} and returns the typed payload", async () => {
    const payload: VideoJobResponse = {
      jobId: "vid-1",
      status: "running",
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
      audioRestore: null,
      modelId: "realesrgan-x4plus",
      device: "dml:0",
      createdAt: "2026-01-01T00:00:00Z",
      startedAt: "2026-01-01T00:00:01Z",
      finishedAt: null,
      error: null,
      metadata: { stage: "upscaling_frames" },
      progressPct: null,
      downloadUrl: null,
    };
    mockFetchOnce(payload);

    const result = await getVideoJob("vid-1");

    expect(fetch).toHaveBeenCalledWith("/api/v1/video/jobs/vid-1", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});

describe("searchHfModels", () => {
  it("issues a GET to /api/v1/models/search with the query string encoded", async () => {
    const payload: ModelSearchResponse = {
      results: [
        {
          id: "example/anime-2x",
          author: "example",
          pipelineTag: "image-to-image",
          downloads: 120,
          likes: 5,
          tags: ["onnx"],
        },
      ],
    };
    mockFetchOnce(payload);

    const result = await searchHfModels("anime 2x");

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/models/search?q=anime%202x",
      expect.objectContaining({ method: "GET" }),
    );
    expect(result).toEqual(payload);
  });
});

describe("installModel", () => {
  it("issues a JSON POST to /api/v1/models/install with the repoId", async () => {
    const payload: CreateInstallResponse = {
      installId: "install-1",
      statusUrl: "/api/v1/models/install/install-1",
    };
    mockFetchOnce(payload, { status: 202 });

    const result = await installModel("example/anime-2x");

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/models/install",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ "Content-Type": "application/json" }),
        body: JSON.stringify({ repoId: "example/anime-2x" }),
      }),
    );
    expect(result).toEqual(payload);
  });
});

describe("getInstallStatus", () => {
  it("issues a GET to /api/v1/models/install/{id} and returns the typed payload", async () => {
    const payload: InstallStatusResponse = {
      installId: "install-1",
      repoId: "example/anime-2x",
      status: "downloading",
      progressPct: 42.5,
      modelId: null,
      error: null,
    };
    mockFetchOnce(payload);

    const result = await getInstallStatus("install-1");

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/models/install/install-1",
      expect.objectContaining({ method: "GET" }),
    );
    expect(result).toEqual(payload);
  });
});

describe("deleteModel", () => {
  it("issues a DELETE to /api/v1/models/{id} and resolves with no content", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 204 })));

    await expect(deleteModel("custom-anime-2x")).resolves.toBeUndefined();

    expect(fetch).toHaveBeenCalledWith("/api/v1/models/custom-anime-2x", expect.objectContaining({ method: "DELETE" }));
  });

  it("throws an ApiError with the detail message when the delete is rejected", async () => {
    mockFetchOnce({ detail: "Cannot delete a builtin model" }, { status: 409 });

    await expect(deleteModel("realesrgan-x4plus")).rejects.toMatchObject(
      new ApiError(409, "Cannot delete a builtin model"),
    );
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
