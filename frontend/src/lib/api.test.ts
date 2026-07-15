import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, getDevices, getEngineInfo, getHealth, getModels } from "./api";
import type { DevicesResponse, EngineInfoResponse, HealthResponse, ModelsResponse } from "./apiTypes";

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
