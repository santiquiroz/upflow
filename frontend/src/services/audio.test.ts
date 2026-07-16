import { afterEach, describe, expect, it, vi } from "vitest";
import type { AudioCapabilities, AudioJob, CreateJobResponse } from "../lib/apiTypes";
import { createAudioJob, fetchAudioCapabilities, getAudioJob } from "./audio";

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

describe("createAudioJob", () => {
  it("issues a multipart POST to /api/v1/audio/jobs with the selected modes and device", async () => {
    const payload: CreateJobResponse = {
      jobId: "aud-1",
      status: "queued",
      statusUrl: "/api/v1/audio/jobs/aud-1",
      downloadUrl: null,
    };
    mockFetchOnce(payload, { status: 202 });
    const file = new File(["binary"], "voice.wav", { type: "audio/wav" });

    const result = await createAudioJob({ file, denoise: "deepfilter", restore: "apollo", device: "dml:0" });

    expect(fetch).toHaveBeenCalledWith("/api/v1/audio/jobs", expect.objectContaining({ method: "POST" }));
    const body = vi.mocked(fetch).mock.calls[0][1]?.body as FormData;
    expect(body.get("file")).toBe(file);
    expect(body.get("denoise")).toBe("deepfilter");
    expect(body.get("restore")).toBe("apollo");
    expect(body.get("device")).toBe("dml:0");
    expect(result).toEqual(payload);
  });

  it("omits denoise, restore and device fields when they are not selected", async () => {
    mockFetchOnce(
      { jobId: "aud-2", status: "queued", statusUrl: "/api/v1/audio/jobs/aud-2", downloadUrl: null },
      { status: 202 },
    );
    const file = new File(["binary"], "voice.wav", { type: "audio/wav" });

    await createAudioJob({ file, denoise: "rnnoise", restore: null, device: null });

    const body = vi.mocked(fetch).mock.calls[0][1]?.body as FormData;
    expect(body.get("denoise")).toBe("rnnoise");
    expect(body.has("restore")).toBe(false);
    expect(body.has("device")).toBe(false);
  });
});

describe("getAudioJob", () => {
  it("issues a GET to /api/v1/audio/jobs/{id} and returns the typed payload", async () => {
    const payload: AudioJob = {
      id: "aud-1",
      status: "running",
      originalFilename: "voice.wav",
      denoise: "deepfilter",
      restore: "apollo",
      device: "dml:0",
      progressPct: 30,
      stages: null,
      error: null,
      downloadUrl: null,
    };
    mockFetchOnce(payload);

    const result = await getAudioJob("aud-1");

    expect(fetch).toHaveBeenCalledWith("/api/v1/audio/jobs/aud-1", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});

describe("fetchAudioCapabilities", () => {
  it("issues a GET to /api/v1/audio/capabilities and returns the typed payload", async () => {
    const payload: AudioCapabilities = { denoiseModes: ["deepfilter", "rnnoise"], restoreAvailable: true };
    mockFetchOnce(payload);

    const result = await fetchAudioCapabilities();

    expect(fetch).toHaveBeenCalledWith("/api/v1/audio/capabilities", expect.objectContaining({ method: "GET" }));
    expect(result).toEqual(payload);
  });
});
