import { afterEach, describe, expect, it, vi } from "vitest";
import { capturedUploads as uploads, mockUploadOnce } from "../lib/uploadTestStub";
import {
  analyzePhoto,
  createRestoreJob,
  getRestoreCapabilities,
  recomposeFaces,
  restoreArtifactUrl,
  setPhotoGeometry,
  uploadDamageMask,
  type CreateRestoreJobParams,
} from "./restore";

function mockFetchOnce(body: unknown) {
  const response = new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));
}

function lastFetchBody(): unknown {
  const init = vi.mocked(fetch).mock.calls[0][1] as RequestInit;
  return JSON.parse(init.body as string);
}

function sentBody(): FormData {
  return uploads[0].body;
}

function jobParams(overrides: Partial<CreateRestoreJobParams> = {}): CreateRestoreJobParams {
  return {
    source: { token: "tok-1" },
    steps: ["repair", "denoise"],
    options: {},
    scale: 1,
    modelId: null,
    device: null,
    outputFormat: "png",
    ...overrides,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("restore service", () => {
  it("reads the capabilities with a GET", async () => {
    mockFetchOnce({ steps: [], presets: [], halftoneDenoiseLimit: 0.2 });

    await getRestoreCapabilities();

    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/restore/capabilities",
      expect.objectContaining({ method: "GET" }),
    );
  });

  it("uploads the photo to analyze as multipart field file", async () => {
    mockUploadOnce({ token: "tok-1" });
    const photo = new File(["x"], "scan.tif", { type: "image/tiff" });

    await analyzePhoto(photo);

    expect(uploads[0].url).toBe("/api/v1/restore/analyze");
    expect(sentBody().get("file")).toBe(photo);
  });

  it("posts the geometry as JSON to the session", async () => {
    mockFetchOnce({ token: "tok-1" });

    await setPhotoGeometry("tok-1", { rotate90: 1, crop: [0, 0, 10, 20], angle: -1.5 });

    expect(vi.mocked(fetch).mock.calls[0][0]).toBe("/api/v1/restore/analysis/tok-1/geometry");
    expect(lastFetchBody()).toEqual({ rotate90: 1, crop: [0, 0, 10, 20], angle: -1.5 });
  });

  it("uploads the painted mask as a PNG file", async () => {
    mockUploadOnce({ coverage: 0.02, width: 10, height: 10 });

    await uploadDamageMask("tok-1", new Blob(["png"], { type: "image/png" }));

    expect(uploads[0].url).toBe("/api/v1/restore/analysis/tok-1/mask");
    expect((sentBody().get("file") as File).name).toBe("damage_mask.png");
  });

  it("creates a job from the session token with the steps as CSV", async () => {
    mockUploadOnce({ jobId: "job-1" });

    await createRestoreJob(jobParams());

    const body = sentBody();
    expect(uploads[0].url).toBe("/api/v1/restore/jobs");
    expect(body.get("token")).toBe("tok-1");
    expect(body.get("file")).toBeNull();
    expect(body.get("restore_steps")).toBe("repair,denoise");
    expect(body.get("scale")).toBe("1");
    expect(body.get("output_format")).toBe("png");
  });

  it("omits the options, model and device when they are not chosen", async () => {
    mockUploadOnce({ jobId: "job-1" });

    await createRestoreJob(jobParams());

    const body = sentBody();
    expect(body.get("restore_options")).toBeNull();
    expect(body.get("model_id")).toBeNull();
    expect(body.get("model_name")).toBeNull();
    expect(body.get("device")).toBeNull();
  });

  it("sends a file source, the options as JSON, the AI model and the device", async () => {
    mockUploadOnce({ jobId: "job-1" });
    const photo = new File(["x"], "old.jpg", { type: "image/jpeg" });

    await createRestoreJob(
      jobParams({
        source: { file: photo },
        options: { upscale_mode: "ai", denoise: { strength: 0.3, keep_grain: 0.25 } },
        scale: 2,
        modelId: "realesrgan-x2",
        device: "dml:0",
      }),
    );

    const body = sentBody();
    expect(body.get("file")).toBe(photo);
    expect(body.get("token")).toBeNull();
    expect(JSON.parse(body.get("restore_options") as string)).toEqual({
      upscale_mode: "ai",
      denoise: { strength: 0.3, keep_grain: 0.25 },
    });
    expect(body.get("model_id")).toBe("realesrgan-x2");
    expect(body.get("model_name")).toBe("realesrgan-x2");
    expect(body.get("device")).toBe("dml:0");
  });

  it("posts the face choices to recompose", async () => {
    mockFetchOnce({ sidecar: {} });

    await recomposeFaces("job-1", { 0: { enabled: true, blend: 0.4 } });

    expect(vi.mocked(fetch).mock.calls[0][0]).toBe("/api/v1/restore/jobs/job-1/recompose");
    expect(lastFetchBody()).toEqual({ faces: { 0: { enabled: true, blend: 0.4 } } });
  });

  it("builds artifact URLs with the name escaped", () => {
    expect(restoreArtifactUrl("job-1", "face:2:after")).toBe(
      "/api/v1/jobs/job-1/artifacts/face%3A2%3Aafter",
    );
  });
});
