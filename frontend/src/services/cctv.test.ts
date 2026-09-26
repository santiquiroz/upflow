import { afterEach, describe, expect, it, vi } from "vitest";
import { capturedUploads, mockUploadOnce } from "../lib/uploadTestStub";
import { ANALYSIS } from "../modules/enhance/cctv/cctvFixtures";
import {
  analyzeCctv,
  cctvFilterPreviewUrl,
  cctvFrameUrl,
  checkCctvOsd,
  createCctvJob,
  getCctvAnalysis,
  getCctvPresets,
  toAnalyzeReply,
  type CctvAnalysisJob,
  type CctvJobRequest,
} from "./cctv";

function mockFetchOnce(body: unknown, init: ResponseInit = { status: 200 }) {
  const response = new Response(JSON.stringify(body), {
    ...init,
    headers: { "Content-Type": "application/json" },
  });
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(response));
}

const PENDING: CctvAnalysisJob = {
  analysisJobId: "an-1",
  status: "running",
  statusUrl: "/api/v1/video/cctv/analysis/an-1",
  result: null,
  error: null,
  errorKey: null,
};

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("toAnalyzeReply", () => {
  it("treats a body with the analysis job id as still running", () => {
    expect(toAnalyzeReply(PENDING)).toEqual({ kind: "pending", analysisJobId: "an-1" });
  });

  it("returns the analysis when the backend finished inside the synchronous window", () => {
    expect(toAnalyzeReply(ANALYSIS)).toEqual({ kind: "done", analysis: ANALYSIS });
  });
});

describe("analyzeCctv", () => {
  it("uploads the recording as multipart under the file field", async () => {
    mockUploadOnce(ANALYSIS);
    const file = new File(["IMKH"], "ch01.dav");

    const reply = await analyzeCctv(file);

    expect(capturedUploads[0].url).toBe("/api/v1/video/cctv/analyze");
    expect(capturedUploads[0].body.get("file")).toBe(file);
    expect(reply).toEqual({ kind: "done", analysis: ANALYSIS });
  });

  it("hands back the analysis job id when the backend answers 202", async () => {
    mockUploadOnce(PENDING, 202);

    await expect(analyzeCctv(new File(["x"], "long.264"))).resolves.toEqual({
      kind: "pending",
      analysisJobId: "an-1",
    });
  });

  it("keeps the translation key of a rejected analysis", async () => {
    mockUploadOnce({ detail: { key: "cctv.error.noVideoStream", reason: "No video stream." } }, 400);

    await expect(analyzeCctv(new File(["x"], "audio.wav"))).rejects.toMatchObject({
      key: "cctv.error.noVideoStream",
    });
  });
});

describe("polling and presets", () => {
  it("reads the analysis status by id", async () => {
    mockFetchOnce({ ...PENDING, status: "completed", result: ANALYSIS });

    const job = await getCctvAnalysis("an-1");

    expect(fetch).toHaveBeenCalledWith("/api/v1/video/cctv/analysis/an-1", { method: "GET" });
    expect(job.result).toEqual(ANALYSIS);
  });

  it("reads the presets and the step catalog", async () => {
    mockFetchOnce({ presets: [], steps: { classic: [], ai: [] } });

    await getCctvPresets();

    expect(fetch).toHaveBeenCalledWith("/api/v1/video/cctv/presets", { method: "GET" });
  });
});

describe("createCctvJob", () => {
  it("posts the camelCase request as JSON", async () => {
    mockFetchOnce({ jobId: "job-1", status: "queued" }, { status: 202 });
    const request: CctvJobRequest = {
      token: "tok-1",
      task: "clarify",
      preset: "day",
      steps: [{ id: "deblock", params: { filter: "deblock", filter_type: "weak" } }],
      osdBoxes: [],
      osdBoxesConfirmed: false,
      noOsd: true,
      trim: [10, 20],
    };

    await createCctvJob(request);

    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe("/api/v1/video/cctv/jobs");
    expect(JSON.parse(String((init as RequestInit).body))).toEqual(request);
  });
});

describe("frame previews", () => {
  it("asks for one decoded frame of the session", () => {
    expect(cctvFrameUrl("tok/1", 42)).toBe("/api/v1/video/cctv/tok%2F1/preview?frame=42");
  });

  it("sends the preview steps as a JSON array", () => {
    const url = new URL(cctvFilterPreviewUrl("tok-1", 7, [{ id: "gray", params: { filter: "gray" } }]), "http://localhost");

    expect(url.pathname).toBe("/api/v1/video/cctv/tok-1/preview");
    expect(url.searchParams.get("frame")).toBe("7");
    expect(JSON.parse(url.searchParams.get("steps") ?? "")).toEqual([{ id: "gray", params: { filter: "gray" } }]);
  });
});

describe("checkCctvOsd", () => {
  it("posts the boxes and the frame the user looked at", async () => {
    mockFetchOnce({ checks: [], warnings: [] });

    await checkCctvOsd("tok-1", [[0, 0, 10, 10]], 12);

    const [url, init] = vi.mocked(fetch).mock.calls[0];
    expect(url).toBe("/api/v1/video/cctv/tok-1/osd-check");
    expect(JSON.parse(String((init as RequestInit).body))).toEqual({ boxes: [[0, 0, 10, 10]], frame: 12 });
  });
});
