import { describe, expect, it } from "vitest";
import { ANALYSIS } from "./cctvFixtures";
import { analysisJobError, resolveAnalysis, sessionPhase } from "./useCctvSession";

const IDLE = { isUploading: false, uploadPercent: null, error: null, analysis: null, waitingForAnalysis: false };
const JOB = { analysisJobId: "an-1", statusUrl: "/api/v1/video/cctv/analysis/an-1", error: null, errorKey: null };

describe("sessionPhase", () => {
  it("moves from uploading to analyzing once the whole file is up", () => {
    expect(sessionPhase({ ...IDLE, isUploading: true, uploadPercent: 40 })).toBe("uploading");
    expect(sessionPhase({ ...IDLE, isUploading: true, uploadPercent: 100 })).toBe("analyzing");
  });

  it("keeps analyzing while a long analysis is polled", () => {
    expect(sessionPhase({ ...IDLE, waitingForAnalysis: true })).toBe("analyzing");
  });

  it("reports a failure before a stale analysis", () => {
    expect(sessionPhase({ ...IDLE, error: { key: null, message: "x" }, analysis: ANALYSIS })).toBe("failed");
    expect(sessionPhase({ ...IDLE, analysis: ANALYSIS })).toBe("ready");
    expect(sessionPhase(IDLE)).toBe("idle");
  });
});

describe("resolveAnalysis", () => {
  it("takes the synchronous result or the completed poll", () => {
    const pending = { kind: "pending" as const, analysisJobId: "an-1" };

    expect(resolveAnalysis({ kind: "done", analysis: ANALYSIS }, undefined)).toBe(ANALYSIS);
    expect(resolveAnalysis(pending, { ...JOB, status: "completed", result: ANALYSIS })).toBe(ANALYSIS);
    expect(resolveAnalysis(pending, { ...JOB, status: "running", result: null })).toBeNull();
  });

  it("turns a failed analysis job into a keyed error", () => {
    const failed = { ...JOB, status: "failed" as const, result: null, error: "Boom", errorKey: "cctv.error.ingestFailed" };

    expect(analysisJobError(failed)).toEqual({ key: "cctv.error.ingestFailed", message: "Boom" });
  });
});
