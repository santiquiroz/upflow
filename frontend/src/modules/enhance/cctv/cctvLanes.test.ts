import { describe, expect, it } from "vitest";
import type { VideoCapabilities } from "../../../lib/apiTypes";
import { ANALYSIS } from "./cctvFixtures";
import {
  aiLaneState,
  defaultTask,
  estimateAiCpuSeconds,
  formatRoughDuration,
  isVideoTask,
  LANE_TASKS,
  needsAiConfirmation,
  startBlocker,
  type StartInputs,
} from "./cctvLanes";

const CAPS: VideoCapabilities = {
  interpEngines: ["rife"],
  cctvAvailable: true,
  cctvReasonKey: null,
  cctvAiAvailable: false,
  cctvAiReasonKey: "capability.setup.needsGpu",
  cctvUnavailableSteps: [],
};

const READY: StartInputs = {
  modeAvailable: true,
  decodeFailed: false,
  lane: "classic",
  aiAvailable: false,
  task: "clarify",
  noOsd: true,
  osdBoxesConfirmed: false,
  osdBoxCount: 0,
  incompleteStepIds: [],
  trimValid: true,
  caseDetailsValid: true,
  roiBlockerKey: null,
};

describe("tasks", () => {
  it("offers Clarify in the classic lane, Enhance in the AI lane and the multi-frame still in both", () => {
    expect(LANE_TASKS.classic).toEqual(["clarify", "roi_fusion"]);
    expect(LANE_TASKS.ai).toEqual(["enhance", "roi_fusion"]);
    expect(defaultTask("classic")).toBe("clarify");
    expect(defaultTask("ai")).toBe("enhance");
  });

  it("treats the multi-frame still as a still, not a video", () => {
    expect(isVideoTask("clarify")).toBe(true);
    expect(isVideoTask("enhance")).toBe(true);
    expect(isVideoTask("roi_fusion")).toBe(false);
  });

  it("asks for the AI confirmation only before an AI video", () => {
    expect(needsAiConfirmation("enhance")).toBe(true);
    expect(needsAiConfirmation("roi_fusion")).toBe(false);
    expect(needsAiConfirmation("clarify")).toBe(false);
  });
});

describe("AI lane on the CPU", () => {
  it("scales the per-frame cost by the pixel count", () => {
    expect(estimateAiCpuSeconds(100, 1920, 1080)).toBe(1500);
    expect(estimateAiCpuSeconds(100, 960, 1080)).toBe(750);
  });

  it("rounds the estimate to a readable size", () => {
    expect(formatRoughDuration(21600)).toBe("6 h");
    expect(formatRoughDuration(1500)).toBe("25 min");
    expect(formatRoughDuration(20)).toBe("1 min");
  });

  it("says how long the CPU would take when the only thing missing is a GPU", () => {
    expect(aiLaneState(CAPS, ANALYSIS)).toEqual({
      available: false,
      reason: { key: "cctv.ai.cpuBlocked", params: { eta: "2 h" } },
    });
  });

  it("falls back to the capability reason when there is no frame count to estimate from", () => {
    expect(aiLaneState(CAPS, null)).toEqual({
      available: false,
      reason: { key: "capability.setup.needsGpu", params: {} },
    });
  });

  it("is available when the backend says so", () => {
    expect(aiLaneState({ ...CAPS, cctvAiAvailable: true, cctvAiReasonKey: null }, ANALYSIS)).toEqual({
      available: true,
      reason: null,
    });
  });

  it("stays closed while the capabilities load", () => {
    expect(aiLaneState(undefined, ANALYSIS).available).toBe(false);
  });
});

describe("startBlocker", () => {
  it("lets a complete classic job start", () => {
    expect(startBlocker(READY)).toBeNull();
  });

  it("needs confirmed on-screen text boxes or 'No on-screen text'", () => {
    expect(startBlocker({ ...READY, noOsd: false })).toEqual({ key: "cctv.osd.confirm" });
    expect(startBlocker({ ...READY, noOsd: false, osdBoxesConfirmed: true, osdBoxCount: 0 })).toEqual({
      key: "cctv.osd.confirm",
    });
    expect(startBlocker({ ...READY, noOsd: false, osdBoxesConfirmed: true, osdBoxCount: 2 })).toBeNull();
  });

  it("explains why, in priority order", () => {
    expect(startBlocker({ ...READY, modeAvailable: false })).toEqual({ key: "cctv.blocked.modeUnavailable" });
    expect(startBlocker({ ...READY, decodeFailed: true })).toEqual({ key: "cctv.undecodable" });
    expect(startBlocker({ ...READY, lane: "ai", task: "enhance" })).toEqual({ key: "cctv.blocked.aiUnavailable" });
    expect(startBlocker({ ...READY, incompleteStepIds: ["crop"] })).toEqual({ key: "cctv.blocked.incompleteSteps" });
    expect(startBlocker({ ...READY, trimValid: false })).toEqual({ key: "cctv.trim.invalid" });
    expect(startBlocker({ ...READY, caseDetailsValid: false })).toEqual({ key: "cctv.case.offsetInvalid" });
  });

  it("gates the multi-frame still on its region instead of the on-screen text", () => {
    const roi = { ...READY, task: "roi_fusion" as const, noOsd: false };

    expect(startBlocker(roi)).toBeNull();
    expect(startBlocker({ ...roi, roiBlockerKey: "cctv.roi.blocked.box" })).toEqual({ key: "cctv.roi.blocked.box" });
  });

  it("ignores a stale region outside the multi-frame still", () => {
    expect(startBlocker({ ...READY, roiBlockerKey: "cctv.roi.blocked.box" })).toBeNull();
  });
});
