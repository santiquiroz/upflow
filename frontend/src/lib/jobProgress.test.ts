import { describe, expect, it } from "vitest";
import type { JobStage } from "./apiTypes";
import {
  areFramesReportable,
  deriveStepper,
  isProgressDeterminate,
  toMonotonicProgressPct,
} from "./jobProgress";

function stage(key: string, status: JobStage["status"]): JobStage {
  return { key, label: `${key} label`, weight: 0.25, status };
}

describe("deriveStepper", () => {
  it("returns an empty list when stages are absent", () => {
    expect(deriveStepper(undefined)).toEqual([]);
  });

  it("maps a done stage to the done icon state", () => {
    const items = deriveStepper([stage("probing", "done")]);

    expect(items).toEqual([{ key: "probing", label: "probing label", iconState: "done" }]);
  });

  it("maps an active stage to the active icon state", () => {
    const items = deriveStepper([stage("upscaling_frames", "active")]);

    expect(items[0].iconState).toBe("active");
  });

  it("maps a pending stage to the pending icon state", () => {
    const items = deriveStepper([stage("encoding_video", "pending")]);

    expect(items[0].iconState).toBe("pending");
  });

  it("preserves stage order", () => {
    const items = deriveStepper([stage("probing", "done"), stage("extracting_frames", "active"), stage("encoding_video", "pending")]);

    expect(items.map((item) => item.key)).toEqual(["probing", "extracting_frames", "encoding_video"]);
  });
});

describe("isProgressDeterminate", () => {
  it("is true for a finite number", () => {
    expect(isProgressDeterminate(42)).toBe(true);
  });

  it("is true for zero", () => {
    expect(isProgressDeterminate(0)).toBe(true);
  });

  it("is false for null", () => {
    expect(isProgressDeterminate(null)).toBe(false);
  });

  it("is false for undefined", () => {
    expect(isProgressDeterminate(undefined)).toBe(false);
  });
});

describe("toMonotonicProgressPct", () => {
  it("advances to a higher candidate", () => {
    expect(toMonotonicProgressPct(10, 25)).toBe(25);
  });

  it("keeps the previous max when the candidate regresses", () => {
    expect(toMonotonicProgressPct(50, 30)).toBe(50);
  });

  it("keeps the previous max when the candidate is null", () => {
    expect(toMonotonicProgressPct(50, null)).toBe(50);
  });

  it("keeps the previous max when the candidate is undefined", () => {
    expect(toMonotonicProgressPct(50, undefined)).toBe(50);
  });
});

describe("areFramesReportable", () => {
  it("is true when both frame counts are present and total is positive", () => {
    expect(areFramesReportable(120, 600)).toBe(true);
  });

  it("is false when framesTotal is null (e.g. VFR source)", () => {
    expect(areFramesReportable(120, null)).toBe(false);
  });

  it("is false when framesTotal is zero", () => {
    expect(areFramesReportable(0, 0)).toBe(false);
  });

  it("is false when framesDone is undefined", () => {
    expect(areFramesReportable(undefined, 600)).toBe(false);
  });
});
