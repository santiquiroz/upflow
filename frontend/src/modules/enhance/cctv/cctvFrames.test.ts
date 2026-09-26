import { describe, expect, it } from "vitest";
import type { CctvFrameIndex } from "../../../services/cctv";
import {
  clampFrame,
  displaySizeOf,
  formatTimecode,
  frameElapsedSeconds,
  frameTimecode,
  fullRange,
  isTrimValid,
  storedSizeOf,
  trimFrameCount,
} from "./cctvFrames";
import { ANALYSIS } from "./cctvFixtures";

const INDEX = ANALYSIS.frameIndex as CctvFrameIndex;
const CFR_INDEX: CctvFrameIndex = { ...INDEX, isVfr: false, gaps: [], medianDelta: 0.04, measuredFps: 25 };

describe("clampFrame", () => {
  it("keeps the frame inside the recording", () => {
    expect(clampFrame(-3, 750)).toBe(0);
    expect(clampFrame(9000, 750)).toBe(749);
    expect(clampFrame(12.7, 750)).toBe(12);
    expect(clampFrame(Number.NaN, 750)).toBe(0);
  });
});

describe("frame sizes", () => {
  it("draws on the stored size and shows the real aspect of a Lite recording", () => {
    expect(storedSizeOf(ANALYSIS)).toEqual({ width: 960, height: 1080 });
    expect(displaySizeOf(ANALYSIS)).toEqual({ width: 1920, height: 1080 });
  });

  it("shows the stored size when the recording isn't Lite", () => {
    const plain = { ...ANALYSIS, video: { ...ANALYSIS.video, width: 1920, lite: null } };

    expect(displaySizeOf(plain)).toEqual({ width: 1920, height: 1080 });
  });
});

describe("frame timecode", () => {
  it("adds the length of the gaps the recorder left", () => {
    expect(frameElapsedSeconds(INDEX, 10)).toBeCloseTo(0.8);
    // Hueco de 0,8 s despues del cuadro 10: el 11 cae en 1,6 s.
    expect(frameElapsedSeconds(INDEX, 11)).toBeCloseTo(1.6);
  });

  it("uses the measured frame rate when the index has no median step", () => {
    const index = { ...CFR_INDEX, medianDelta: null };

    expect(frameElapsedSeconds(index, 50)).toBeCloseTo(2);
  });

  it("has no timecode without an index", () => {
    expect(frameElapsedSeconds(null, 5)).toBeNull();
    expect(frameTimecode(null, 5)).toBeNull();
  });

  it("marks the timecode as approximate for a variable frame rate", () => {
    expect(frameTimecode(INDEX, 11)).toBe("≈ 00:00:01.600");
    expect(frameTimecode(CFR_INDEX, 25 * 3661)).toBe("01:01:01.000");
  });

  it("formats hours, minutes, seconds and milliseconds", () => {
    expect(formatTimecode(0)).toBe("00:00:00.000");
    expect(formatTimecode(59.9996)).toBe("00:01:00.000");
    expect(formatTimecode(3725.25)).toBe("01:02:05.250");
  });
});

describe("trim range", () => {
  it("starts from the whole recording", () => {
    expect(fullRange(750)).toEqual([0, 749]);
    expect(trimFrameCount([10, 19])).toBe(10);
  });

  it("accepts no trim and ordered ranges inside the recording", () => {
    expect(isTrimValid(null, 750)).toBe(true);
    expect(isTrimValid([0, 749], 750)).toBe(true);
    expect(isTrimValid([5, 5], 750)).toBe(true);
  });

  it("rejects reversed, negative or out-of-range trims", () => {
    expect(isTrimValid([20, 10], 750)).toBe(false);
    expect(isTrimValid([-1, 10], 750)).toBe(false);
    expect(isTrimValid([0, 750], 750)).toBe(false);
    expect(isTrimValid([0.5, 10], 750)).toBe(false);
  });
});
