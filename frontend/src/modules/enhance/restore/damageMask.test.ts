import { describe, expect, it } from "vitest";
import type { BinaryMask, BrushStroke } from "../../editor/maskCanvas";
import {
  composeDamageMask,
  growMask,
  maskCoverage,
  maskSettingsFrom,
  maskReviewKey,
  needsMaskReview,
  overlayPixels,
  sensitivityThreshold,
  thresholdProbability,
  withMaskChoice,
  type ProbabilityMap,
} from "./damageMask";

function maskFrom(rows: string[]): BinaryMask {
  const width = rows[0].length;
  const data = Uint8Array.from(rows.join("").split(""), (cell) => (cell === "#" ? 1 : 0));
  return { width, height: rows.length, data };
}

function rowsOf(mask: BinaryMask): string[] {
  const cells = Array.from(mask.data, (value) => (value ? "#" : "."));
  return Array.from({ length: mask.height }, (_, y) => cells.slice(y * mask.width, (y + 1) * mask.width).join(""));
}

function probability(values: number[], width = values.length): ProbabilityMap {
  return { width, height: values.length / width, data: Uint8Array.from(values) };
}

describe("sensitivityThreshold", () => {
  it("mirrors scratch_detect: 0 → 0.6, 0.5 → BOPBTL's 0.4, 1 → 0.2", () => {
    expect(sensitivityThreshold(0)).toBeCloseTo(0.6);
    expect(sensitivityThreshold(0.5)).toBeCloseTo(0.4);
    expect(sensitivityThreshold(1)).toBeCloseTo(0.2);
  });
});

describe("thresholdProbability", () => {
  it("marks only what is strictly above the threshold of the chosen sensitivity", () => {
    const map = probability([0, 51, 102, 103, 153, 255]);

    expect(Array.from(thresholdProbability(map, 0.5).data)).toEqual([0, 0, 0, 1, 1, 1]);
    expect(Array.from(thresholdProbability(map, 1).data)).toEqual([0, 0, 1, 1, 1, 1]);
    expect(Array.from(thresholdProbability(map, 0).data)).toEqual([0, 0, 0, 0, 0, 1]);
  });
});

describe("growMask", () => {
  const dot = maskFrom([".....", ".....", "..#..", ".....", "....."]);

  it("grows a square around every marked pixel, like dilate_mask", () => {
    expect(rowsOf(growMask(dot, 1))).toEqual([".....", ".###.", ".###.", ".###.", "....."]);
  });

  it("shrinks from the inside and treats the photo edge as marked, like cv2.erode", () => {
    const block = maskFrom(["###..", "###..", "###..", ".....", "....."]);

    expect(rowsOf(growMask(block, -1))).toEqual(["##...", "##...", ".....", ".....", "....."]);
  });

  it("returns the same mask at zero", () => {
    expect(growMask(dot, 0)).toBe(dot);
  });
});

describe("composeDamageMask", () => {
  const map = probability([0, 0, 0, 0, 255, 255, 0, 0, 0], 3);

  it("thresholds, grows and then applies the strokes in order", () => {
    const strokes: BrushStroke[] = [
      { mode: "paint", radius: 0.5, points: [{ x: 0.5, y: 0.5 }] },
      { mode: "erase", radius: 0.5, points: [{ x: 1.5, y: 1.5 }] },
    ];

    const mask = composeDamageMask(map, { width: 3, height: 3 }, { sensitivity: 0.5, growPx: 0 }, strokes);

    expect(rowsOf(mask)).toEqual(["#..", "..#", "..."]);
  });

  it("starts from an empty mask when there is no detector map", () => {
    const mask = composeDamageMask(null, { width: 2, height: 2 }, { sensitivity: 0.5, growPx: 2 }, []);

    expect(Array.from(mask.data)).toEqual([0, 0, 0, 0]);
  });

  it("ignores a map that no longer matches the working copy", () => {
    const mask = composeDamageMask(map, { width: 4, height: 2 }, { sensitivity: 0.5, growPx: 0 }, []);

    expect(mask.data).toHaveLength(8);
    expect(maskCoverage(mask)).toBe(0);
  });
});

describe("maskCoverage", () => {
  it("is the marked fraction of the photo", () => {
    expect(maskCoverage(maskFrom(["#...", "#..."]))).toBe(0.25);
  });
});

describe("needsMaskReview", () => {
  it("asks for a review above 3% coverage, with large holes or with damage over a face", () => {
    expect(needsMaskReview({ coverage: 0.03, largeHoles: 0, damageOverFaces: false })).toBe(false);
    expect(needsMaskReview({ coverage: 0.031, largeHoles: 0, damageOverFaces: false })).toBe(true);
    expect(needsMaskReview({ coverage: 0.01, largeHoles: 1, damageOverFaces: false })).toBe(true);
    expect(needsMaskReview({ coverage: 0.01, largeHoles: 0, damageOverFaces: true })).toBe(true);
  });
});

describe("maskReviewKey", () => {
  it("changes when the sensitivity, the grow or the strokes change", () => {
    const base = maskReviewKey({ sensitivity: 0.5, growPx: 0 }, 0);

    expect(maskReviewKey({ sensitivity: 0.5, growPx: 0 }, 0)).toBe(base);
    expect(maskReviewKey({ sensitivity: 0.55, growPx: 0 }, 0)).not.toBe(base);
    expect(maskReviewKey({ sensitivity: 0.5, growPx: 1 }, 0)).not.toBe(base);
    expect(maskReviewKey({ sensitivity: 0.5, growPx: 0 }, 1)).not.toBe(base);
  });
});

describe("overlayPixels", () => {
  it("paints the mask solid and hints the probability below the threshold", () => {
    const map = probability([0, 128, 255]);
    const mask = maskFrom(["..#"]);

    const pixels = overlayPixels(map, mask);

    expect(pixels).toHaveLength(12);
    expect(pixels[3]).toBe(0);
    expect(pixels[7]).toBeGreaterThan(0);
    expect(pixels[7]).toBeLessThan(pixels[11]);
  });

  it("paints only the mask without a detector map", () => {
    const pixels = overlayPixels(null, maskFrom(["#."]));

    expect(pixels[3]).toBeGreaterThan(0);
    expect(pixels[7]).toBe(0);
  });
});

describe("withMaskChoice", () => {
  const options = { preset: "gentle", repair: { sensitivity: 0.5 } };

  it("leaves the request alone when no painted mask is in play", () => {
    expect(withMaskChoice(options, false, false)).toBe(options);
  });

  it("asks for the painted mask when there are edits", () => {
    expect(withMaskChoice(options, true, true)).toEqual({ preset: "gentle", repair: { sensitivity: 0.5, use_user_mask: true } });
  });

  it("tells the backend to ignore a mask uploaded earlier once the edits are gone", () => {
    expect(withMaskChoice(options, false, true).repair).toEqual({ sensitivity: 0.5, use_user_mask: false });
  });

  it("does nothing without the repair step", () => {
    const noRepair = { preset: "gentle" };

    expect(withMaskChoice(noRepair, true, true)).toBe(noRepair);
  });
});

describe("maskSettingsFrom", () => {
  it("reads the repair options with the backend defaults", () => {
    expect(maskSettingsFrom({ sensitivity: 0.7, grow_px: -2 })).toEqual({ sensitivity: 0.7, growPx: -2 });
    expect(maskSettingsFrom({})).toEqual({ sensitivity: 0.5, growPx: 0 });
  });
});
