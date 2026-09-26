import { describe, expect, it } from "vitest";
import type { RestoreGeometry } from "../../../lib/restoreApiTypes";
import {
  MAX_STRAIGHTEN_DEG,
  cropFromSelection,
  isNeutralGeometry,
  previewRotationDeg,
  rotateClockwise,
  selectionFromPoints,
  withAngle,
  withoutCrop,
} from "./geometryMath";

const NEUTRAL: RestoreGeometry = { rotate90: 0, crop: null, angle: 0 };
const RECT = { left: 100, top: 50, width: 200, height: 100 };

describe("rotateClockwise", () => {
  it("adds a quarter turn and wraps after four", () => {
    expect(rotateClockwise({ ...NEUTRAL, rotate90: 3 }).rotate90).toBe(0);
    expect(rotateClockwise(NEUTRAL).rotate90).toBe(1);
  });

  it("drops the crop because it was measured on the unrotated image", () => {
    const rotated = rotateClockwise({ rotate90: 0, crop: [1, 2, 3, 4], angle: 2 });

    expect(rotated).toEqual({ rotate90: 1, crop: null, angle: 2 });
  });
});

describe("withAngle", () => {
  it("keeps the crop because straightening keeps the image size", () => {
    expect(withAngle({ rotate90: 1, crop: [1, 2, 3, 4], angle: 0 }, 2.5)).toEqual({
      rotate90: 1,
      crop: [1, 2, 3, 4],
      angle: 2.5,
    });
  });

  it("clamps to the backend limit and rounds to a tenth of a degree", () => {
    expect(withAngle(NEUTRAL, 99).angle).toBe(MAX_STRAIGHTEN_DEG);
    expect(withAngle(NEUTRAL, -99).angle).toBe(-MAX_STRAIGHTEN_DEG);
    expect(withAngle(NEUTRAL, 1.26).angle).toBe(1.3);
  });
});

describe("withoutCrop", () => {
  it("clears only the crop", () => {
    expect(withoutCrop({ rotate90: 2, crop: [1, 2, 3, 4], angle: 1 })).toEqual({
      rotate90: 2,
      crop: null,
      angle: 1,
    });
  });
});

describe("selectionFromPoints", () => {
  it("normalizes a drag in any direction to fractions of the image box", () => {
    expect(selectionFromPoints({ x: 300, y: 150 }, { x: 200, y: 100 }, RECT)).toEqual({
      x: 0.5,
      y: 0.5,
      width: 0.5,
      height: 0.5,
    });
  });

  it("clamps a drag that leaves the image", () => {
    expect(selectionFromPoints({ x: 50, y: 0 }, { x: 400, y: 500 }, RECT)).toEqual({
      x: 0,
      y: 0,
      width: 1,
      height: 1,
    });
  });

  it("returns an empty selection for a box with no size", () => {
    expect(selectionFromPoints({ x: 1, y: 1 }, { x: 2, y: 2 }, { ...RECT, width: 0 })).toEqual({
      x: 0,
      y: 0,
      width: 0,
      height: 0,
    });
  });
});

describe("cropFromSelection", () => {
  it("maps the selection to pixels of the working copy", () => {
    const next = cropFromSelection(NEUTRAL, { x: 0.25, y: 0.5, width: 0.5, height: 0.25 }, {
      width: 400,
      height: 200,
    });

    expect(next).toEqual({ rotate90: 0, crop: [100, 100, 200, 50], angle: 0 });
  });

  it("offsets a new crop by the crop already applied", () => {
    const current: RestoreGeometry = { rotate90: 1, crop: [40, 30, 400, 200], angle: 1 };

    const next = cropFromSelection(current, { x: 0.5, y: 0, width: 0.5, height: 1 }, {
      width: 400,
      height: 200,
    });

    expect(next).toEqual({ rotate90: 1, crop: [240, 30, 200, 200], angle: 1 });
  });

  it("never leaves the working copy when rounding", () => {
    const next = cropFromSelection(NEUTRAL, { x: 0.999, y: 0.999, width: 0.5, height: 0.5 }, {
      width: 3,
      height: 3,
    });

    expect(next).toBeNull();
  });

  it("rejects a selection smaller than the minimum crop", () => {
    const next = cropFromSelection(NEUTRAL, { x: 0, y: 0, width: 0.01, height: 0.5 }, {
      width: 400,
      height: 200,
    });

    expect(next).toBeNull();
  });
});

describe("previewRotationDeg", () => {
  it("turns the counter-clockwise straighten angle into a CSS rotation", () => {
    expect(previewRotationDeg(1, 3.5)).toBe(-2.5);
    expect(previewRotationDeg(2, 2)).toBe(0);
  });
});

describe("isNeutralGeometry", () => {
  it("is true only without rotation, crop or angle", () => {
    expect(isNeutralGeometry(NEUTRAL)).toBe(true);
    expect(isNeutralGeometry({ ...NEUTRAL, angle: 0.1 })).toBe(false);
    expect(isNeutralGeometry({ ...NEUTRAL, crop: [0, 0, 1, 1] })).toBe(false);
    expect(isNeutralGeometry({ ...NEUTRAL, rotate90: 2 })).toBe(false);
  });
});
