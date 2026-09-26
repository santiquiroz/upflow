import { describe, expect, it } from "vitest";
import type { BinaryMask } from "../../editor/maskCanvas";
import { holeWidths, largeHoleCount } from "./damageHoles";

function maskOf(width: number, height: number, isMarked: (x: number, y: number) => boolean): BinaryMask {
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) data[y * width + x] = isMarked(x, y) ? 1 : 0;
  }
  return { width, height, data };
}

const inRect = (x: number, y: number, left: number, top: number, right: number, bottom: number) =>
  x >= left && x < right && y >= top && y < bottom;

describe("holeWidths", () => {
  it("matches scratch_fill.hole_widths on the same mask (square, edge-to-edge line, disk joined to a band)", () => {
    const mask = maskOf(60, 40, (x, y) => {
      const disk = (x + 0.5 - 48) ** 2 + (y + 0.5 - 15) ** 2 <= 81;
      return inRect(x, y, 2, 2, 32, 32) || y === 36 || disk || inRect(x, y, 36, 0, 39, 31);
    });

    const widths = holeWidths(mask).sort((a, b) => a - b);

    expect(widths).toHaveLength(3);
    expect(widths[0]).toBeCloseTo(2, 4);
    expect(widths[1]).toBeCloseTo(16.9706, 4);
    expect(widths[2]).toBeCloseTo(30, 4);
  });

  it("is empty without marked pixels", () => {
    expect(holeWidths(maskOf(5, 5, () => false))).toEqual([]);
  });
});

describe("largeHoleCount", () => {
  it("counts only components wider than 24 px", () => {
    const band = (width: number) => maskOf(80, 40, (x) => x >= 10 && x < 10 + width);

    expect(largeHoleCount(band(24))).toBe(0);
    expect(largeHoleCount(band(26))).toBe(1);
  });

  it("treats the photo edge as the border of a hole", () => {
    const corner = maskOf(50, 50, (x, y) => x < 30 && y < 30);

    expect(largeHoleCount(corner)).toBe(1);
  });

  it("does not join pieces that only touch at a corner", () => {
    const mask = maskOf(4, 4, (x, y) => (x === 0 && y === 0) || (x === 1 && y === 1));

    expect(holeWidths(mask)).toEqual([2, 2]);
  });
});
