import { describe, expect, it } from "vitest";

import { maskHasPixels, maskToOpaquePixels, subtractProtectedPixels } from "./protectMask";

describe("maskHasPixels", () => {
  it("reports empty and fully transparent masks as empty even with colored RGB channels", () => {
    expect(maskHasPixels(new Uint8ClampedArray())).toBe(false);
    expect(maskHasPixels(new Uint8ClampedArray([255, 80, 20, 0, 255, 255, 255, 0]))).toBe(false);
  });

  it.each([1, 128, 255])("detects alpha %i at the last pixel", (alpha) => {
    const pixels = new Uint8ClampedArray(3 * 2 * 4);
    pixels[pixels.length - 1] = alpha;

    expect(maskHasPixels(pixels)).toBe(true);
    expect(pixels[pixels.length - 1]).toBe(alpha);
  });
});

describe("maskToOpaquePixels", () => {
  it("exports white protection and black background using alpha without changing dimensions or input", () => {
    const input = new Uint8ClampedArray([
      255, 255, 255, 0, 0, 0, 0, 1,
      12, 34, 56, 128, 50, 60, 70, 255,
    ]);
    const original = input.slice();

    const output = maskToOpaquePixels(input);

    expect(output).toEqual(new Uint8ClampedArray([
      0, 0, 0, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255,
    ]));
    expect(output.length / 4).toBe(2 * 2);
    expect(output).not.toBe(input);
    expect(input).toEqual(original);
  });

  it("exports an empty mask without allocating pixels", () => {
    expect(maskToOpaquePixels(new Uint8ClampedArray())).toEqual(new Uint8ClampedArray());
  });

  it("exports a fully protected mask as opaque white for every pixel", () => {
    const input = new Uint8ClampedArray(3 * 2 * 4).fill(255);

    expect(maskToOpaquePixels(input)).toEqual(new Uint8ClampedArray(3 * 2 * 4).fill(255));
  });
});

describe("subtractProtectedPixels", () => {
  it("removes protected alpha including partial protection while preserving RGB, unprotected pixels, and inputs", () => {
    const edit = new Uint8ClampedArray([
      255, 20, 30, 200, 40, 50, 60, 128,
      70, 80, 90, 255, 100, 110, 120, 0,
    ]);
    const protect = new Uint8ClampedArray([
      255, 255, 255, 0, 0, 0, 0, 1,
      0, 255, 0, 255, 255, 255, 255, 128,
    ]);
    const originalEdit = edit.slice();
    const originalProtect = protect.slice();

    const output = subtractProtectedPixels(edit, protect);

    expect(output).toEqual(new Uint8ClampedArray([
      255, 20, 30, 200, 40, 50, 60, 0,
      70, 80, 90, 0, 100, 110, 120, 0,
    ]));
    expect(output.length / 4).toBe(2 * 2);
    expect(output).not.toBe(edit);
    expect(edit).toEqual(originalEdit);
    expect(protect).toEqual(originalProtect);
  });

  it("keeps the whole edit mask when no protection exists", () => {
    const edit = new Uint8ClampedArray([255, 0, 0, 255]);

    expect(subtractProtectedPixels(edit, new Uint8ClampedArray(4))).toEqual(edit);
  });

  it("removes every editable pixel for full protection", () => {
    const edit = new Uint8ClampedArray(3 * 2 * 4).fill(255);

    const output = subtractProtectedPixels(edit, edit);

    expect(maskHasPixels(output)).toBe(false);
    expect(output.filter((_, offset) => offset % 4 !== 3)).toEqual(new Uint8ClampedArray(18).fill(255));
    expect(edit).toEqual(new Uint8ClampedArray(3 * 2 * 4).fill(255));
  });

  it("accepts two empty masks", () => {
    expect(subtractProtectedPixels(new Uint8ClampedArray(), new Uint8ClampedArray())).toEqual(new Uint8ClampedArray());
  });

  it.each([[4, 8], [8, 4], [0, 4]])("rejects mismatched pixel lengths %i and %i", (editLength, protectLength) => {
    expect(() => subtractProtectedPixels(
      new Uint8ClampedArray(editLength),
      new Uint8ClampedArray(protectLength),
    )).toThrow(RangeError);
  });
});
