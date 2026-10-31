import { describe, expect, it } from "vitest";
import { cloneWithMask, restoreWithMask } from "./retouch";

// jsdom no trae ImageData: los tests usan objetos con la misma forma.
function image(width: number, height: number, fill: (index: number) => number[]): ImageData {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    data.set(fill(pixel), pixel * 4);
  }
  return { width, height, data } as ImageData;
}

function solid(width: number, height: number, [r, g, b]: number[], alpha = 255): ImageData {
  return image(width, height, () => [r, g, b, alpha]);
}

function maskAt(width: number, height: number, points: number[][]): ImageData {
  return image(width, height, (pixel) => {
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    const painted = points.some(([px, py]) => px === x && py === y);
    return [0, 0, 0, painted ? 255 : 0];
  });
}

function pixel(img: ImageData, x: number, y: number): number[] {
  const index = (y * img.width + x) * 4;
  return Array.from(img.data.slice(index, index + 4));
}

describe("restoreWithMask", () => {
  it("restoreWithMask returns the original pixel where the mask is opaque", () => {
    const out = restoreWithMask(solid(4, 4, [10, 10, 10]), solid(4, 4, [200, 0, 0]), maskAt(4, 4, [[1, 1]]));
    expect(pixel(out, 1, 1)).toEqual([200, 0, 0, 255]);
    expect(pixel(out, 0, 0)).toEqual([10, 10, 10, 255]);
  });

  it("blends by the mask alpha in between", () => {
    const mask = image(1, 1, () => [0, 0, 0, 128]);
    const out = restoreWithMask(solid(1, 1, [0, 0, 0]), solid(1, 1, [255, 255, 255]), mask);
    expect(pixel(out, 0, 0)[0]).toBeGreaterThan(120);
    expect(pixel(out, 0, 0)[0]).toBeLessThan(136);
  });

  it("does not mutate its inputs", () => {
    const current = solid(2, 2, [10, 10, 10]);
    const original = solid(2, 2, [200, 0, 0]);
    const mask = maskAt(2, 2, [[0, 0]]);
    restoreWithMask(current, original, mask);
    expect(pixel(current, 0, 0)).toEqual([10, 10, 10, 255]);
    expect(pixel(original, 0, 0)).toEqual([200, 0, 0, 255]);
  });
});

describe("cloneWithMask", () => {
  it("cloneWithMask copies from the offset source", () => {
    const current = image(4, 1, (x) => [x * 10, 0, 0, 255]);
    const out = cloneWithMask(current, maskAt(4, 1, [[0, 0]]), 2, 0);
    expect(pixel(out, 0, 0)).toEqual([20, 0, 0, 255]);
  });

  it("rounds a fractional offset instead of reading an empty source", () => {
    const current = image(4, 1, (x) => [x * 10, 0, 0, 255]);
    const out = cloneWithMask(current, maskAt(4, 1, [[0, 0]]), 1.6, 0.2);
    expect(pixel(out, 0, 0)).toEqual([20, 0, 0, 255]);
  });

  it("leaves pixels whose source falls outside the image unchanged", () => {
    const current = image(3, 1, (x) => [x * 10, 0, 0, 255]);
    const out = cloneWithMask(current, maskAt(3, 1, [[2, 0]]), 5, 0);
    expect(pixel(out, 2, 0)).toEqual([20, 0, 0, 255]);
  });

  it("does not mutate its inputs", () => {
    const current = image(2, 1, (x) => [x * 10, 0, 0, 255]);
    const mask = maskAt(2, 1, [[0, 0]]);
    cloneWithMask(current, mask, 1, 0);
    expect(pixel(current, 0, 0)).toEqual([0, 0, 0, 255]);
  });
});
