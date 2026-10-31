import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { BrushStroke } from "./maskCanvas";
import { applyRetouch } from "./retouchApply";

const PHOTO = { width: 6000, height: 4000 };
const STROKE: BrushStroke = { mode: "paint", radius: 10, points: [{ x: 100, y: 100 }, { x: 120, y: 110 }] };
// Caja del trazo: los puntos más el radio en cada borde → x 90..130, y 90..120.
const BOX = { x: 90, y: 90, width: 40, height: 30 };

let readBoxes: Array<[number, number, number, number]>;
let writes: Array<[number, number, number, number]>;
let createImageBitmap: ReturnType<typeof vi.fn>;

function fakeContext() {
  return {
    filter: "none",
    globalCompositeOperation: "source-over",
    fillStyle: "",
    lineWidth: 0,
    lineCap: "",
    lineJoin: "",
    strokeStyle: "",
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    drawImage: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    getImageData: vi.fn((x: number, y: number, w: number, h: number) => {
      readBoxes.push([x, y, w, h]);
      return { width: w, height: h, data: new Uint8ClampedArray(w * h * 4) };
    }),
    putImageData: vi.fn((image: { width: number; height: number }, x: number, y: number) => {
      writes.push([image.width, image.height, x, y]);
    }),
  };
}

beforeEach(() => {
  readBoxes = [];
  writes = [];
  createImageBitmap = vi.fn(async () => ({ width: PHOTO.width, height: PHOTO.height }));
  vi.stubGlobal("createImageBitmap", createImageBitmap);
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(() => fakeContext() as never);
  vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation((callback) => callback(new Blob(["png"])));
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

function expectOnlyBoxReads(maxWidth: number, maxHeight: number) {
  expect(readBoxes.length).toBeGreaterThan(0);
  for (const [, , width, height] of readBoxes) {
    expect(width).toBeLessThanOrEqual(maxWidth);
    expect(height).toBeLessThanOrEqual(maxHeight);
  }
}

describe("applyRetouch", () => {
  it("restore reads and writes only the stroke box, never the full photo", async () => {
    await applyRetouch("restore", STROKE, new Blob(["current"]), new Blob(["original"]), { dx: 0, dy: 0 });

    expectOnlyBoxReads(BOX.width, BOX.height);
    // Pesos del trazo, foto actual y foto original: tres lecturas, todas del tamaño de la caja.
    expect(readBoxes).toHaveLength(3);
    expect(writes).toEqual([[BOX.width, BOX.height, BOX.x, BOX.y]]);
  });

  it("clone reads the stroke box and the source box at the offset, clipped to the photo", async () => {
    await applyRetouch("clone", STROKE, new Blob(["current"]), new Blob(["original"]), { dx: 50, dy: 0 });

    expectOnlyBoxReads(BOX.width, BOX.height);
    expect(readBoxes).toHaveLength(3);
    expect(writes).toEqual([[BOX.width, BOX.height, BOX.x, BOX.y]]);
  });

  it("clips the box to the photo edge when the stroke starts outside it", async () => {
    const edge: BrushStroke = { mode: "paint", radius: 10, points: [{ x: 2, y: 2 }, { x: 5, y: 5 }] };

    await applyRetouch("restore", edge, new Blob(["current"]), new Blob(["original"]), { dx: 0, dy: 0 });

    expect(writes).toEqual([[15, 15, 0, 0]]);
    expectOnlyBoxReads(15, 15);
  });

  it("decodes each base version once and reuses the bitmap for later strokes", async () => {
    const current = new Blob(["current"]);
    const original = new Blob(["original"]);

    await applyRetouch("restore", STROKE, current, original, { dx: 0, dy: 0 });
    await applyRetouch("restore", STROKE, current, original, { dx: 0, dy: 0 });

    expect(createImageBitmap).toHaveBeenCalledTimes(2);
  });
});
