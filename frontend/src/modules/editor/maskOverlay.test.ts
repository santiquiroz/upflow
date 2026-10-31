import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { OverlayLayers } from "./maskOverlay";
import { displaySize, overlayLayers, paintOverlay, previewAdjust } from "./maskOverlay";
import { startStroke } from "./maskCanvas";

function fakeContext(pixels: Uint8ClampedArray = new Uint8ClampedArray(8)) {
  return {
    clearRect: vi.fn(),
    drawImage: vi.fn(),
    getImageData: vi.fn(() => ({ data: pixels.slice() })),
    putImageData: vi.fn(),
    fillRect: vi.fn(),
    setTransform: vi.fn(),
    beginPath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    stroke: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    lineWidth: 0,
    lineCap: "",
    lineJoin: "",
    strokeStyle: "",
    fillStyle: "",
    globalCompositeOperation: "source-over",
    filter: "none",
  };
}

type FakeContext = ReturnType<typeof fakeContext>;
let contexts: WeakMap<HTMLCanvasElement, FakeContext>;
let created: FakeContext[];
let newCanvasPixels: Uint8ClampedArray | undefined;

function contextFor(canvas: HTMLCanvasElement): FakeContext {
  let context = contexts.get(canvas);
  if (!context) {
    context = fakeContext(newCanvasPixels);
    contexts.set(canvas, context);
    created.push(context);
  }
  return context;
}

function maskCanvas(pixels?: Uint8ClampedArray) {
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 1;
  const context = fakeContext(pixels);
  contexts.set(canvas, context);
  return { canvas, context };
}

function overlayCanvas(width: number, height: number) {
  const overlay = maskCanvas();
  overlay.canvas.width = width;
  overlay.canvas.height = height;
  return overlay;
}

function layersOf(width: number, height: number): OverlayLayers {
  return { width, height, edit: maskCanvas().canvas, protect: maskCanvas().canvas };
}

function workCanvas(overlay: { context: FakeContext }): HTMLCanvasElement {
  return overlay.context.drawImage.mock.lastCall?.[0] as HTMLCanvasElement;
}

function thresholdedAlpha(overlay: { context: FakeContext }): number[] {
  const data = contextFor(workCanvas(overlay)).putImageData.mock.calls[0][0].data as Uint8ClampedArray;
  return [data[3], data[7]];
}

function blurFilters(): string[] {
  return created.map((context) => context.filter).filter((filter) => filter !== "none");
}

beforeEach(() => {
  contexts = new WeakMap();
  created = [];
  newCanvasPixels = undefined;
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (this: HTMLCanvasElement) {
    return contextFor(this) as unknown as CanvasRenderingContext2D;
  });
});

afterEach(() => vi.restoreAllMocks());

describe("previewAdjust guards", () => {
  it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])("does not touch the canvas when the scale is %s", (scale) => {
    const overlay = overlayCanvas(2, 1);

    expect(() => previewAdjust(overlay.canvas, 12, 4, scale)).not.toThrow();

    expect(overlay.context.clearRect).not.toHaveBeenCalled();
    expect(overlay.context.drawImage).not.toHaveBeenCalled();
    expect(created).toHaveLength(0);
  });

  it("does not touch the canvas when the work canvas would be 0x0", () => {
    const overlay = overlayCanvas(2, 1);

    previewAdjust(overlay.canvas, 12, 0, 0.1);

    expect(overlay.context.clearRect).not.toHaveBeenCalled();
    expect(overlay.context.drawImage).not.toHaveBeenCalled();
  });

  it("skips the preview when the context has no filter support", () => {
    const overlay = overlayCanvas(2, 1);
    delete (overlay.context as Partial<FakeContext>).filter;

    previewAdjust(overlay.canvas, 12, 4, 1);

    expect(overlay.context.clearRect).not.toHaveBeenCalled();
    expect(overlay.context.drawImage).not.toHaveBeenCalled();
  });

  it("never grows the work canvas beyond the photo size at a large zoom", () => {
    const overlay = overlayCanvas(4, 2);

    previewAdjust(overlay.canvas, 2, 0, 8);

    expect([workCanvas(overlay).width, workCanvas(overlay).height]).toEqual([4, 2]);
    expect(blurFilters()).toEqual(["blur(2px)"]);
  });
});

describe("previewAdjust thresholds", () => {
  beforeEach(() => {
    newCanvasPixels = new Uint8ClampedArray([0, 0, 0, 210, 0, 0, 0, 100]);
  });

  it("grows by keeping every pixel the blur reached above the low alpha", () => {
    const overlay = overlayCanvas(2, 1);

    previewAdjust(overlay.canvas, 12, 0, 1);

    expect(thresholdedAlpha(overlay)).toEqual([255, 255]);
    expect(blurFilters()).toEqual(["blur(12px)"]);
  });

  it("shrinks by keeping only the pixels that stay above the high alpha after the blur", () => {
    const overlay = overlayCanvas(2, 1);

    previewAdjust(overlay.canvas, -12, 0, 1);

    expect(thresholdedAlpha(overlay)).toEqual([255, 0]);
    expect(blurFilters()).toEqual(["blur(12px)"]);
  });

  it("scales the work canvas and the blur radius to display pixels", () => {
    const overlay = overlayCanvas(4, 2);

    previewAdjust(overlay.canvas, 12, 4, 0.5);

    expect([workCanvas(overlay).width, workCanvas(overlay).height]).toEqual([2, 1]);
    expect(overlay.context.drawImage).toHaveBeenLastCalledWith(expect.any(HTMLCanvasElement), 0, 0, 4, 2);
    expect(blurFilters()).toEqual(["blur(6px)", "blur(2px)"]);
  });
});

describe("previewAdjust", () => {
  it("leaves the overlay untouched when both values are 0", () => {
    const overlay = maskCanvas(new Uint8ClampedArray([239, 68, 68, 255, 239, 68, 68, 0]));

    previewAdjust(overlay.canvas, 0, 0, 1);

    expect(overlay.context.clearRect).not.toHaveBeenCalled();
    expect(overlay.context.drawImage).not.toHaveBeenCalled();
    expect(overlay.context.putImageData).not.toHaveBeenCalled();
  });

  it("redraws the overlay with the approximate grow when a value is set", () => {
    const overlay = maskCanvas(new Uint8ClampedArray([239, 68, 68, 255, 239, 68, 68, 0]));

    previewAdjust(overlay.canvas, 12, 0, 1);

    expect(overlay.context.clearRect).toHaveBeenCalledWith(0, 0, 2, 1);
    expect(overlay.context.drawImage).toHaveBeenCalledWith(expect.any(HTMLCanvasElement), 0, 0, 2, 1);
  });
});

describe("paintOverlay", () => {
  it("paints the edit layer minus its protection tinted red, and the protection tinted green", () => {
    const red = maskCanvas();
    const green = maskCanvas();
    const layers = layersOf(2, 1);
    const redDrawn: unknown[][] = [];
    const greenDrawn: unknown[][] = [];
    const redTints: string[][] = [];
    const greenTints: string[][] = [];
    red.context.drawImage.mockImplementation((source: unknown) => redDrawn.push([red.context.globalCompositeOperation, source]));
    green.context.drawImage.mockImplementation((source: unknown) => greenDrawn.push([green.context.globalCompositeOperation, source]));
    red.context.fillRect.mockImplementation(() => redTints.push([red.context.globalCompositeOperation, red.context.fillStyle]));
    green.context.fillRect.mockImplementation(() => greenTints.push([green.context.globalCompositeOperation, green.context.fillStyle]));

    paintOverlay(red.canvas, green.canvas, layers, null, 2, null);

    expect(redDrawn).toEqual([["source-over", layers.edit], ["destination-out", layers.protect]]);
    expect(redTints).toEqual([["source-in", "#ef4444"]]);
    expect(greenDrawn).toEqual([["source-over", layers.protect]]);
    expect(greenTints).toEqual([["source-in", "#22c55e"]]);
    expect(red.context.clearRect).toHaveBeenCalledWith(0, 0, 2, 1);
    expect(green.context.clearRect).toHaveBeenCalledWith(0, 0, 2, 1);
    expect(red.context.globalCompositeOperation).toBe("source-over");
    expect(green.context.globalCompositeOperation).toBe("source-over");
  });

  it("previews the edge adjustment on the red overlay and leaves the protection overlay alone", () => {
    const red = maskCanvas();
    const green = maskCanvas();

    paintOverlay(red.canvas, green.canvas, layersOf(2, 1), null, 2, { growPx: 12, featherPx: 0 });

    expect(red.context.clearRect).toHaveBeenCalledWith(0, 0, 2, 1);
    expect(red.context.drawImage).toHaveBeenCalledWith(expect.any(HTMLCanvasElement), 0, 0, 2, 1);
    expect(green.context.clearRect).toHaveBeenCalledTimes(1);
  });

  it("scales the edge values from photo pixels to display pixels", () => {
    const red = maskCanvas();

    paintOverlay(red.canvas, maskCanvas().canvas, layersOf(2, 1), null, 4, { growPx: 12, featherPx: 4 });

    expect(blurFilters()).toEqual(["blur(6px)", "blur(2px)"]);
  });

  it("skips the edge preview while the overlay has no size", () => {
    const red = maskCanvas();

    paintOverlay(red.canvas, maskCanvas().canvas, layersOf(0, 0), null, 640, { growPx: 12, featherPx: 3 });

    expect(blurFilters()).toEqual([]);
  });

  it("draws the live stroke at display scale on top of the committed edit layer", () => {
    const red = maskCanvas();
    const live = startStroke("paint", 4, { x: 1, y: 1 });

    paintOverlay(red.canvas, maskCanvas().canvas, layersOf(2, 1), live, 4, null);

    expect(red.context.setTransform).toHaveBeenNthCalledWith(1, 0.5, 0, 0, 0.5, 0, 0);
    expect(red.context.setTransform).toHaveBeenNthCalledWith(2, 1, 0, 0, 1, 0, 0);
  });
});

describe("overlayLayers", () => {
  it("copies the edit mask at display size and thresholds the protection to opaque", () => {
    newCanvasPixels = new Uint8ClampedArray([0, 0, 0, 1, 0, 0, 0, 0]);
    const masks = { edit: maskCanvas().canvas, protect: maskCanvas().canvas };

    const layers = overlayLayers(masks, 2, 1);

    expect(layers.width).toBe(2);
    expect(layers.height).toBe(1);
    expect(contextFor(layers.edit).drawImage).toHaveBeenCalledWith(masks.edit, 0, 0, 2, 1);
    expect(contextFor(layers.protect).putImageData.mock.calls[0][0].data).toEqual(
      new Uint8ClampedArray([239, 68, 68, 255, 239, 68, 68, 0]),
    );
  });
});

describe("displaySize", () => {
  const originalRatio = window.devicePixelRatio;

  afterEach(() => {
    Object.defineProperty(window, "devicePixelRatio", { value: originalRatio, configurable: true });
  });

  it("sizes the overlay to the stage in device pixels and never above the photo", () => {
    Object.defineProperty(window, "devicePixelRatio", { value: 2, configurable: true });
    const stage = document.createElement("div");
    Object.defineProperty(stage, "clientWidth", { value: 16 });

    expect(displaySize(stage, 64, 40)).toEqual({ width: 32, height: 20 });
    expect(displaySize(stage, 24, 16)).toEqual({ width: 24, height: 16 });
  });

  it("falls back to the photo size before the stage has layout", () => {
    expect(displaySize(document.createElement("div"), 64, 40)).toEqual({ width: 64, height: 40 });
  });
});
