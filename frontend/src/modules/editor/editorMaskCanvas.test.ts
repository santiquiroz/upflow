import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  canvasHasPixels,
  exportMaskBlob,
  exportProtectMaskBlob,
  flattenHistory,
  hasProtection,
  maskHasContent,
  previewAdjust,
  renderMaskOverlays,
  replayMasks,
  withoutEdits,
  type HistoryEntry,
} from "./editorMaskCanvas";

function fakeContext(pixels: Uint8ClampedArray = new Uint8ClampedArray(8)) {
  return {
    clearRect: vi.fn(),
    drawImage: vi.fn(),
    getImageData: vi.fn(() => ({ data: pixels.slice() })),
    putImageData: vi.fn(),
    fillRect: vi.fn(),
    globalCompositeOperation: "source-over",
    fillStyle: "",
  };
}

type FakeContext = ReturnType<typeof fakeContext>;
let contexts: WeakMap<HTMLCanvasElement, FakeContext>;

function contextFor(canvas: HTMLCanvasElement): FakeContext {
  let context = contexts.get(canvas);
  if (!context) {
    context = fakeContext();
    contexts.set(canvas, context);
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

beforeEach(() => {
  contexts = new WeakMap();
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function (this: HTMLCanvasElement) {
    return contextFor(this) as unknown as CanvasRenderingContext2D;
  });
});

afterEach(() => vi.restoreAllMocks());

describe("exportMaskBlob", () => {
  it("flattens edit alpha onto black before encoding without promoting antialias pixels to binary white", async () => {
    const { canvas, context } = maskCanvas(new Uint8ClampedArray([255, 255, 255, 1, 255, 255, 255, 128]));
    const png = new Blob(["png"], { type: "image/png" });
    let encodedCanvas: HTMLCanvasElement | undefined;
    const toBlob = vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(function (
      this: HTMLCanvasElement,
      callback: BlobCallback,
    ) {
      encodedCanvas = this;
      callback(png);
    });

    const result = await exportMaskBlob(canvas);

    expect(result).toBe(png);
    expect(toBlob).toHaveBeenCalledWith(expect.any(Function), "image/png");
    expect(encodedCanvas?.width).toBe(2);
    expect(encodedCanvas?.height).toBe(1);
    const exportedContext = contextFor(encodedCanvas!);
    expect(exportedContext.fillStyle).toBe("#000000");
    expect(exportedContext.fillRect).toHaveBeenCalledWith(0, 0, 2, 1);
    expect(exportedContext.drawImage).toHaveBeenCalledWith(canvas, 0, 0);
    expect(exportedContext.fillRect.mock.invocationCallOrder[0]).toBeLessThan(
      exportedContext.drawImage.mock.invocationCallOrder[0],
    );
    expect(exportedContext.drawImage.mock.invocationCallOrder[0]).toBeLessThan(
      toBlob.mock.invocationCallOrder[0],
    );
    expect(exportedContext.putImageData).not.toHaveBeenCalled();
    expect(context.getImageData).not.toHaveBeenCalled();
  });
});

describe("exportProtectMaskBlob", () => {
  it("encodes a base-sized PNG with opaque white protection and opaque black background", async () => {
    const source = new Uint8ClampedArray([30, 50, 70, 0, 0, 0, 0, 1]);
    const { canvas, context } = maskCanvas(source);
    const png = new Blob(["png"], { type: "image/png" });
    let encodedCanvas: HTMLCanvasElement | undefined;
    const toBlob = vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation(function (
      this: HTMLCanvasElement,
      callback: BlobCallback,
    ) {
      encodedCanvas = this;
      callback(png);
    });

    const result = await exportProtectMaskBlob(canvas);

    expect(result).toBe(png);
    expect(toBlob).toHaveBeenCalledWith(expect.any(Function), "image/png");
    expect(encodedCanvas).toBeDefined();
    expect(encodedCanvas?.width).toBe(2);
    expect(encodedCanvas?.height).toBe(1);
    expect(encodedCanvas).not.toBe(canvas);
    expect(context.getImageData).toHaveBeenCalledWith(0, 0, 2, 1);
    expect(contextFor(encodedCanvas!).putImageData).toHaveBeenCalledWith({
      data: new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255]),
    }, 0, 0);
    expect(source).toEqual(new Uint8ClampedArray([30, 50, 70, 0, 0, 0, 0, 1]));
  });

  it("rejects when the browser cannot encode a PNG", async () => {
    vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation((callback) => callback(null));

    await expect(exportProtectMaskBlob(maskCanvas().canvas)).rejects.toThrow("could not export the mask");
  });

  it("rejects when the source has no rendering context", async () => {
    const { canvas } = maskCanvas();
    vi.spyOn(canvas, "getContext").mockReturnValue(null);

    await expect(exportProtectMaskBlob(canvas)).rejects.toThrow("canvas 2d context unavailable");
  });
});

describe("renderMaskOverlays", () => {
  it("subtracts protected pixels from red edit overlay and renders green protection independently", () => {
    const editPixels = new Uint8ClampedArray([255, 255, 255, 200, 255, 255, 255, 150]);
    const protectionPixels = new Uint8ClampedArray([255, 255, 255, 1, 0, 0, 0, 0]);
    const edit = maskCanvas(editPixels);
    const protect = maskCanvas(protectionPixels);
    const red = maskCanvas();
    const green = maskCanvas();
    const redTints: string[] = [];
    const greenTints: string[] = [];
    red.context.fillRect.mockImplementation(() => {
      expect(red.context.globalCompositeOperation).toBe("source-in");
      redTints.push(red.context.fillStyle);
    });
    green.context.fillRect.mockImplementation(() => {
      expect(green.context.globalCompositeOperation).toBe("source-in");
      greenTints.push(green.context.fillStyle);
    });

    renderMaskOverlays(red.canvas, green.canvas, edit.canvas, protect.canvas);

    expect(red.context.putImageData).toHaveBeenCalledWith({
      data: new Uint8ClampedArray([255, 255, 255, 0, 255, 255, 255, 150]),
    }, 0, 0);
    expect(redTints).toEqual(["#ef4444"]);
    expect(greenTints).toEqual(["#22c55e"]);
    expect(green.context.clearRect).toHaveBeenCalledWith(0, 0, 2, 1);
    expect(green.context.drawImage).toHaveBeenCalledWith(protect.canvas, 0, 0);
    expect(green.context.putImageData).not.toHaveBeenCalled();
    expect(red.context.globalCompositeOperation).toBe("source-over");
    expect(green.context.globalCompositeOperation).toBe("source-over");
    expect(editPixels[3]).toBe(200);
    expect(protectionPixels[3]).toBe(1);
  });

  it("previews the edge adjustment on the red overlay and leaves the protection overlay alone", () => {
    const edit = maskCanvas(new Uint8ClampedArray([255, 255, 255, 255, 255, 255, 255, 0]));
    const protect = maskCanvas();
    const red = maskCanvas();
    const green = maskCanvas();

    renderMaskOverlays(red.canvas, green.canvas, edit.canvas, protect.canvas, { growPx: 12, featherPx: 0 });

    expect(red.context.clearRect).toHaveBeenCalledWith(0, 0, 2, 1);
    expect(green.context.clearRect).toHaveBeenCalledTimes(1);
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

describe("replayMasks", () => {
  it("routes segments and protection separately, then restores protection when its clear is undone", () => {
    const edit = maskCanvas();
    const protect = maskCanvas();
    const editSegment = maskCanvas().canvas;
    const protectedSegment = maskCanvas().canvas;
    const history: HistoryEntry[] = [
      { kind: "segment", mask: editSegment },
      { kind: "protect", mask: protectedSegment },
      { kind: "clearProtection" },
    ];

    replayMasks(edit.canvas, protect.canvas, history, null);

    expect(edit.context.drawImage).toHaveBeenCalledTimes(1);
    expect(edit.context.drawImage).toHaveBeenCalledWith(editSegment, 0, 0);
    expect(protect.context.drawImage).toHaveBeenCalledTimes(1);
    expect(protect.context.drawImage).toHaveBeenCalledWith(protectedSegment, 0, 0);
    expect(edit.context.clearRect).toHaveBeenCalledTimes(1);
    expect(protect.context.clearRect).toHaveBeenCalledTimes(2);
    expect(protect.context.drawImage.mock.invocationCallOrder[0]).toBeLessThan(
      protect.context.clearRect.mock.invocationCallOrder[1],
    );
    expect(hasProtection(history)).toBe(false);
    vi.clearAllMocks();

    replayMasks(edit.canvas, protect.canvas, history.slice(0, -1), null);

    expect(protect.context.clearRect).toHaveBeenCalledTimes(1);
    expect(protect.context.clearRect).toHaveBeenCalledWith(0, 0, 2, 1);
    expect(protect.context.drawImage).toHaveBeenCalledTimes(2);
    expect(protect.context.drawImage).toHaveBeenNthCalledWith(1, protectedSegment, 0, 0);
    expect(protect.context.drawImage).toHaveBeenNthCalledWith(2, edit.canvas, 0, 0);
    expect(protect.context.clearRect.mock.invocationCallOrder[0]).toBeLessThan(
      protect.context.drawImage.mock.invocationCallOrder[0],
    );
    expect(hasProtection(history.slice(0, -1))).toBe(true);
    vi.clearAllMocks();

    replayMasks(edit.canvas, protect.canvas, history.slice(0, -2), null);

    expect(protect.context.drawImage).not.toHaveBeenCalled();
    expect(hasProtection(history.slice(0, -2))).toBe(false);
  });

  it("lets the erase win over a protection whose own tap falls outside the erase area", () => {
    const edit = maskCanvas(new Uint8ClampedArray([255, 255, 255, 0, 255, 255, 255, 0]));
    const protect = maskCanvas();
    const greedyProtection = maskCanvas().canvas;
    const operations: string[] = [];
    protect.context.drawImage.mockImplementation(() => operations.push(protect.context.globalCompositeOperation));

    replayMasks(edit.canvas, protect.canvas, [
      { kind: "segment", mask: maskCanvas().canvas, point: { x: 1, y: 0 } },
      { kind: "protect", mask: greedyProtection, point: { x: 0, y: 0 } },
    ], null);

    expect(edit.context.drawImage.mock.calls.some(([source]) => source === greedyProtection)).toBe(false);
    expect(protect.context.drawImage.mock.lastCall?.[0]).toBe(edit.canvas);
    expect(operations).toEqual(["source-over", "destination-out"]);
    expect(protect.context.globalCompositeOperation).toBe("source-over");
  });

  it("gives a subject back when the erase area covers that subject's own protection tap", () => {
    const edit = maskCanvas(new Uint8ClampedArray([255, 255, 255, 255, 255, 255, 255, 255]));
    const protect = maskCanvas();
    const subject = maskCanvas().canvas;
    const operations: string[] = [];
    edit.context.drawImage.mockImplementation(() => operations.push(edit.context.globalCompositeOperation));

    replayMasks(edit.canvas, protect.canvas, [
      { kind: "segment", mask: maskCanvas().canvas, point: { x: 1, y: 0 } },
      { kind: "protect", mask: subject, point: { x: 0.6, y: 0.2 } },
    ], null);

    expect(edit.context.getImageData).toHaveBeenCalledWith(0, 0, 1, 1);
    expect(edit.context.drawImage.mock.lastCall?.[0]).toBe(subject);
    expect(operations).toEqual(["source-over", "destination-out"]);
    expect(edit.context.globalCompositeOperation).toBe("source-over");
  });
});

describe("batches", () => {
  it("replays a batch's segments and protects in order", () => {
    const edit = maskCanvas();
    const protect = maskCanvas();
    const first = maskCanvas().canvas;
    const second = maskCanvas().canvas;
    const kept = maskCanvas().canvas;

    replayMasks(edit.canvas, protect.canvas, [
      { kind: "batch", personIds: [1, 2], entries: [
        { kind: "segment", mask: first, point: { x: 0, y: 0 } },
        { kind: "protect", mask: kept, point: { x: 1, y: 0 } },
        { kind: "segment", mask: second, point: { x: 1, y: 0 } },
      ] },
    ], null);

    expect(edit.context.drawImage).toHaveBeenNthCalledWith(1, first, 0, 0);
    expect(edit.context.drawImage).toHaveBeenNthCalledWith(2, second, 0, 0);
    expect(protect.context.drawImage).toHaveBeenNthCalledWith(1, kept, 0, 0);
    expect(protect.context.drawImage).toHaveBeenLastCalledWith(edit.canvas, 0, 0);
  });

  it("sees content and protection inside batches", () => {
    const batch: HistoryEntry = {
      kind: "batch",
      personIds: [3],
      entries: [{ kind: "segment" }, { kind: "protect" }],
    };

    expect(maskHasContent([batch])).toBe(true);
    expect(hasProtection([batch])).toBe(true);
    expect(maskHasContent([{ kind: "batch", personIds: [], entries: [{ kind: "protect" }] }])).toBe(false);
  });

  it("lets a clear after a batch clear the batch's protects", () => {
    const edit = maskCanvas();
    const protect = maskCanvas();
    const history: HistoryEntry[] = [
      { kind: "batch", personIds: [1], entries: [
        { kind: "segment", mask: maskCanvas().canvas },
        { kind: "protect", mask: maskCanvas().canvas },
      ] },
      { kind: "clearProtection" },
    ];

    replayMasks(edit.canvas, protect.canvas, history, null);

    expect(hasProtection(history)).toBe(false);
    expect(protect.context.drawImage).toHaveBeenCalledTimes(1);
    expect(protect.context.clearRect).toHaveBeenCalledTimes(2);
  });

  it("flattens nested entries in order", () => {
    const stroke: HistoryEntry = { kind: "segment" };
    const kept: HistoryEntry = { kind: "protect" };
    expect(flattenHistory([stroke, { kind: "batch", personIds: [], entries: [kept] }])).toEqual([stroke, kept]);
  });

  it("drops a batch's segments but keeps its protects when the erase is cleared", () => {
    const kept: HistoryEntry = { kind: "protect" };
    expect(withoutEdits([
      { kind: "segment" },
      { kind: "batch", personIds: [1], entries: [{ kind: "segment" }, kept] },
    ])).toEqual([kept]);
  });
});

describe("hasProtection", () => {
  it("uses only protection entries after the last clear", () => {
    expect(hasProtection([])).toBe(false);
    expect(hasProtection([{ kind: "segment" }])).toBe(false);
    expect(hasProtection([{ kind: "protect" }, { kind: "clearProtection" }])).toBe(false);
    expect(hasProtection([
      { kind: "protect" }, { kind: "clearProtection" }, { kind: "protect" },
    ])).toBe(true);
  });
});

describe("canvasHasPixels", () => {
  it("detects protection from canvas alpha instead of its RGB color", () => {
    expect(canvasHasPixels(maskCanvas(new Uint8ClampedArray([255, 255, 255, 0])).canvas)).toBe(false);
    expect(canvasHasPixels(maskCanvas(new Uint8ClampedArray([0, 0, 0, 1])).canvas)).toBe(true);
  });

  it("treats an unavailable canvas context as empty", () => {
    const { canvas } = maskCanvas();
    vi.spyOn(canvas, "getContext").mockReturnValue(null);

    expect(canvasHasPixels(canvas)).toBe(false);
  });
});
