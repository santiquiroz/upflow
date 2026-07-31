import { describe, expect, it, vi } from "vitest";
import {
  appendPoint,
  drawStrokes,
  fitGenerationSize,
  hasEditableArea,
  startStroke,
  toImagePoint,
  undoLastStroke,
  type BrushStroke,
} from "./maskCanvas";

describe("fitGenerationSize", () => {
  it("snaps to multiples of 64 preserving aspect", () => {
    expect(fitGenerationSize(1920, 1080)).toEqual({ width: 1024, height: 576 });
  });

  it("keeps small images near their size", () => {
    expect(fitGenerationSize(500, 500)).toEqual({ width: 512, height: 512 });
  });

  it("never goes below 64 nor above 1024", () => {
    expect(fitGenerationSize(20, 4000)).toEqual({ width: 64, height: 1024 });
  });
});

describe("toImagePoint", () => {
  it("maps element coords to image coords proportionally", () => {
    const rect = { left: 10, top: 20, width: 200, height: 100 };
    expect(toImagePoint(110, 70, rect, 800, 400)).toEqual({ x: 400, y: 200 });
  });

  it("clamps outside clicks and survives a zero-size rect (jsdom)", () => {
    const rect = { left: 0, top: 0, width: 0, height: 0 };
    expect(toImagePoint(50, 50, rect, 800, 400)).toEqual({ x: 0, y: 0 });
    expect(toImagePoint(999, -5, { left: 0, top: 0, width: 100, height: 100 }, 80, 40)).toEqual({
      x: 80,
      y: 0,
    });
  });
});

describe("strokes", () => {
  it("appendPoint ignores points closer than the minimum distance", () => {
    let stroke = startStroke("paint", 10, { x: 0, y: 0 });
    stroke = appendPoint(stroke, { x: 0.5, y: 0.5 });
    stroke = appendPoint(stroke, { x: 30, y: 0 });
    expect(stroke.points).toHaveLength(2);
  });

  it("hasEditableArea requires at least one paint stroke", () => {
    const erase: BrushStroke = { mode: "erase", radius: 5, points: [{ x: 1, y: 1 }] };
    const paint: BrushStroke = { mode: "paint", radius: 5, points: [{ x: 1, y: 1 }] };
    expect(hasEditableArea([erase])).toBe(false);
    expect(hasEditableArea([erase, paint])).toBe(true);
  });

  it("undoLastStroke drops only the last stroke", () => {
    const first = startStroke("paint", 5, { x: 0, y: 0 });
    const second = startStroke("erase", 5, { x: 9, y: 9 });
    expect(undoLastStroke([first, second])).toEqual([first]);
  });
});

describe("drawStrokes", () => {
  function fakeContext() {
    return {
      lineWidth: 0,
      lineCap: "",
      lineJoin: "",
      strokeStyle: "",
      fillStyle: "",
      globalCompositeOperation: "source-over",
      operations: [] as string[],
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
    };
  }

  it("paints multi-point strokes as round lines of double the radius", () => {
    const ctx = fakeContext();
    drawStrokes(ctx, [{ mode: "paint", radius: 12, points: [{ x: 0, y: 0 }, { x: 5, y: 5 }] }], "#fff");
    expect(ctx.lineWidth).toBe(24);
    expect(ctx.lineCap).toBe("round");
    expect(ctx.stroke).toHaveBeenCalledOnce();
  });

  it("renders single taps as filled circles", () => {
    const ctx = fakeContext();
    drawStrokes(ctx, [{ mode: "paint", radius: 8, points: [{ x: 3, y: 4 }] }], "#fff");
    expect(ctx.arc).toHaveBeenCalledWith(3, 4, 8, 0, Math.PI * 2);
    expect(ctx.fill).toHaveBeenCalledOnce();
  });

  it("erase strokes cut through with destination-out and it resets at the end", () => {
    const ctx = fakeContext();
    const seen: string[] = [];
    ctx.stroke = vi.fn(() => seen.push(ctx.globalCompositeOperation));
    drawStrokes(ctx, [{ mode: "erase", radius: 4, points: [{ x: 0, y: 0 }, { x: 9, y: 9 }] }], "#fff");
    expect(seen).toEqual(["destination-out"]);
    expect(ctx.globalCompositeOperation).toBe("source-over");
  });
});
