import { describe, expect, it } from "vitest";
import type { RestoreGeometry } from "../../../lib/restoreApiTypes";
import {
  DEFAULT_HANDLE_INSET,
  NUDGE_STEP,
  frameToWorking,
  initialHandles,
  isConvexQuad,
  nudgedHandle,
  perspectiveFrame,
  perspectiveFromHandles,
  suggestionOutline,
  workingToFrame,
  type Handles,
} from "./perspectiveMath";

const NEUTRAL: RestoreGeometry = { rotate90: 0, crop: null, angle: 0 };
const FRAME = { width: 400, height: 200 };
const SQUARE: Handles = [
  { x: 0.1, y: 0.2 },
  { x: 0.9, y: 0.2 },
  { x: 0.9, y: 0.8 },
  { x: 0.1, y: 0.8 },
];

describe("working and frame coordinates", () => {
  it("add the crop offset when there is no straighten", () => {
    const geometry: RestoreGeometry = { rotate90: 1, crop: [30, 20, 100, 50], angle: 0 };

    expect(workingToFrame({ x: 5, y: 6 }, geometry, FRAME)).toEqual({ x: 35, y: 26 });
    expect(frameToWorking({ x: 35, y: 26 }, geometry, FRAME)).toEqual({ x: 5, y: 6 });
  });

  it("follow the counterclockwise straighten of the backend", () => {
    const geometry: RestoreGeometry = { rotate90: 0, crop: null, angle: 10 };

    const moved = frameToWorking({ x: 300, y: 100 }, geometry, FRAME);

    expect(moved.x).toBeCloseTo(200 + 100 * Math.cos(Math.PI / 18));
    expect(moved.y).toBeCloseTo(100 - 100 * Math.sin(Math.PI / 18));
  });

  it("round-trip through straighten and crop", () => {
    const geometry: RestoreGeometry = { rotate90: 0, crop: [12, 8, 300, 150], angle: -7.5 };

    const back = workingToFrame(frameToWorking({ x: 77, y: 143 }, geometry, FRAME), geometry, FRAME);

    expect(back.x).toBeCloseTo(77);
    expect(back.y).toBeCloseTo(143);
  });
});

describe("perspectiveFromHandles", () => {
  it("turns handles on an untouched photo into corners in photo pixels", () => {
    expect(perspectiveFromHandles(NEUTRAL, SQUARE, FRAME, FRAME)).toEqual({
      rotate90: 0,
      crop: null,
      angle: 0,
      corners: [
        [40, 40],
        [360, 40],
        [360, 160],
        [40, 160],
      ],
    });
  });

  it("maps handles drawn on a cropped copy back onto the whole photo and drops crop and angle", () => {
    const geometry: RestoreGeometry = { rotate90: 3, crop: [100, 50, 200, 100], angle: 0 };

    const result = perspectiveFromHandles(geometry, SQUARE, { width: 200, height: 100 }, FRAME);

    expect(result).toEqual({
      rotate90: 3,
      crop: null,
      angle: 0,
      corners: [
        [120, 70],
        [280, 70],
        [280, 130],
        [120, 130],
      ],
    });
  });
});

describe("initialHandles", () => {
  it("starts from the corners the analysis found", () => {
    const suggestion: RestoreGeometry = {
      rotate90: 0,
      crop: null,
      angle: 0,
      corners: [
        [40, 20],
        [360, 40],
        [380, 180],
        [20, 160],
      ],
    };

    const handles = initialHandles(NEUTRAL, suggestion, FRAME, FRAME);

    expect(handles[0]).toEqual({ x: 0.1, y: 0.1 });
    expect(handles[2]).toEqual({ x: 0.95, y: 0.9 });
  });

  it("falls back to an inset square without a suggestion or after another rotation", () => {
    const rotated: RestoreGeometry = { rotate90: 1, crop: null, angle: 0, corners: [[0, 0], [1, 0], [1, 1], [0, 1]] };

    expect(initialHandles(NEUTRAL, null, FRAME, FRAME)[0]).toEqual({ x: DEFAULT_HANDLE_INSET, y: DEFAULT_HANDLE_INSET });
    expect(initialHandles(NEUTRAL, rotated, FRAME, FRAME)[0]).toEqual({ x: DEFAULT_HANDLE_INSET, y: DEFAULT_HANDLE_INSET });
  });
});

describe("perspectiveFrame", () => {
  it("prefers the frame size the analysis reported", () => {
    const capture = { autoCrop: null, photos: [], perspective: null, frameWidth: 640, frameHeight: 480 };

    expect(perspectiveFrame(capture, { rotate90: 0, crop: [1, 1, 20, 20], angle: 0 }, FRAME)).toEqual({
      width: 640,
      height: 480,
    });
  });

  it("uses the working size only when nothing was cropped", () => {
    expect(perspectiveFrame(undefined, { rotate90: 0, crop: null, angle: 4 }, FRAME)).toEqual(FRAME);
    expect(perspectiveFrame(undefined, { rotate90: 0, crop: [1, 1, 20, 20], angle: 0 }, FRAME)).toBeNull();
  });
});

describe("handle editing", () => {
  it("tells a convex quad from a crossed one", () => {
    expect(isConvexQuad(SQUARE)).toBe(true);
    expect(isConvexQuad([SQUARE[0], SQUARE[2], SQUARE[1], SQUARE[3]])).toBe(false);
  });

  it("nudges one corner with the arrow keys and keeps it on the photo", () => {
    const nudged = nudgedHandle(SQUARE, 1, "ArrowRight", false);

    expect(nudged?.[1].x).toBeCloseTo(0.9 + NUDGE_STEP);
    expect(nudged?.[0]).toEqual(SQUARE[0]);
    expect(nudgedHandle([{ x: 0, y: 0 }, ...SQUARE.slice(1)] as Handles, 0, "ArrowUp", true)?.[0]).toEqual({ x: 0, y: 0 });
    expect(nudgedHandle(SQUARE, 0, "Enter", false)).toBeNull();
  });
});

describe("suggestionOutline", () => {
  it("draws a straight suggested crop where it sits on the untouched sheet", () => {
    const suggestion: RestoreGeometry = { rotate90: 0, crop: [40, 20, 200, 100], angle: 0 };

    expect(suggestionOutline(suggestion, NEUTRAL, FRAME, FRAME)).toEqual([
      { x: 0.1, y: 0.1 },
      { x: 0.6, y: 0.1 },
      { x: 0.6, y: 0.6 },
      { x: 0.1, y: 0.6 },
    ]);
  });

  it("tilts the outline of a leveled suggestion back onto the sheet", () => {
    const suggestion: RestoreGeometry = { rotate90: 0, crop: [100, 50, 200, 100], angle: 5 };

    const outline = suggestionOutline(suggestion, NEUTRAL, FRAME, FRAME);

    expect(outline).not.toBeNull();
    const [topLeft, topRight] = outline!;
    // El contenido se nivelo girandolo 5 grados en sentido antihorario: en la hoja su borde baja hacia la derecha.
    expect(topRight.y).toBeGreaterThan(topLeft.y);
  });

  it("draws nothing once perspective is set or the photo was turned", () => {
    const suggestion: RestoreGeometry = { rotate90: 0, crop: [40, 20, 200, 100], angle: 0 };

    expect(suggestionOutline(suggestion, { ...NEUTRAL, rotate90: 1 }, FRAME, FRAME)).toBeNull();
    expect(
      suggestionOutline(suggestion, { ...NEUTRAL, corners: [[0, 0], [10, 0], [10, 10], [0, 10]] }, FRAME, FRAME),
    ).toBeNull();
  });
});
