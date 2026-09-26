import { describe, expect, it } from "vitest";
import {
  boxAfterKey,
  boxFromCorners,
  boxPercentStyle,
  notTextIndices,
  pointOnFrame,
  suggestedOsdBoxes,
  withBoxAdded,
  withBoxRemoved,
  withBoxReplaced,
} from "./cctvBoxes";

const LITE = { width: 960, height: 1080 };

describe("suggestedOsdBoxes", () => {
  it("proposes the Hikvision clock at the top left and the camera name at the bottom right", () => {
    const [clock, camera] = suggestedOsdBoxes(LITE);

    expect(clock).toEqual([19, 22, 384, 64]);
    expect(camera).toEqual([960 - 19 - 240, 1080 - 22 - 64, 240, 64]);
  });

  it("keeps every suggested box inside the frame with even sides", () => {
    for (const [x, y, w, h] of suggestedOsdBoxes({ width: 353, height: 289 })) {
      expect(w % 2).toBe(0);
      expect(h % 2).toBe(0);
      expect(x + w).toBeLessThanOrEqual(353);
      expect(y + h).toBeLessThanOrEqual(289);
    }
  });
});

describe("pointOnFrame", () => {
  it("maps a point on the stretched display back to stored pixels", () => {
    // Un lite 960x1080 se muestra a 1920x1080: la mitad del ancho mostrado es el ancho guardado.
    const rect = { left: 100, top: 50, width: 1920, height: 1080 };

    expect(pointOnFrame(100 + 960, 50 + 540, rect, LITE)).toEqual({ x: 480, y: 540 });
  });

  it("clamps points dragged outside the frame", () => {
    const rect = { left: 0, top: 0, width: 480, height: 540 };

    expect(pointOnFrame(-40, 9000, rect, LITE)).toEqual({ x: 0, y: 1080 });
  });
});

describe("boxFromCorners", () => {
  it("orders the corners and rounds the sides down to even pixels", () => {
    expect(boxFromCorners({ x: 105.6, y: 40.2 }, { x: 10.2, y: 5.9 }, LITE)).toEqual([10, 5, 96, 36]);
  });

  it("ignores a click that doesn't draw a box", () => {
    expect(boxFromCorners({ x: 10, y: 10 }, { x: 11, y: 30 }, LITE)).toBeNull();
  });

  it("never reaches past the frame edge", () => {
    const box = boxFromCorners({ x: 900.5, y: 1000 }, { x: 960, y: 1080 }, LITE);

    expect(box).toEqual([900, 1000, 60, 80]);
  });
});

describe("keyboard editing", () => {
  it("moves the box two pixels per arrow press inside the frame", () => {
    expect(boxAfterKey([0, 0, 10, 10], "ArrowRight", false, LITE)).toEqual([2, 0, 10, 10]);
    expect(boxAfterKey([0, 0, 10, 10], "ArrowLeft", false, LITE)).toEqual([0, 0, 10, 10]);
  });

  it("resizes with Shift and keeps the sides even and at least two pixels", () => {
    expect(boxAfterKey([0, 0, 10, 10], "ArrowDown", true, LITE)).toEqual([0, 0, 10, 12]);
    expect(boxAfterKey([0, 0, 2, 2], "ArrowLeft", true, LITE)).toEqual([0, 0, 2, 2]);
    expect(boxAfterKey([950, 0, 10, 10], "ArrowRight", true, LITE)).toEqual([950, 0, 10, 10]);
  });

  it("ignores keys that aren't arrows", () => {
    expect(boxAfterKey([0, 0, 10, 10], "Enter", false, LITE)).toBeNull();
  });
});

describe("box lists", () => {
  it("adds boxes up to the limit", () => {
    expect(withBoxAdded([[0, 0, 2, 2]], [4, 4, 2, 2], 2)).toEqual([[0, 0, 2, 2], [4, 4, 2, 2]]);
    expect(withBoxAdded([[0, 0, 2, 2], [4, 4, 2, 2]], [8, 8, 2, 2], 2)).toHaveLength(2);
  });

  it("replaces the only box of a region of interest", () => {
    expect(withBoxAdded([[0, 0, 2, 2]], [4, 4, 2, 2], 1)).toEqual([[4, 4, 2, 2]]);
  });

  it("replaces and removes boxes without touching the original list", () => {
    const boxes = [[0, 0, 2, 2], [4, 4, 2, 2]] as const;

    expect(withBoxReplaced(boxes, 1, [6, 6, 2, 2])).toEqual([[0, 0, 2, 2], [6, 6, 2, 2]]);
    expect(withBoxRemoved(boxes, 0)).toEqual([[4, 4, 2, 2]]);
    expect(boxes).toHaveLength(2);
  });
});

describe("notTextIndices", () => {
  it("flags only the boxes the check found not to look like text", () => {
    const boxes = [[0, 0, 10, 10], [20, 20, 10, 10], [40, 40, 10, 10]] as const;
    const checks = [
      { box: [0, 0, 10, 10], looksLikeText: true },
      { box: [20, 20, 10, 10], looksLikeText: false },
      { box: [40, 40, 12, 10], looksLikeText: false },
    ];

    expect(notTextIndices(boxes, checks)).toEqual([1]);
  });
});

describe("boxPercentStyle", () => {
  it("places the box over the displayed frame whatever its size", () => {
    expect(boxPercentStyle([96, 108, 480, 54], LITE)).toEqual({
      left: "10%",
      top: "10%",
      width: "50%",
      height: "5%",
    });
  });
});
