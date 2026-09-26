import { describe, expect, it } from "vitest";
import {
  COMPARE_BASE_MAX_ZOOM,
  actualSizeZoom,
  clampCurtain,
  curtainAfterKey,
  curtainAtPointer,
  maxZoomFor,
  pixelPercent,
  zoomAroundCenter,
} from "./compareView";

const BOX = { width: 400, height: 300 };

describe("curtain", () => {
  it("stays between 0 and 100 and centers on nonsense", () => {
    expect(clampCurtain(-5)).toBe(0);
    expect(clampCurtain(140)).toBe(100);
    expect(clampCurtain(Number.NaN)).toBe(50);
  });

  it("follows the pointer across the box width", () => {
    expect(curtainAtPointer(150, 50, 400)).toBe(25);
    expect(curtainAtPointer(900, 50, 400)).toBe(100);
  });

  it("keeps the current position when the box is not measured", () => {
    expect(curtainAtPointer(150, 0, 0, 40)).toBe(40);
  });

  it("moves with the arrow, page, home and end keys", () => {
    expect(curtainAfterKey(50, "ArrowLeft")).toBe(48);
    expect(curtainAfterKey(50, "ArrowRight")).toBe(52);
    expect(curtainAfterKey(50, "ArrowUp")).toBe(52);
    expect(curtainAfterKey(50, "PageDown")).toBe(40);
    expect(curtainAfterKey(99, "PageUp")).toBe(100);
    expect(curtainAfterKey(50, "Home")).toBe(0);
    expect(curtainAfterKey(50, "End")).toBe(100);
  });

  it("ignores keys that do not move the curtain", () => {
    expect(curtainAfterKey(50, "Enter")).toBeNull();
  });
});

describe("zoom", () => {
  it("needs the natural width over the fitted width to show real pixels", () => {
    expect(actualSizeZoom(1600, 400)).toBe(4);
  });

  it("has no real-pixel zoom until both sizes are known", () => {
    expect(actualSizeZoom(0, 400)).toBeNull();
    expect(actualSizeZoom(1600, 0)).toBeNull();
  });

  it("never lets real pixels be smaller than the fitted view", () => {
    expect(actualSizeZoom(300, 400)).toBe(1);
  });

  it("lets a big photo zoom past the default limit up to real pixels", () => {
    expect(maxZoomFor(null)).toBe(COMPARE_BASE_MAX_ZOOM);
    expect(maxZoomFor(3)).toBe(COMPARE_BASE_MAX_ZOOM);
    expect(maxZoomFor(12)).toBe(12);
  });

  it("reports the zoom against real pixels", () => {
    expect(pixelPercent(1, 4)).toBe(25);
    expect(pixelPercent(4, 4)).toBe(100);
  });

  it("zooms around the center and keeps the content glued to the edges", () => {
    const zoomed = zoomAroundCenter({ zoom: 1, panX: 0, panY: 0 }, 2, BOX, 8);
    expect(zoomed).toEqual({ zoom: 2, panX: -200, panY: -150 });
    expect(zoomAroundCenter(zoomed, 1, BOX, 8)).toEqual({ zoom: 1, panX: 0, panY: 0 });
  });

  it("clamps the zoom to the allowed range", () => {
    expect(zoomAroundCenter({ zoom: 1, panX: 0, panY: 0 }, 40, BOX, 8).zoom).toBe(8);
    expect(zoomAroundCenter({ zoom: 2, panX: 0, panY: 0 }, 0.2, BOX, 8).zoom).toBe(1);
  });
});
