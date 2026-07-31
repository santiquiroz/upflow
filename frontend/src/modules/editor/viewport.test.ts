import { describe, expect, it } from "vitest";
import {
  IDENTITY_VIEWPORT,
  MAX_ZOOM,
  MIN_ZOOM,
  ZOOM_STEP,
  clampPan,
  clampZoom,
  isZoomed,
  transformStyle,
  zoomAtPoint,
  zoomByStep,
} from "./viewport";

const CONTAINER = { width: 800, height: 600 };

describe("clampZoom", () => {
  it("keeps zoom inside the allowed range", () => {
    expect(clampZoom(0.1)).toBe(MIN_ZOOM);
    expect(clampZoom(99)).toBe(MAX_ZOOM);
    expect(clampZoom(2.5)).toBe(2.5);
  });

  it("survives NaN from a broken wheel delta", () => {
    expect(clampZoom(Number.NaN)).toBe(MIN_ZOOM);
  });
});

describe("zoomAtPoint", () => {
  it("keeps the content point under the cursor fixed", () => {
    const focus = { x: 200, y: 150 };
    const zoomed = zoomAtPoint(IDENTITY_VIEWPORT, 2, focus, CONTAINER);
    // contenido bajo el cursor antes: (200-0)/1 = 200; después debe seguir ahí
    const contentAfter = (focus.x - zoomed.panX) / zoomed.zoom;
    expect(contentAfter).toBeCloseTo(200, 5);
    expect((focus.y - zoomed.panY) / zoomed.zoom).toBeCloseTo(150, 5);
  });

  it("keeps the anchor fixed when zooming out from a panned state", () => {
    const start = zoomAtPoint(IDENTITY_VIEWPORT, 4, { x: 600, y: 400 }, CONTAINER);
    const focus = { x: 300, y: 200 };
    const contentBefore = (focus.x - start.panX) / start.zoom;

    const out = zoomAtPoint(start, 2, focus, CONTAINER);

    expect((focus.x - out.panX) / out.zoom).toBeCloseTo(contentBefore, 5);
  });

  it("recenters when zooming back to 1", () => {
    const zoomed = zoomAtPoint(IDENTITY_VIEWPORT, 4, { x: 700, y: 500 }, CONTAINER);
    const back = zoomAtPoint(zoomed, 1, { x: 100, y: 100 }, CONTAINER);
    expect(back).toEqual(IDENTITY_VIEWPORT);
  });
});

describe("clampPan", () => {
  it("centers the content when it is smaller than the container", () => {
    expect(clampPan({ zoom: 1, panX: 250, panY: -80 }, CONTAINER)).toEqual(IDENTITY_VIEWPORT);
  });

  it("never lets an edge drift inside the container", () => {
    const dragged = clampPan({ zoom: 2, panX: 400, panY: 400 }, CONTAINER);
    expect(dragged.panX).toBe(0);
    expect(dragged.panY).toBe(0);

    const overshot = clampPan({ zoom: 2, panX: -5000, panY: -5000 }, CONTAINER);
    expect(overshot.panX).toBe(-800);
    expect(overshot.panY).toBe(-600);
  });

  it("leaves a legal pan untouched", () => {
    const legal = { zoom: 2, panX: -100, panY: -50 };
    expect(clampPan(legal, CONTAINER)).toEqual(legal);
  });

  it("fails open on a zero-sized container (jsdom / hidden tab)", () => {
    const viewport = { zoom: 2, panX: -10, panY: -10 };
    expect(clampPan(viewport, { width: 0, height: 0 })).toEqual(viewport);
  });
});

describe("zoomByStep", () => {
  it("zooms around the container center", () => {
    const zoomed = zoomByStep(IDENTITY_VIEWPORT, 1, CONTAINER);
    expect(zoomed.zoom).toBe(ZOOM_STEP);
    expect((400 - zoomed.panX) / zoomed.zoom).toBeCloseTo(400, 5);
  });

  it("stops at the bounds instead of drifting", () => {
    let viewport = IDENTITY_VIEWPORT;
    for (let i = 0; i < 20; i += 1) {
      viewport = zoomByStep(viewport, 1, CONTAINER);
    }
    expect(viewport.zoom).toBe(MAX_ZOOM);

    for (let i = 0; i < 40; i += 1) {
      viewport = zoomByStep(viewport, -1, CONTAINER);
    }
    expect(viewport).toEqual(IDENTITY_VIEWPORT);
  });
});

describe("transformStyle / isZoomed", () => {
  it("emits a compositor-friendly transform", () => {
    expect(transformStyle({ zoom: 2, panX: -30, panY: 12 })).toBe("translate(-30px, 12px) scale(2)");
  });

  it("knows when the view is untouched", () => {
    expect(isZoomed(IDENTITY_VIEWPORT)).toBe(false);
    expect(isZoomed({ zoom: 1.5, panX: 0, panY: 0 })).toBe(true);
    expect(isZoomed({ zoom: 1, panX: -10, panY: 0 })).toBe(true);
  });
});
