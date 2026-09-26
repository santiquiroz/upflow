import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CompareSlider } from "./CompareSlider";

// jsdom no trae PointerEvent: sin esto los eventos llegan sin clientX/clientY.
class TestPointerEvent extends MouseEvent {
  readonly pointerId: number;

  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 0;
  }
}

beforeEach(() => {
  vi.stubGlobal("PointerEvent", TestPointerEvent);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const STAGE = { left: 0, top: 0, width: 400, height: 300 };

function renderSlider(fullResolution = true) {
  render(
    <CompareSlider
      beforeSrc="/before.jpg"
      afterSrc="/after.jpg"
      beforeAlt="Before: grandma.jpg"
      afterAlt="After: grandma.jpg"
      fullResolution={fullResolution}
    />,
  );
  const stage = screen.getByTestId("compare-stage");
  Object.defineProperty(stage, "clientWidth", { configurable: true, value: STAGE.width });
  Object.defineProperty(stage, "clientHeight", { configurable: true, value: STAGE.height });
  vi.spyOn(stage, "getBoundingClientRect").mockReturnValue({
    ...STAGE,
    x: 0,
    y: 0,
    right: STAGE.width,
    bottom: STAGE.height,
    toJSON: () => ({}),
  });
  return stage;
}

function loadAfterImage(naturalWidth: number, naturalHeight: number) {
  const image = screen.getByAltText("After: grandma.jpg");
  Object.defineProperty(image, "naturalWidth", { configurable: true, value: naturalWidth });
  Object.defineProperty(image, "naturalHeight", { configurable: true, value: naturalHeight });
  fireEvent.load(image);
}

function divider() {
  return screen.getByRole("slider", { name: "Before / after divider" });
}

function zoomedLayer() {
  return screen.getByAltText("After: grandma.jpg").parentElement as HTMLElement;
}

describe("CompareSlider", () => {
  it("shows both sides with the divider in the middle", () => {
    renderSlider();
    expect(screen.getByAltText("Before: grandma.jpg")).toHaveAttribute("src", "/before.jpg");
    expect(screen.getByAltText("After: grandma.jpg")).toHaveAttribute("src", "/after.jpg");
    expect(divider()).toHaveAttribute("aria-valuenow", "50");
  });

  it("moves the divider with the keyboard", () => {
    renderSlider();
    fireEvent.keyDown(divider(), { key: "ArrowLeft" });
    expect(divider()).toHaveAttribute("aria-valuenow", "48");
    fireEvent.keyDown(divider(), { key: "End" });
    expect(divider()).toHaveAttribute("aria-valuenow", "100");
    fireEvent.keyDown(divider(), { key: "Home" });
    expect(divider()).toHaveAttribute("aria-valuenow", "0");
  });

  it("drags the divider with the pointer", () => {
    const stage = renderSlider();
    fireEvent.pointerDown(divider(), { clientX: 200, pointerId: 1 });
    fireEvent.pointerMove(stage, { clientX: 100, pointerId: 1 });
    fireEvent.pointerUp(stage, { pointerId: 1 });
    expect(divider()).toHaveAttribute("aria-valuenow", "25");
    fireEvent.pointerMove(stage, { clientX: 300, pointerId: 1 });
    expect(divider()).toHaveAttribute("aria-valuenow", "25");
  });

  it("offers 100% only when the result is shown at full resolution", () => {
    renderSlider(true);
    loadAfterImage(1600, 1200);
    expect(screen.getByRole("status", { name: "Zoom level" })).toHaveTextContent("25%");
    fireEvent.click(screen.getByRole("button", { name: "100%" }));
    expect(screen.getByRole("status", { name: "Zoom level" })).toHaveTextContent("100%");
    expect(zoomedLayer().style.transform).toContain("scale(4)");
  });

  it("labels the zoom Preview and hides 100% when the view is a reduced copy", () => {
    renderSlider(false);
    loadAfterImage(8192, 6144);
    expect(screen.queryByRole("button", { name: "100%" })).not.toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Zoom level" })).toHaveTextContent("Preview");
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(screen.getByRole("status", { name: "Zoom level" })).toHaveTextContent("Preview");
  });

  it("zooms in, pans by dragging and fits back", () => {
    const stage = renderSlider();
    loadAfterImage(1600, 1200);
    fireEvent.click(screen.getByRole("button", { name: "Zoom in" }));
    expect(zoomedLayer().style.transform).toBe("translate(-100px, -75px) scale(1.5)");
    fireEvent.pointerDown(stage, { clientX: 200, clientY: 150, pointerId: 1 });
    fireEvent.pointerMove(stage, { clientX: 250, clientY: 170, pointerId: 1 });
    fireEvent.pointerUp(stage, { pointerId: 1 });
    expect(zoomedLayer().style.transform).toBe("translate(-50px, -55px) scale(1.5)");
    expect(divider()).toHaveAttribute("aria-valuenow", "50");
    fireEvent.click(screen.getByRole("button", { name: "Fit" }));
    expect(zoomedLayer().style.transform).toBe("translate(0px, 0px) scale(1)");
  });

  it("moves the divider when dragging the photo at fit", () => {
    const stage = renderSlider();
    fireEvent.pointerDown(stage, { clientX: 300, clientY: 150, pointerId: 1 });
    expect(divider()).toHaveAttribute("aria-valuenow", "75");
  });

  it("gives the photo the aspect ratio of the result", () => {
    const stage = renderSlider();
    loadAfterImage(1600, 1200);
    expect(stage.style.aspectRatio).toBe("1600 / 1200");
  });
});
