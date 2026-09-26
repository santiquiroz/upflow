import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RestoreCapture, RestoreGeometry } from "../../../lib/restoreApiTypes";
import { GeometryTools } from "./GeometryTools";

const NEUTRAL: RestoreGeometry = { rotate90: 0, crop: null, angle: 0 };

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

function renderTools(geometry: RestoreGeometry = NEUTRAL, busy = false, capture?: RestoreCapture) {
  const onApply = vi.fn();
  render(
    <GeometryTools
      previewUrl="/api/v1/restore/analysis/tok-1/preview.jpg?v=1"
      alt="Working copy of grandma.jpg"
      geometry={geometry}
      workingSize={{ width: 400, height: 200 }}
      busy={busy}
      capture={capture}
      onApply={onApply}
    />,
  );
  return onApply;
}

function capture(overrides: Partial<RestoreCapture> = {}): RestoreCapture {
  return { autoCrop: null, photos: [], perspective: null, frameWidth: 400, frameHeight: 200, ...overrides };
}

const AUTO_CROP: RestoreGeometry = { rotate90: 0, crop: [40, 20, 300, 150], angle: 2.5 };
const SHEET_PHOTOS: RestoreGeometry[] = [
  { rotate90: 0, crop: [20, 20, 160, 120], angle: 0 },
  { rotate90: 0, crop: [220, 40, 160, 120], angle: -3 },
];
const KEYSTONE: RestoreGeometry = {
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

function mockBox(element: HTMLElement) {
  vi.spyOn(element, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    width: 200,
    height: 100,
    right: 200,
    bottom: 100,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  });
}

function dragCorner(name: string, to: [number, number]) {
  const overlay = screen.getByTestId("restore-perspective-overlay");
  mockBox(overlay);
  fireEvent.pointerDown(screen.getByRole("button", { name }), { pointerId: 2 });
  fireEvent.pointerMove(overlay, { clientX: to[0], clientY: to[1], pointerId: 2 });
  fireEvent.pointerUp(overlay, { pointerId: 2 });
}

function dragOnOverlay(from: [number, number], to: [number, number]) {
  const overlay = screen.getByTestId("restore-crop-overlay");
  vi.spyOn(overlay, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    width: 200,
    height: 100,
    right: 200,
    bottom: 100,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  });
  fireEvent.pointerDown(overlay, { clientX: from[0], clientY: from[1], pointerId: 1 });
  fireEvent.pointerMove(overlay, { clientX: to[0], clientY: to[1], pointerId: 1 });
  fireEvent.pointerUp(overlay, { pointerId: 1 });
}

describe("GeometryTools", () => {
  it("shows the working copy", () => {
    renderTools();

    expect(screen.getByRole("img", { name: "Working copy of grandma.jpg" })).toHaveAttribute(
      "src",
      "/api/v1/restore/analysis/tok-1/preview.jpg?v=1",
    );
  });

  it("rotates a quarter turn clockwise right away", () => {
    const onApply = renderTools();

    fireEvent.click(screen.getByRole("button", { name: "Rotate" }));

    expect(onApply).toHaveBeenCalledWith({ rotate90: 1, crop: null, angle: 0 });
  });

  it("applies the crop dragged on the photo in working-copy pixels", () => {
    const onApply = renderTools();

    fireEvent.click(screen.getByRole("button", { name: "Crop" }));
    dragOnOverlay([50, 25], [150, 75]);
    fireEvent.click(screen.getByRole("button", { name: "Apply crop" }));

    expect(onApply).toHaveBeenCalledWith({ rotate90: 0, crop: [100, 50, 200, 100], angle: 0 });
  });

  it("keeps Apply crop disabled until an area is drawn", () => {
    renderTools();

    fireEvent.click(screen.getByRole("button", { name: "Crop" }));

    expect(screen.getByRole("button", { name: "Apply crop" })).toBeDisabled();
    expect(screen.getByText("Drag on the photo to mark the area to keep.")).toBeInTheDocument();
  });

  it("previews the straighten angle with a grid and applies it on request", () => {
    const onApply = renderTools();

    fireEvent.click(screen.getByRole("button", { name: "Straighten" }));
    fireEvent.change(screen.getByRole("slider", { name: "Straighten" }), { target: { value: "2.5" } });

    expect(screen.getByTestId("restore-straighten-grid")).toBeInTheDocument();
    expect(screen.getByRole("img")).toHaveStyle({ transform: "rotate(-2.5deg)" });
    expect(screen.getByText("2.5°")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Apply straighten" }));
    expect(onApply).toHaveBeenCalledWith({ rotate90: 0, crop: null, angle: 2.5 });
  });

  it("cancels a straighten without sending anything", () => {
    const onApply = renderTools();

    fireEvent.click(screen.getByRole("button", { name: "Straighten" }));
    fireEvent.change(screen.getByRole("slider", { name: "Straighten" }), { target: { value: "5" } });
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onApply).not.toHaveBeenCalled();
    expect(screen.queryByTestId("restore-straighten-grid")).not.toBeInTheDocument();
    expect(screen.getByRole("img").style.transform).toBe("");
  });

  it("offers to remove the crop and to reset the framing only when there is one", () => {
    const onApply = renderTools({ rotate90: 1, crop: [10, 10, 100, 100], angle: 1 });

    fireEvent.click(screen.getByRole("button", { name: "Remove crop" }));
    expect(onApply).toHaveBeenLastCalledWith({ rotate90: 1, crop: null, angle: 1 });
    fireEvent.click(screen.getByRole("button", { name: "Reset framing" }));
    expect(onApply).toHaveBeenLastCalledWith({ rotate90: 0, crop: null, angle: 0 });
  });

  it("hides remove and reset for an untouched photo", () => {
    renderTools();

    expect(screen.queryByRole("button", { name: "Remove crop" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reset framing" })).not.toBeInTheDocument();
  });

  it("disables every tool while the working copy is being updated", () => {
    renderTools(NEUTRAL, true);

    expect(screen.getByRole("button", { name: "Rotate" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Crop" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Straighten" })).toBeDisabled();
  });

  it("offers the auto crop the analysis found and applies it", () => {
    const onApply = renderTools(NEUTRAL, false, capture({ autoCrop: AUTO_CROP }));

    expect(screen.getByText("The photo sits on a larger scanner sheet.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Auto crop" }));

    expect(onApply).toHaveBeenCalledWith(AUTO_CROP);
  });

  it("shows an applied suggestion as pressed and does not send it again", () => {
    const onApply = renderTools(AUTO_CROP, false, capture({ autoCrop: AUTO_CROP }));

    const button = screen.getByRole("button", { name: "Auto crop" });
    fireEvent.click(button);

    expect(button).toHaveAttribute("aria-pressed", "true");
    expect(onApply).not.toHaveBeenCalled();
  });

  it("numbers every photo found on a scanner sheet and restores the one picked", () => {
    const onApply = renderTools(NEUTRAL, false, capture({ photos: SHEET_PHOTOS }));

    expect(
      screen.getByText("2 photos found on this sheet. Pick the one to restore; the others stay on the sheet for later."),
    ).toBeInTheDocument();
    const outlines = screen.getByTestId("restore-photo-outlines");
    expect(outlines).toHaveTextContent("12");
    expect(outlines.querySelectorAll("polygon")).toHaveLength(2);
    fireEvent.click(screen.getByRole("button", { name: "Photo 2" }));

    expect(onApply).toHaveBeenCalledWith(SHEET_PHOTOS[1]);
  });

  it("hides the sheet outlines once a photo is cropped", () => {
    renderTools(SHEET_PHOTOS[0], false, capture({ photos: SHEET_PHOTOS }));

    expect(screen.queryByTestId("restore-photo-outlines")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Photo 1" })).toHaveAttribute("aria-pressed", "true");
  });

  it("fixes the perspective with the corners the analysis found", () => {
    const onApply = renderTools(NEUTRAL, false, capture({ perspective: KEYSTONE }));

    expect(screen.getByText("The print looks photographed at an angle.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Fix perspective" }));

    expect(onApply).toHaveBeenCalledWith(KEYSTONE);
  });

  it("opens the corners at the suggestion and nudges one with the arrow keys", () => {
    const onApply = renderTools(NEUTRAL, false, capture({ perspective: KEYSTONE }));

    fireEvent.click(screen.getByRole("button", { name: "Adjust corners" }));
    fireEvent.keyDown(screen.getByRole("button", { name: "Top-left corner" }), { key: "ArrowRight" });
    fireEvent.click(screen.getByRole("button", { name: "Apply perspective" }));

    expect(onApply).toHaveBeenCalledWith({
      rotate90: 0,
      crop: null,
      angle: 0,
      corners: [
        [42, 20],
        [360, 40],
        [380, 180],
        [20, 160],
      ],
    });
  });

  it("drags a corner by hand on an untouched photo", () => {
    const onApply = renderTools();

    fireEvent.click(screen.getByRole("button", { name: "Perspective" }));
    expect(screen.getByText(/Drag each corner onto a corner of the print/)).toBeInTheDocument();
    dragCorner("Top-left corner", [40, 20]);
    fireEvent.click(screen.getByRole("button", { name: "Apply perspective" }));

    expect(onApply.mock.calls[0][0].corners[0]).toEqual([80, 40]);
    expect(onApply.mock.calls[0][0].angle).toBe(0);
  });

  it("refuses crossed corners", () => {
    const onApply = renderTools();

    fireEvent.click(screen.getByRole("button", { name: "Perspective" }));
    dragCorner("Top-left corner", [195, 98]);

    expect(screen.getByRole("button", { name: "Apply perspective" })).toBeDisabled();
    expect(
      screen.getByText("The corners cross each other. Move them so they follow the edge of the print."),
    ).toBeInTheDocument();
    expect(onApply).not.toHaveBeenCalled();
  });

  it("with perspective set offers to remove it and locks the hand straighten", () => {
    const onApply = renderTools({ ...KEYSTONE, crop: [5, 5, 100, 100] }, false, capture({ perspective: KEYSTONE }));

    expect(screen.getByRole("button", { name: "Straighten" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Adjust corners" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Remove perspective" }));

    expect(onApply).toHaveBeenCalledWith({ rotate90: 0, crop: null, angle: 0 });
  });

  it("disables the perspective tool when an older analysis cannot place a cropped copy", () => {
    renderTools({ rotate90: 0, crop: [10, 10, 100, 100], angle: 0 });

    expect(screen.getByRole("button", { name: "Perspective" })).toBeDisabled();
  });
});
