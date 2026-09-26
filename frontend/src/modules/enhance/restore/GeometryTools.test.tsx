import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { RestoreGeometry } from "../../../lib/restoreApiTypes";
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

function renderTools(geometry: RestoreGeometry = NEUTRAL, busy = false) {
  const onApply = vi.fn();
  render(
    <GeometryTools
      previewUrl="/api/v1/restore/analysis/tok-1/preview.jpg?v=1"
      alt="Working copy of grandma.jpg"
      geometry={geometry}
      workingSize={{ width: 400, height: 200 }}
      busy={busy}
      onApply={onApply}
    />,
  );
  return onApply;
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
});
