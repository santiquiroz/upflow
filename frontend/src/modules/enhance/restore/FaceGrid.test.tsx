import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RestoreFace } from "../../../lib/restoreApiTypes";
import { FaceGrid } from "./FaceGrid";
import { makeFace } from "./restoreTestFixtures";

const LARGE_BLURRY = makeFace({ index: 0, eyePx: 48, sharpness: 0.02, confidence: 0.99, enabled: true, blend: 0.6 });
const LARGE_CLEAR = makeFace({ index: 1, eyePx: 40, sharpness: 0.2, confidence: 0.95, enabled: false, blend: 0.4 });
const SMALL = makeFace({ index: 2, eyePx: 20, sharpness: 0.02, confidence: 0.93, enabled: false, blend: 0.5 });
const VERY_SMALL = makeFace({ index: 3, eyePx: 12, sharpness: 0.02, confidence: 0.92, enabled: false, blend: 0.4 });
const TINY = makeFace({ index: 4, eyePx: 6, sharpness: 0.02, confidence: 0.91, enabled: false, blend: 0.4 });
const PROPOSED = [LARGE_BLURRY, LARGE_CLEAR, SMALL, VERY_SMALL, TINY];

interface GridProps {
  faces?: RestoreFace[];
  stepBlend?: number;
  imageRevision?: number;
}

function renderGrid({ faces = PROPOSED, stepBlend = 0.6, imageRevision = 2 }: GridProps = {}) {
  const onChange = vi.fn();
  render(
    <FaceGrid
      faces={faces}
      proposed={PROPOSED}
      stepBlend={stepBlend}
      imageRevision={imageRevision}
      onChange={onChange}
    />,
  );
  return onChange;
}

function tile(number: number) {
  return screen.getByRole("listitem", { name: `Face ${number}` });
}

function checkbox(number: number) {
  return within(tile(number)).getByRole("checkbox", { name: `Restore face ${number}` });
}

describe("FaceGrid", () => {
  it("warns that faces are AI-generated detail", () => {
    renderGrid();
    expect(screen.getByText("AI-generated facial detail. It can change how a person looks.")).toBeInTheDocument();
  });

  it("lists the faces from largest to smallest with their size and confidence", () => {
    const shuffled = [SMALL, LARGE_CLEAR, LARGE_BLURRY];
    renderGrid({ faces: shuffled });
    const names = screen.getAllByRole("listitem").map((item) => item.getAttribute("aria-label"));
    expect(names).toEqual(["Face 1", "Face 2", "Face 3"]);
    expect(within(tile(1)).getByText("48 px between the eyes · 99% confidence")).toBeInTheDocument();
  });

  it("shows each face thumbnail refreshed after a rotation or crop", () => {
    renderGrid({ imageRevision: 3 });
    expect(within(tile(1)).getByRole("img", { name: "Face 1" })).toHaveAttribute(
      "src",
      "/api/v1/restore/analysis/tok-1/face-0.jpg?v=3",
    );
  });

  it("labels each face with the policy for its size and sharpness", () => {
    renderGrid();
    expect(within(tile(1)).getByText("AI-restored face")).toBeInTheDocument();
    expect(within(tile(2)).getByText("Already clear — restoring may change it")).toBeInTheDocument();
    expect(within(tile(3)).getByText("Small face — most detail will be invented")).toBeInTheDocument();
    expect(within(tile(4)).getByText("Too small to restore faithfully")).toBeInTheDocument();
    expect(within(tile(5)).getByText("Too small")).toBeInTheDocument();
  });

  it("checks only the faces the policy turned on", () => {
    renderGrid();
    expect(checkbox(1)).toBeChecked();
    expect(checkbox(2)).not.toBeChecked();
    expect(checkbox(3)).not.toBeChecked();
  });

  it("does not let a face under 8 px be restored", () => {
    renderGrid({ faces: PROPOSED.map((face) => (face.index === 4 ? { ...face, enabled: true } : face)) });
    expect(checkbox(5)).toBeDisabled();
    expect(checkbox(5)).not.toBeChecked();
  });

  it("turns a small face on with one click", () => {
    const onChange = renderGrid();
    fireEvent.click(checkbox(3));
    expect(onChange).toHaveBeenCalledWith(2, { enabled: true });
  });

  it("asks before restoring a face under 16 px", () => {
    const onChange = renderGrid();
    fireEvent.click(checkbox(4));
    expect(onChange).not.toHaveBeenCalled();
    expect(within(tile(4)).getByText(/almost all of its detail would be invented/)).toBeInTheDocument();

    fireEvent.click(within(tile(4)).getByRole("button", { name: "Restore anyway" }));
    expect(onChange).toHaveBeenCalledWith(3, { enabled: true });
  });

  it("keeps a very small face off when the confirmation is cancelled", () => {
    const onChange = renderGrid();
    fireEvent.click(checkbox(4));
    fireEvent.click(within(tile(4)).getByRole("button", { name: "Cancel" }));
    expect(onChange).not.toHaveBeenCalled();
    expect(within(tile(4)).queryByRole("button", { name: "Restore anyway" })).not.toBeInTheDocument();
  });

  it("turns a face off without asking", () => {
    const onChange = renderGrid();
    fireEvent.click(checkbox(1));
    expect(onChange).toHaveBeenCalledWith(0, { enabled: false });
  });

  it("shows a blend slider only on the faces that will be restored, following the step blend", () => {
    renderGrid({ stepBlend: 0.8 });
    const slider = within(tile(1)).getByRole("slider", { name: "Blend with original" });
    expect(slider).toHaveValue("0.8");
    expect(within(tile(1)).getByText("80%")).toBeInTheDocument();
    expect(within(tile(2)).queryByRole("slider")).not.toBeInTheDocument();
  });

  it("sets the blend of one face", () => {
    const onChange = renderGrid();
    fireEvent.change(within(tile(1)).getByRole("slider", { name: "Blend with original" }), {
      target: { value: "0.35" },
    });
    expect(onChange).toHaveBeenCalledWith(0, { blend: 0.35 });
  });

  it("says when no face will be restored", () => {
    renderGrid({ faces: PROPOSED.map((face) => ({ ...face, enabled: false })) });
    expect(screen.getByText("No face is selected, so no face will be restored.")).toBeInTheDocument();
  });
});
