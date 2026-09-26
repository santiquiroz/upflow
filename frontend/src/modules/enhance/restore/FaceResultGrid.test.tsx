import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { RecomposeResponse } from "../../../lib/restoreApiTypes";
import { FaceResultGrid } from "./FaceResultGrid";
import type { RestoredFace } from "./faceResultModel";

const FACES: RestoredFace[] = [
  { index: 0, enabled: true, blend: 0.6 },
  { index: 2, enabled: true, blend: 0.5 },
];
const SIDECAR = { faces: [], compositeReasons: [] };
const FACE_WARNING = "AI-restored face. It may not look like the real person.";

interface GridOptions {
  faces?: RestoredFace[];
  recomposeAvailable?: boolean;
  recompose?: (jobId: string, faces: Record<number, { enabled: boolean; blend: number }>) => Promise<RecomposeResponse>;
}

function renderGrid({
  faces = FACES,
  recomposeAvailable = true,
  recompose = vi.fn().mockResolvedValue({ sidecar: SIDECAR }),
}: GridOptions = {}) {
  const onRecomposed = vi.fn();
  render(
    <FaceResultGrid
      jobId="job-1"
      faces={faces}
      recomposeAvailable={recomposeAvailable}
      onRecomposed={onRecomposed}
      recompose={recompose}
    />,
  );
  return { onRecomposed, recompose };
}

function tile(number: number) {
  return screen.getByRole("listitem", { name: `Face ${number}` });
}

function restoredLayer(number: number) {
  return within(tile(number)).getByAltText(`Face ${number} after`);
}

function applyButton() {
  return screen.getByRole("button", { name: "Apply" });
}

describe("FaceResultGrid", () => {
  it("shows each restored face before and after, from the job artifacts", () => {
    renderGrid();
    expect(within(tile(1)).getByAltText("Face 1 before")).toHaveAttribute("src", "/api/v1/jobs/job-1/artifacts/face%3A0%3Abefore");
    expect(restoredLayer(1)).toHaveAttribute("src", "/api/v1/jobs/job-1/artifacts/face%3A0%3Aafter");
    expect(restoredLayer(3)).toHaveAttribute("src", "/api/v1/jobs/job-1/artifacts/face%3A2%3Aafter");
  });

  it("previews the blend over the original face", () => {
    renderGrid();
    expect(restoredLayer(1)).toHaveStyle({ opacity: "0.6" });
    fireEvent.change(within(tile(1)).getByRole("slider", { name: "Blend with original" }), { target: { value: "0.25" } });
    expect(restoredLayer(1)).toHaveStyle({ opacity: "0.25" });
  });

  it("warns on every restored face", () => {
    renderGrid();
    expect(within(tile(1)).getByText(FACE_WARNING)).toBeInTheDocument();
    expect(within(tile(3)).getByText(FACE_WARNING)).toBeInTheDocument();
  });

  it("puts the original face back with Show original face", () => {
    renderGrid();
    fireEvent.click(within(tile(1)).getByRole("checkbox", { name: "Show original face" }));
    expect(restoredLayer(1)).toHaveStyle({ opacity: "0" });
    expect(within(tile(1)).getByRole("slider", { name: "Blend with original" })).toBeDisabled();
    expect(within(tile(1)).queryByText(FACE_WARNING)).not.toBeInTheDocument();
  });

  it("starts with the original shown for a face that was already put back", () => {
    renderGrid({ faces: [{ index: 0, enabled: false, blend: 0.6 }] });
    expect(within(tile(1)).getByRole("checkbox", { name: "Show original face" })).toBeChecked();
    expect(restoredLayer(1)).toHaveStyle({ opacity: "0" });
  });

  it("applies only after a change and sends every restored face", async () => {
    const { recompose, onRecomposed } = renderGrid();
    expect(applyButton()).toBeDisabled();

    fireEvent.click(within(tile(3)).getByRole("checkbox", { name: "Show original face" }));
    fireEvent.change(within(tile(1)).getByRole("slider", { name: "Blend with original" }), { target: { value: "0.8" } });
    fireEvent.click(applyButton());

    expect(recompose).toHaveBeenCalledWith("job-1", {
      0: { enabled: true, blend: 0.8 },
      2: { enabled: false, blend: 0.5 },
    });
    await waitFor(() => expect(onRecomposed).toHaveBeenCalledWith(SIDECAR));
    expect(screen.getByRole("status")).toHaveTextContent("Faces updated.");
  });

  it("does not apply twice while the photo is being rebuilt", () => {
    const recompose = vi.fn().mockReturnValue(new Promise(() => undefined));
    renderGrid({ recompose });
    fireEvent.change(within(tile(1)).getByRole("slider", { name: "Blend with original" }), { target: { value: "0.8" } });
    fireEvent.click(applyButton());
    expect(screen.getByRole("button", { name: "Applying…" })).toBeDisabled();
    expect(within(tile(1)).getByRole("slider", { name: "Blend with original" })).toBeDisabled();
  });

  it("shows the server error and lets the user try again", async () => {
    const recompose = vi.fn().mockRejectedValue(new Error("This result has no saved faces to recompose."));
    const { onRecomposed } = renderGrid({ recompose });
    fireEvent.change(within(tile(1)).getByRole("slider", { name: "Blend with original" }), { target: { value: "0.8" } });
    fireEvent.click(applyButton());

    expect(await screen.findByRole("alert")).toHaveTextContent("This result has no saved faces to recompose.");
    expect(applyButton()).toBeEnabled();
    expect(onRecomposed).not.toHaveBeenCalled();
  });

  it("explains why faces cannot be changed when their files were not saved", () => {
    renderGrid({ recomposeAvailable: false });
    expect(
      screen.getByText("Faces can't be changed on this result because their working files were not saved."),
    ).toBeInTheDocument();
    expect(within(tile(1)).getByRole("checkbox", { name: "Show original face" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Apply" })).not.toBeInTheDocument();
  });
});
