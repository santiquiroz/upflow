import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { en } from "../../../i18n/en";
import type { CctvStepRequest } from "../../../services/cctv";
import { CctvFilterPreview } from "./CctvFilterPreview";

const STEPS: CctvStepRequest[] = [{ id: "deblock", params: { filter: "deblock", filter_type: "strong", block: 8 } }];
const DISPLAY = { width: 1920, height: 1080 };

function renderPreview(frame = 40, steps: CctvStepRequest[] = STEPS, skipsAiSteps = false) {
  return render(<CctvFilterPreview token="tok-1" frame={frame} steps={steps} skipsAiSteps={skipsAiSteps} displaySize={DISPLAY} />);
}

function previewButton(): HTMLElement {
  return screen.getByRole("button", { name: en["cctv.preview.action"] });
}

function processedImage(frame = 40): HTMLImageElement {
  return screen.getByAltText(`Frame ${frame}, processed`) as HTMLImageElement;
}

describe("CctvFilterPreview", () => {
  it("processes the steps around the current frame only when asked", () => {
    renderPreview();

    expect(screen.queryByRole("img")).not.toBeInTheDocument();

    fireEvent.click(previewButton());

    const url = new URL(processedImage().src, "http://localhost");
    expect(url.pathname).toBe("/api/v1/video/cctv/tok-1/preview");
    expect(url.searchParams.get("frame")).toBe("40");
    expect(JSON.parse(url.searchParams.get("steps") ?? "")).toEqual(STEPS);
    expect(screen.getByAltText("Frame 40, original").getAttribute("src")).toBe("/api/v1/video/cctv/tok-1/preview?frame=40");
    expect(screen.getByRole("status")).toHaveTextContent(en["cctv.preview.loading"]);
  });

  it("blends the processed frame over the original with the transparency slider", () => {
    renderPreview();
    fireEvent.click(previewButton());
    fireEvent.load(processedImage());

    expect(processedImage()).toHaveStyle({ opacity: "0.5" });

    fireEvent.change(screen.getByRole("slider", { name: en["cctv.preview.blend"] }), { target: { value: "100" } });

    expect(processedImage()).toHaveStyle({ opacity: "1" });
    expect(screen.getByRole("slider", { name: en["cctv.preview.blend"] })).toHaveAttribute("aria-valuetext", "100% processed");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("says the preview is out of date after the frame or the steps change", () => {
    const view = renderPreview();
    fireEvent.click(previewButton());

    view.rerender(<CctvFilterPreview token="tok-1" frame={41} steps={STEPS} skipsAiSteps={false} displaySize={DISPLAY} />);

    expect(screen.getByText(en["cctv.preview.stale"])).toBeInTheDocument();
    expect(processedImage(40)).toBeInTheDocument();
  });

  it("reports a preview the backend couldn't process", () => {
    renderPreview();
    fireEvent.click(previewButton());

    fireEvent.error(processedImage());

    expect(screen.getByRole("alert")).toHaveTextContent(en["cctv.preview.failed"]);
  });

  it("can't preview without steps and says AI steps are left out", () => {
    renderPreview(40, [], true);

    expect(previewButton()).toBeDisabled();
    expect(screen.getByText(en["cctv.preview.classicOnly"])).toBeInTheDocument();
  });
});
