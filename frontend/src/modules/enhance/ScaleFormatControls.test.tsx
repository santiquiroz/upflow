import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ScaleFormatControls } from "./ScaleFormatControls";

describe("ScaleFormatControls", () => {
  it("renders a tabular-numbered button per allowed scale and marks the active one", () => {
    render(
      <ScaleFormatControls
        allowedScales={[2, 3, 4]}
        scale={4}
        onScaleChange={vi.fn()}
        format="png"
        onFormatChange={vi.fn()}
      />,
    );

    const activeScale = screen.getByRole("button", { name: "4x" });
    expect(activeScale).toHaveAttribute("aria-pressed", "true");
    expect(activeScale).toHaveClass("font-mono-tabular");
    expect(screen.getByRole("button", { name: "2x" })).toHaveAttribute("aria-pressed", "false");
  });

  it("calls onScaleChange when a different scale is picked", () => {
    const onScaleChange = vi.fn();
    render(
      <ScaleFormatControls
        allowedScales={[2, 3, 4]}
        scale={4}
        onScaleChange={onScaleChange}
        format="png"
        onFormatChange={vi.fn()}
      />,
    );

    screen.getByRole("button", { name: "2x" }).click();

    expect(onScaleChange).toHaveBeenCalledWith(2);
  });

  it("renders a button per output format and marks the active one", () => {
    render(
      <ScaleFormatControls
        allowedScales={[2, 3, 4]}
        scale={4}
        onScaleChange={vi.fn()}
        format="webp"
        onFormatChange={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "WEBP" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "PNG" })).toHaveAttribute("aria-pressed", "false");
  });

  it("calls onFormatChange when a different format is picked", () => {
    const onFormatChange = vi.fn();
    render(
      <ScaleFormatControls
        allowedScales={[2, 3, 4]}
        scale={4}
        onScaleChange={vi.fn()}
        format="png"
        onFormatChange={onFormatChange}
      />,
    );

    screen.getByRole("button", { name: "JPG" }).click();

    expect(onFormatChange).toHaveBeenCalledWith("jpg");
  });
});
