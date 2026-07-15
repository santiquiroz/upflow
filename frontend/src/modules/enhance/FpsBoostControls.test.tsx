import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { FpsBoostControls } from "./FpsBoostControls";

describe("FpsBoostControls", () => {
  it("marks Off as active and leaves both groups enabled when neither mode is set", () => {
    render(<FpsBoostControls value={{ fpsMultiplier: 1, targetFps: null }} onChange={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Off" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "2×" })).not.toBeDisabled();
    expect(screen.getByRole("button", { name: "59.94 fps" })).not.toBeDisabled();
  });

  it("calls onChange with the multiplier and a cleared targetFps when a multiplier is picked", () => {
    const onChange = vi.fn();
    render(<FpsBoostControls value={{ fpsMultiplier: 1, targetFps: null }} onChange={onChange} />);

    screen.getByRole("button", { name: "3×" }).click();

    expect(onChange).toHaveBeenCalledWith({ fpsMultiplier: 3, targetFps: null });
  });

  it("calls onChange with the target and a reset multiplier when a target fps is picked", () => {
    const onChange = vi.fn();
    render(<FpsBoostControls value={{ fpsMultiplier: 1, targetFps: null }} onChange={onChange} />);

    screen.getByRole("button", { name: "60 fps" }).click();

    expect(onChange).toHaveBeenCalledWith({ fpsMultiplier: 1, targetFps: "60/1" });
  });

  it("disables the target buttons and clears any target once a multiplier is active", () => {
    render(<FpsBoostControls value={{ fpsMultiplier: 4, targetFps: null }} onChange={vi.fn()} />);

    expect(screen.getByRole("button", { name: "59.94 fps" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "60 fps" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "4×" })).toHaveAttribute("aria-pressed", "true");
  });

  it("disables the multiplier buttons once a target fps is active", () => {
    render(<FpsBoostControls value={{ fpsMultiplier: 1, targetFps: "60000/1001" }} onChange={vi.fn()} />);

    expect(screen.getByRole("button", { name: "2×" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "3×" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "4×" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "59.94 fps" })).toHaveAttribute("aria-pressed", "true");
  });

  it("returns to the off state, re-enabling both groups, when Off is picked from a target-active state", () => {
    const onChange = vi.fn();
    render(<FpsBoostControls value={{ fpsMultiplier: 1, targetFps: "60/1" }} onChange={onChange} />);

    screen.getByRole("button", { name: "Off" }).click();

    expect(onChange).toHaveBeenCalledWith({ fpsMultiplier: 1, targetFps: null });
  });

  it("keeps the Off button enabled regardless of the active mode", () => {
    render(<FpsBoostControls value={{ fpsMultiplier: 4, targetFps: null }} onChange={vi.fn()} />);

    expect(screen.getByRole("button", { name: "Off" })).not.toBeDisabled();
  });
});
