import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { InterpEngineControls, interpEngineLabel } from "./InterpEngineControls";

describe("InterpEngineControls", () => {
  it("renders one button per available engine", () => {
    render(<InterpEngineControls engines={["rife", "gmfss"]} value="rife" onChange={vi.fn()} />);

    expect(screen.getByRole("button", { name: "RIFE" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /GMFSS/ })).toBeInTheDocument();
  });

  it("marks the selected engine as pressed and the rest as not pressed", () => {
    render(<InterpEngineControls engines={["rife", "gmfss"]} value="gmfss" onChange={vi.fn()} />);

    expect(screen.getByRole("button", { name: /GMFSS/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "RIFE" })).toHaveAttribute("aria-pressed", "false");
  });

  it("calls onChange with the picked engine id", () => {
    const onChange = vi.fn();
    render(<InterpEngineControls engines={["rife", "gmfss"]} value="rife" onChange={onChange} />);

    screen.getByRole("button", { name: /GMFSS/ }).click();

    expect(onChange).toHaveBeenCalledWith("gmfss");
  });

  it("labels the GMFSS button as very slow directly, not just higher quality", () => {
    render(<InterpEngineControls engines={["rife", "gmfss"]} value="rife" onChange={vi.fn()} />);

    expect(screen.getByRole("button", { name: /very slow/i })).toBeInTheDocument();
  });

  it("shows the cost hint only once GMFSS is the selected engine", () => {
    const { rerender } = render(
      <InterpEngineControls engines={["rife", "gmfss"]} value="rife" onChange={vi.fn()} />,
    );
    expect(screen.queryByRole("status")).not.toBeInTheDocument();

    rerender(<InterpEngineControls engines={["rife", "gmfss"]} value="gmfss" onChange={vi.fn()} />);
    expect(screen.getByRole("status")).toHaveTextContent(/10x or more/i);
  });
});

describe("interpEngineLabel", () => {
  it("returns a plain label for rife", () => {
    expect(interpEngineLabel("rife")).toBe("RIFE");
  });

  it("returns a very-slow-qualified label for gmfss", () => {
    expect(interpEngineLabel("gmfss")).toMatch(/GMFSS/);
    expect(interpEngineLabel("gmfss")).toMatch(/very slow/i);
  });

  it("falls back to the raw engine id for an unknown engine", () => {
    expect(interpEngineLabel("mystery")).toBe("mystery");
  });
});
