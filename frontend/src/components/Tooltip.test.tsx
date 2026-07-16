import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Tooltip } from "./Tooltip";

function renderTooltip() {
  render(<Tooltip label="About Model" content="Choose which upscaling model runs the job." />);
  return screen.getByRole("button", { name: "About Model" });
}

describe("Tooltip", () => {
  it("hides the tooltip content until triggered", () => {
    renderTooltip();

    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("shows the tooltip content on hover", () => {
    const trigger = renderTooltip();

    fireEvent.mouseEnter(trigger);

    expect(screen.getByRole("tooltip")).toHaveTextContent("Choose which upscaling model runs the job.");
  });

  it("shows the tooltip content on keyboard focus, not only on hover", () => {
    const trigger = renderTooltip();

    fireEvent.focus(trigger);

    expect(screen.getByRole("tooltip")).toBeInTheDocument();
  });

  it("hides on mouseleave", () => {
    const trigger = renderTooltip();

    fireEvent.mouseEnter(trigger);
    fireEvent.mouseLeave(trigger);

    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("hides on blur", () => {
    const trigger = renderTooltip();

    fireEvent.focus(trigger);
    fireEvent.blur(trigger);

    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("hides on Escape", () => {
    const trigger = renderTooltip();

    fireEvent.focus(trigger);
    fireEvent.keyDown(trigger, { key: "Escape" });

    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("associates the trigger with the tooltip content via aria-describedby", () => {
    const trigger = renderTooltip();

    fireEvent.focus(trigger);

    const tooltip = screen.getByRole("tooltip");
    expect(trigger).toHaveAttribute("aria-describedby", tooltip.id);
  });

  it("clears aria-describedby once hidden", () => {
    const trigger = renderTooltip();

    fireEvent.focus(trigger);
    fireEvent.blur(trigger);

    expect(trigger).not.toHaveAttribute("aria-describedby");
  });
});
