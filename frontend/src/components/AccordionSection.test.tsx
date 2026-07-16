import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { AccordionSection } from "./AccordionSection";

function renderSection(defaultOpen = false) {
  render(
    <AccordionSection
      title="Model"
      summary="RealESRGAN x4plus"
      tooltip="Choose the AI model that upscales the file."
      defaultOpen={defaultOpen}
    >
      <p>Model options here</p>
    </AccordionSection>,
  );
}

function getToggle() {
  return screen.getByRole("button", { name: /^Model/ });
}

describe("AccordionSection", () => {
  it("starts collapsed by default, hiding the body but showing the summary", () => {
    renderSection();

    expect(getToggle()).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText("RealESRGAN x4plus")).toBeInTheDocument();
    expect(screen.getByText("Model options here")).not.toBeVisible();
  });

  it("expands on click, revealing the body and flipping aria-expanded", () => {
    renderSection();

    fireEvent.click(getToggle());

    expect(getToggle()).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Model options here")).toBeVisible();
  });

  it("collapses again on a second click", () => {
    renderSection();
    const toggle = getToggle();

    fireEvent.click(toggle);
    fireEvent.click(toggle);

    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText("Model options here")).not.toBeVisible();
  });

  it("honors defaultOpen", () => {
    renderSection(true);

    expect(getToggle()).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("Model options here")).toBeVisible();
  });

  it("keeps showing the summary while expanded", () => {
    renderSection();

    fireEvent.click(getToggle());

    expect(screen.getByText("RealESRGAN x4plus")).toBeInTheDocument();
  });

  it("toggles open on Enter", () => {
    renderSection();
    const toggle = getToggle();
    toggle.focus();

    fireEvent.keyDown(toggle, { key: "Enter" });

    expect(toggle).toHaveAttribute("aria-expanded", "true");
  });

  it("toggles open on Space", () => {
    renderSection();
    const toggle = getToggle();
    toggle.focus();

    fireEvent.keyDown(toggle, { key: " " });

    expect(toggle).toHaveAttribute("aria-expanded", "true");
  });

  it("links the toggle to the body via aria-controls", () => {
    renderSection(true);

    const toggle = getToggle();
    const controlsId = toggle.getAttribute("aria-controls");
    expect(controlsId).toBeTruthy();
    expect(document.getElementById(controlsId as string)).toContainElement(screen.getByText("Model options here"));
  });

  it("renders a tooltip trigger in the header explaining the section", () => {
    renderSection();

    expect(screen.getByRole("button", { name: "About Model" })).toBeInTheDocument();
  });

  it("keeps a motion-reduce-safe transition class on the chevron", () => {
    renderSection();

    const chevron = getToggle().querySelector("svg");
    expect(chevron).toHaveClass("motion-reduce:transition-none");
  });
});
