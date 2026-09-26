import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { en } from "../../../i18n/en";
import { CctvStepCard } from "./CctvStepCard";
import { CLASSIC_STEPS, LENS } from "./cctvFixtures";
import type { StepChoice } from "./cctvSteps";

function stepById(id: string) {
  const found = [...CLASSIC_STEPS, LENS].find((step) => step.id === id);
  if (!found) throw new Error(id);
  return found;
}

function renderCard(id: string, choice: StepChoice | null) {
  const handlers = { onToggle: vi.fn(), onFilterChange: vi.fn(), onParamChange: vi.fn(), onParamsReplace: vi.fn() };
  render(
    <ol>
      <CctvStepCard step={stepById(id)} choice={choice} {...handlers} />
    </ol>,
  );
  return handlers;
}

describe("CctvStepCard", () => {
  it("toggles the step", () => {
    const handlers = renderCard("gray", null);

    fireEvent.click(screen.getByRole("checkbox", { name: en["cctv.step.gray"] }));

    expect(handlers.onToggle).toHaveBeenCalledWith(true);
  });

  it("describes the chosen filter in plain language with its documentation", () => {
    renderCard("deblock", { filter: "deblock", params: {} });

    expect(screen.getByText(/Applied deblock\./)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: en["cctv.step.docs"] })).toHaveAttribute(
      "href",
      "https://ffmpeg.org/ffmpeg-filters.html#deblock",
    );
  });

  it("offers only the filters this build has", () => {
    renderCard("deblock", { filter: "deblock", params: {} });

    expect(screen.getByRole("option", { name: "fspp" })).toBeDisabled();
    expect(screen.getByRole("option", { name: "deblock" })).toBeEnabled();
  });

  it("edits numbers and enums as typed values, and clears to the default", () => {
    const handlers = renderCard("deblock", { filter: "deblock", params: {} });

    fireEvent.change(screen.getByLabelText("block"), { target: { value: "16" } });
    fireEvent.change(screen.getByLabelText("filter_type"), { target: { value: "strong" } });
    fireEvent.change(screen.getByLabelText("block"), { target: { value: "" } });

    expect(handlers.onParamChange).toHaveBeenNthCalledWith(1, "block", 16);
    expect(handlers.onParamChange).toHaveBeenNthCalledWith(2, "filter_type", "strong");
    expect(handlers.onParamChange).toHaveBeenNthCalledWith(3, "block", null);
  });

  it("asks for required settings that have no default", () => {
    renderCard("crop", { filter: "crop", params: { w: 640 } });

    expect(screen.getByText("Fill in h to use this step.")).toBeInTheDocument();
  });

  it("offers lens presets outside the advanced settings and applies all their values at once", () => {
    const handlers = renderCard("lens", { filter: "lenscorrection", params: {} });

    const preset = screen.getByLabelText(en["cctv.step.preset"]);
    expect(preset).toHaveValue("");
    expect(preset).toHaveAccessibleDescription(en["cctv.step.presetHint"]);
    fireEvent.change(preset, { target: { value: "very_wide" } });

    expect(handlers.onParamsReplace).toHaveBeenCalledWith({ k1: -0.22, k2: -0.02 });
    expect(handlers.onParamChange).not.toHaveBeenCalled();
  });

  it("shows the preset the current values came from, in the viewer's language", () => {
    renderCard("lens", { filter: "lenscorrection", params: { k1: -0.12, k2: -0.01 } });

    expect(screen.getByLabelText(en["cctv.step.preset"])).toHaveValue("wide");
    expect(screen.getByRole("option", { name: en["cctv.filter.lenscorrection.preset.wide"] })).toBeInTheDocument();
    expect(screen.getByText(en["cctv.lens.moved"])).toBeInTheDocument();
  });

  it("has no preset picker for filters without presets", () => {
    renderCard("lens", { filter: "v360", params: {} });

    expect(screen.queryByLabelText(en["cctv.step.preset"])).not.toBeInTheDocument();
  });

  it("warns that sharpening can create halos", () => {
    renderCard("sharpen", { filter: "cas", params: {} });

    expect(screen.getByText(en["cctv.sharpen.halos"])).toBeInTheDocument();
  });
});
