import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  SlowPresetCostHint,
  estimateSoftwareEncodeMinutes,
  shouldWarnAboutSlowPreset,
} from "./SlowPresetCostHint";

describe("shouldWarnAboutSlowPreset", () => {
  it("warns on a slow software encode at 3x or above", () => {
    expect(shouldWarnAboutSlowPreset("software", "slow", 4)).toBe(true);
    expect(shouldWarnAboutSlowPreset("software", "veryslow", 3)).toBe(true);
  });

  it("stays quiet on the GPU encoder, which ignores x264/x265 presets", () => {
    expect(shouldWarnAboutSlowPreset("auto", "slow", 4)).toBe(false);
  });

  it("stays quiet for fast presets and small scales", () => {
    expect(shouldWarnAboutSlowPreset("software", "medium", 4)).toBe(false);
    expect(shouldWarnAboutSlowPreset("software", "slow", 2)).toBe(false);
    expect(shouldWarnAboutSlowPreset("software", "slow", null)).toBe(false);
  });
});

describe("estimateSoftwareEncodeMinutes", () => {
  it("matches the measured 4x anchor (~112 min for a 24-minute episode)", () => {
    expect(estimateSoftwareEncodeMinutes(4)).toBe(112);
  });

  it("scales down with output pixel count", () => {
    expect(estimateSoftwareEncodeMinutes(3)).toBeLessThan(estimateSoftwareEncodeMinutes(4));
    expect(estimateSoftwareEncodeMinutes(2)).toBeLessThan(estimateSoftwareEncodeMinutes(3));
  });
});

describe("SlowPresetCostHint", () => {
  it("renders the estimate and the GPU suggestion when warranted", () => {
    render(<SlowPresetCostHint encoder="software" preset="slow" scale={4} />);
    const note = screen.getByRole("note");
    expect(note).toHaveTextContent(/112 min of encoding alone/);
    expect(note).toHaveTextContent(/Auto \(GPU\)/);
  });

  it("renders nothing on the GPU encoder", () => {
    const { container } = render(<SlowPresetCostHint encoder="auto" preset="slow" scale={4} />);
    expect(container).toBeEmptyDOMElement();
  });
});
