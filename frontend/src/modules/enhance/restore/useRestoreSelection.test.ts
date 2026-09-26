import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { RestoreAnalysis, RestoreCapabilities } from "../../../lib/restoreApiTypes";
import { makeAnalysis, makeCapabilities } from "./restoreTestFixtures";
import { useRestoreSelection } from "./useRestoreSelection";

const PER_STEP = {
  descreen: { gpuSeconds: 1, cpuSeconds: 2 },
  repair: { gpuSeconds: 3, cpuSeconds: 20 },
  denoise: { gpuSeconds: 10, cpuSeconds: 60 },
  tone: { gpuSeconds: 1, cpuSeconds: 1 },
};

function analysisFor(token: string): RestoreAnalysis {
  return makeAnalysis({
    token,
    proposedPreset: "gentle",
    proposedSteps: ["repair", "denoise"],
    proposedOptions: { repair: { sensitivity: 0.5 }, denoise: { strength: 0.3 } },
    presetSelections: {
      gentle: { steps: ["repair", "denoise"], options: { repair: { sensitivity: 0.5 }, denoise: { strength: 0.3 } } },
      newspaper: {
        steps: ["descreen", "denoise"],
        options: { descreen: { mode: "halftone", strength: 1 }, denoise: { strength: 0.2 } },
      },
    },
    eta: { gpuSeconds: 13, cpuSeconds: 80, perStep: PER_STEP },
  });
}

function renderSelection(analysis: RestoreAnalysis, capabilities: RestoreCapabilities = makeCapabilities()) {
  return renderHook(({ current }) => useRestoreSelection(current, capabilities), {
    initialProps: { current: analysis },
  });
}

describe("useRestoreSelection", () => {
  it("starts from the proposal with its preset, its steps in chain order and its time", () => {
    const { result } = renderSelection(analysisFor("tok-1"));

    expect(result.current.presetId).toBe("gentle");
    expect(result.current.enabledIds).toEqual(["repair", "denoise"]);
    expect(result.current.eta).toEqual({ gpuSeconds: 13, cpuSeconds: 80 });
    expect(result.current.requestOptions.preset).toBe("gentle");
  });

  it("follows the steps the user turns on and off, and the time follows them", () => {
    const { result } = renderSelection(analysisFor("tok-1"));

    act(() => result.current.toggleStep("tone", true));
    act(() => result.current.toggleStep("repair", false));

    expect(result.current.enabledIds).toEqual(["denoise", "tone"]);
    expect(result.current.eta).toEqual({ gpuSeconds: 11, cpuSeconds: 61 });
    expect(result.current.customized).toBe(true);
    expect(result.current.requestOptions.preset).toBeUndefined();
  });

  it("switches to another preset as the backend resolved it for this photo", () => {
    const { result } = renderSelection(analysisFor("tok-1"));

    act(() => result.current.choosePreset("newspaper"));

    expect(result.current.presetId).toBe("newspaper");
    expect(result.current.enabledIds).toEqual(["descreen", "denoise"]);
    expect(result.current.optionsOf("descreen").mode).toBe("halftone");
    expect(result.current.customized).toBe(false);
  });

  it("warns about overprocessing from the options, not only from the steps", () => {
    const { result } = renderSelection(analysisFor("tok-1"));

    act(() => result.current.choosePreset("newspaper"));
    expect(result.current.isOverprocessing).toBe(false);
    act(() => result.current.setOption("denoise", "strength", 0.6));

    expect(result.current.isOverprocessing).toBe(true);
  });

  it("leaves out steps whose pack is missing", () => {
    const { result } = renderSelection(analysisFor("tok-1"), makeCapabilities(["restore-core"]));

    expect(result.current.enabledIds).toEqual([]);
  });

  it("keeps the user's choices after a geometry change and starts over with a new photo", () => {
    const { result, rerender } = renderSelection(analysisFor("tok-1"));
    act(() => result.current.toggleStep("tone", true));

    rerender({ current: { ...analysisFor("tok-1"), width: 800 } });
    expect(result.current.enabledIds).toContain("tone");

    rerender({ current: analysisFor("tok-2") });
    expect(result.current.enabledIds).toEqual(["repair", "denoise"]);
    expect(result.current.customized).toBe(false);
  });
});
