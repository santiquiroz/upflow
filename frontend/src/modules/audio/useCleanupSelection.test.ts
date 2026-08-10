import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { CleanupStep } from "../../lib/apiTypes";
import { useCleanupSelection } from "./useCleanupSelection";

// El catálogo llega del backend YA en orden de ejecución, con `covers` como
// única fuente de la exclusión: la UI no conoce ids concretos.
const CATALOG: CleanupStep[] = [
  {
    id: "denoise",
    name: "UVR DeNoise by FoxJoy",
    family: "denoise",
    covers: ["denoise"],
    installed: true,
    descriptionKey: "audio.karaoke.model.denoise.description",
  },
  {
    id: "deecho_normal",
    name: "UVR De-Echo Normal by FoxJoy",
    family: "deecho",
    covers: ["deecho"],
    installed: true,
    descriptionKey: "audio.karaoke.model.deecho_normal.description",
  },
  {
    id: "deecho_aggressive",
    name: "UVR De-Echo Aggressive by FoxJoy",
    family: "deecho",
    covers: ["deecho"],
    installed: true,
    descriptionKey: "audio.karaoke.model.deecho_aggressive.description",
  },
  {
    id: "deecho_dereverb",
    name: "UVR DeEcho-DeReverb by FoxJoy",
    family: "deecho",
    covers: ["deecho", "dereverb"],
    installed: true,
    descriptionKey: "audio.karaoke.model.deecho_dereverb.description",
  },
  {
    id: "reverb_hq",
    name: "Reverb HQ by FoxJoy",
    family: "dereverb",
    covers: ["dereverb"],
    installed: true,
    descriptionKey: "audio.karaoke.model.reverb_hq.description",
  },
];

function setup() {
  return renderHook(() => useCleanupSelection(CATALOG, 3));
}

describe("useCleanupSelection", () => {
  it("starts off, because cleanup is opt-in like the voice chain", () => {
    const { result } = setup();

    expect(result.current.active).toBe(false);
    expect(result.current.enabledIds).toEqual([]);
  });

  it("reports nothing enabled while the chain is off, even after ticking steps", () => {
    const { result } = setup();

    act(() => result.current.toggleStep("denoise", true));

    expect(result.current.enabledIds).toEqual([]);
  });

  it("keeps the catalog order no matter the order they were ticked", () => {
    const { result } = setup();
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStep("reverb_hq", true));
    act(() => result.current.toggleStep("denoise", true));
    act(() => result.current.toggleStep("deecho_normal", true));

    expect(result.current.enabledIds).toEqual(["denoise", "deecho_normal", "reverb_hq"]);
  });

  it("turns off the other intensity of the same model", () => {
    const { result } = setup();
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStep("deecho_normal", true));
    act(() => result.current.toggleStep("deecho_aggressive", true));

    expect(result.current.enabledIds).toEqual(["deecho_aggressive"]);
  });

  it("lets the two-family model turn off both de-echo and de-reverb", () => {
    const { result } = setup();
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStep("deecho_normal", true));
    act(() => result.current.toggleStep("reverb_hq", true));
    act(() => result.current.toggleStep("deecho_dereverb", true));

    expect(result.current.enabledIds).toEqual(["deecho_dereverb"]);
  });

  it("keeps de-noise on when a de-echo model replaces another", () => {
    // La exclusión es POR FAMILIA: no puede llevarse puesto un paso que ataca
    // otro defecto.
    const { result } = setup();
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStep("denoise", true));
    act(() => result.current.toggleStep("deecho_normal", true));
    act(() => result.current.toggleStep("deecho_dereverb", true));

    expect(result.current.enabledIds).toEqual(["denoise", "deecho_dereverb"]);
  });

  it("reports which steps a given one would replace", () => {
    const { result } = setup();

    expect(result.current.conflictsOf("deecho_dereverb").sort()).toEqual([
      "deecho_aggressive",
      "deecho_normal",
      "reverb_hq",
    ]);
    expect(result.current.conflictsOf("denoise")).toEqual([]);
  });

  it("unticking removes only that step", () => {
    const { result } = setup();
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStep("denoise", true));
    act(() => result.current.toggleStep("reverb_hq", true));
    act(() => result.current.toggleStep("denoise", false));

    expect(result.current.enabledIds).toEqual(["reverb_hq"]);
  });

  it("flags over-processing from the third pass on", () => {
    const { result } = setup();
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStep("denoise", true));
    act(() => result.current.toggleStep("deecho_normal", true));
    expect(result.current.isOverprocessing).toBe(false);

    act(() => result.current.toggleStep("reverb_hq", true));
    expect(result.current.isOverprocessing).toBe(true);
  });

  it("survives an undefined catalog while capabilities are loading", () => {
    const { result } = renderHook(() => useCleanupSelection(undefined));

    act(() => result.current.setActive(true));
    act(() => result.current.toggleStep("denoise", true));

    expect(result.current.enabledIds).toEqual([]);
  });
});
