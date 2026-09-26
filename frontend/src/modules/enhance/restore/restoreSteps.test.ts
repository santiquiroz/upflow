import { describe, expect, it } from "vitest";
import type { RestoreStepCapability } from "../../../lib/restoreApiTypes";
import {
  durationText,
  enabledInChainOrder,
  initialSelection,
  isOverprocessing,
  restoreRequestOptions,
  selectionEta,
  selectPreset,
  setStepOption,
  stepOptions,
  summaryKey,
  toggleStep,
} from "./restoreSteps";
import { makeAnalysis } from "./restoreTestFixtures";

function capability(id: string, installed = true): RestoreStepCapability {
  return {
    id,
    phase: "native",
    strategy: "model",
    labelKey: `restore.step.${id}`,
    descriptionKey: `restore.step.${id}.description`,
    inventsDetail: false,
    warningKey: null,
    pack: installed ? null : "restore-core",
    installed,
  };
}

const CHAIN = ["descreen", "repair", "deblock", "denoise", "tone", "faces", "colorize"].map((id) => capability(id));

const ANALYSIS = makeAnalysis({
  proposedPreset: "gentle",
  proposedSteps: ["repair", "denoise"],
  proposedOptions: { repair: { sensitivity: 0.5 }, denoise: { strength: 0.3, keep_grain: 0.25 } },
  presetSelections: {
    gentle: { steps: ["repair", "denoise"], options: { repair: { sensitivity: 0.5 }, denoise: { strength: 0.3 } } },
    newspaper: {
      steps: ["descreen", "denoise"],
      options: { descreen: { mode: "halftone", strength: 1 }, denoise: { strength: 0.2 } },
    },
  },
});

describe("selección inicial y presets", () => {
  it("arranca con el preset y los pasos que propuso el análisis, sin tocar", () => {
    const state = initialSelection(ANALYSIS);

    expect(state.presetId).toBe("gentle");
    expect(state.customized).toBe(false);
    expect(state.enabled).toEqual(["repair", "denoise"]);
  });

  it("elegir otro preset usa la resolución del backend para esta foto", () => {
    const state = selectPreset(ANALYSIS, "newspaper");

    expect(state.enabled).toEqual(["descreen", "denoise"]);
    expect(stepOptions(state, "descreen")).toMatchObject({ mode: "halftone" });
    expect(state.customized).toBe(false);
  });

  it("un preset que el análisis no resolvió no enciende nada", () => {
    expect(selectPreset(ANALYSIS, "portrait").enabled).toEqual([]);
  });
});

describe("cambios a mano", () => {
  it("encender un paso lo agrega y marca el preset como tocado", () => {
    const state = toggleStep(initialSelection(ANALYSIS), "tone", true);

    expect(state.enabled).toContain("tone");
    expect(state.customized).toBe(true);
  });

  it("apagar un paso lo saca sin perder sus opciones", () => {
    const state = toggleStep(initialSelection(ANALYSIS), "repair", false);

    expect(state.enabled).toEqual(["denoise"]);
    expect(stepOptions(state, "repair").sensitivity).toBe(0.5);
  });

  it("cambiar una opción no muta el estado anterior", () => {
    const before = initialSelection(ANALYSIS);
    const after = setStepOption(before, "denoise", "keep_grain", 0.5);

    expect(stepOptions(after, "denoise").keep_grain).toBe(0.5);
    expect(stepOptions(before, "denoise").keep_grain).toBe(0.25);
    expect(after.customized).toBe(true);
  });

  it("un paso sin opciones propias muestra los defaults del backend", () => {
    expect(stepOptions(initialSelection(ANALYSIS), "denoise").keep_grain).toBe(0.25);
    expect(stepOptions(initialSelection(ANALYSIS), "tone")).toMatchObject({ keep_tone: true, neutral_gray: false });
  });
});

describe("pasos que se mandan", () => {
  it("van en el orden de la cadena, no en el orden en que se tildaron", () => {
    const state = toggleStep(toggleStep(initialSelection(ANALYSIS), "descreen", true), "colorize", true);

    expect(enabledInChainOrder(state, CHAIN)).toEqual(["descreen", "repair", "denoise", "colorize"]);
  });

  it("un paso sin su pack no se manda aunque el preset lo encienda", () => {
    const chain = CHAIN.map((step) => (step.id === "denoise" ? capability("denoise", false) : step));

    expect(enabledInChainOrder(initialSelection(ANALYSIS), chain)).toEqual(["repair"]);
  });

  it("el preset intacto viaja con su nombre y las opciones de cada paso encendido", () => {
    const state = initialSelection(ANALYSIS);

    expect(restoreRequestOptions(state, ["repair", "denoise"])).toEqual({
      preset: "gentle",
      repair: { engine: "fast", sensitivity: 0.5, grow_px: 0 },
      denoise: { strength: 0.3, keep_grain: 0.25 },
    });
  });

  it("un preset tocado no se nombra, y un paso apagado no manda opciones", () => {
    const state = toggleStep(initialSelection(ANALYSIS), "repair", false);

    expect(restoreRequestOptions(state, ["denoise"])).toEqual({ denoise: { strength: 0.3, keep_grain: 0.25 } });
  });
});

describe("tiempo estimado y avisos", () => {
  const perStep = {
    repair: { gpuSeconds: 2, cpuSeconds: 10 },
    denoise: { gpuSeconds: 8, cpuSeconds: 60 },
  };

  it("suma el costo de los pasos encendidos", () => {
    expect(selectionEta(perStep, ["repair", "denoise"])).toEqual({ gpuSeconds: 10, cpuSeconds: 70 });
    expect(selectionEta(perStep, [])).toEqual({ gpuSeconds: 0, cpuSeconds: 0 });
  });

  it("avisa sobreprocesado con descreen halftone y denoise por encima del límite", () => {
    const newspaper = selectPreset(ANALYSIS, "newspaper");
    const strong = setStepOption(newspaper, "denoise", "strength", 0.5);

    expect(isOverprocessing(newspaper, ["descreen", "denoise"], 0.3)).toBe(false);
    expect(isOverprocessing(strong, ["descreen", "denoise"], 0.3)).toBe(true);
    expect(isOverprocessing(strong, ["denoise"], 0.3)).toBe(false);
  });

  it("escribe segundos hasta el minuto y medio y minutos después", () => {
    expect(durationText(40)).toEqual({ key: "restore.duration.seconds", count: 40 });
    expect(durationText(0.2)).toEqual({ key: "restore.duration.seconds", count: 1 });
    expect(durationText(240)).toEqual({ key: "restore.duration.minutes", count: 4 });
  });

  it("elige la clave del resumen por cantidad", () => {
    expect([0, 1, 5].map(summaryKey)).toEqual([
      "restore.summary.none",
      "restore.summary.one",
      "restore.summary.many",
    ]);
  });
});
