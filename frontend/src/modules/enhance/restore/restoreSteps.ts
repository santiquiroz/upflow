import type {
  RestoreAnalysis,
  RestoreOptions,
  RestoreStepCapability,
  RestoreStepOptions,
  StepEta,
} from "../../../lib/restoreApiTypes";

export type SliderFormat = "percent" | "pixels";

export interface SliderControl {
  kind: "slider";
  option: string;
  labelKey: string;
  min: number;
  max: number;
  step: number;
  format: SliderFormat;
}

export interface ToggleControl {
  kind: "toggle";
  option: string;
  labelKey: string;
}

// La etiqueta de cada valor es `${labelKey}.${valor}`.
export interface ChoiceControl {
  kind: "choice";
  option: string;
  labelKey: string;
  choices: readonly string[];
}

export type StepControl = SliderControl | ToggleControl | ChoiceControl;

export interface StepUi {
  defaults: RestoreStepOptions;
  intensity: SliderControl;
  advanced: readonly StepControl[];
}

export interface RestoreSelectionState {
  token: string;
  presetId: string;
  customized: boolean;
  enabled: readonly string[];
  options: Readonly<Record<string, RestoreStepOptions>>;
}

// Espejo de los topes de app/schemas_restore.py (MAX_GROW_PX, MAX_SATURATION).
const MAX_GROW_PX = 3;
const MAX_SATURATION = 2;

function unitSlider(option: string, labelKey: string, max = 1): SliderControl {
  return { kind: "slider", option, labelKey, min: 0, max, step: 0.05, format: "percent" };
}

const STRENGTH = unitSlider("strength", "restore.control.strength");

// Los defaults son los de photo_restore_runners.py: lo que el backend haria con
// el paso encendido sin opciones, para que el deslizador muestre lo que va a correr.
export const STEP_UI: Readonly<Record<string, StepUi>> = {
  descreen: {
    defaults: { mode: "auto", strength: 1 },
    intensity: STRENGTH,
    advanced: [{ kind: "choice", option: "mode", labelKey: "restore.control.mode", choices: ["auto", "halftone", "texture"] }],
  },
  repair: {
    defaults: { engine: "fast", sensitivity: 0.5, grow_px: 0 },
    intensity: unitSlider("sensitivity", "restore.control.sensitivity"),
    advanced: [
      { kind: "choice", option: "engine", labelKey: "restore.control.engine", choices: ["fast", "classic"] },
      {
        kind: "slider",
        option: "grow_px",
        labelKey: "restore.control.growPx",
        min: -MAX_GROW_PX,
        max: MAX_GROW_PX,
        step: 1,
        format: "pixels",
      },
    ],
  },
  deblock: { defaults: { strength: 0.5 }, intensity: STRENGTH, advanced: [] },
  denoise: {
    defaults: { strength: 0.3, keep_grain: 0.25 },
    intensity: STRENGTH,
    advanced: [unitSlider("keep_grain", "restore.control.keepGrain")],
  },
  tone: {
    defaults: { strength: 0.7, keep_tone: true, neutral_gray: false, fix_faded: false, local_contrast: false },
    intensity: STRENGTH,
    advanced: [
      { kind: "toggle", option: "fix_faded", labelKey: "restore.control.fixFaded" },
      { kind: "toggle", option: "keep_tone", labelKey: "restore.control.keepTone" },
      { kind: "toggle", option: "neutral_gray", labelKey: "restore.control.neutralGray" },
      { kind: "toggle", option: "local_contrast", labelKey: "restore.control.localContrast" },
    ],
  },
  faces: { defaults: { blend: 0.6 }, intensity: unitSlider("blend", "restore.control.faceBlend"), advanced: [] },
  colorize: {
    defaults: { strength: 1, saturation: 1 },
    intensity: STRENGTH,
    advanced: [unitSlider("saturation", "restore.control.saturation", MAX_SATURATION)],
  },
};

export function stepUi(stepId: string): StepUi | null {
  return STEP_UI[stepId] ?? null;
}

export function initialSelection(analysis: RestoreAnalysis): RestoreSelectionState {
  return {
    token: analysis.token,
    presetId: analysis.proposedPreset,
    customized: false,
    enabled: analysis.proposedSteps,
    options: analysis.proposedOptions,
  };
}

export function selectPreset(analysis: RestoreAnalysis, presetId: string): RestoreSelectionState {
  const resolved = analysis.presetSelections[presetId] ?? { steps: [], options: {} };
  return { token: analysis.token, presetId, customized: false, enabled: resolved.steps, options: resolved.options };
}

export function toggleStep(state: RestoreSelectionState, stepId: string, enabled: boolean): RestoreSelectionState {
  const others = state.enabled.filter((id) => id !== stepId);
  return { ...state, customized: true, enabled: enabled ? [...others, stepId] : others };
}

export function setStepOption(
  state: RestoreSelectionState,
  stepId: string,
  option: string,
  value: unknown,
): RestoreSelectionState {
  const current = state.options[stepId] ?? {};
  return { ...state, customized: true, options: { ...state.options, [stepId]: { ...current, [option]: value } } };
}

export function stepOptions(state: RestoreSelectionState, stepId: string): RestoreStepOptions {
  return { ...stepUi(stepId)?.defaults, ...state.options[stepId] };
}

// El orden sale del catalogo (= orden de ejecucion del backend), no del orden
// en que se tildaron; un paso sin su pack no se manda porque el backend da 400.
export function enabledInChainOrder(state: RestoreSelectionState, steps: readonly RestoreStepCapability[]): string[] {
  return steps.filter((step) => step.installed && state.enabled.includes(step.id)).map((step) => step.id);
}

export function selectionEta(perStep: Readonly<Record<string, StepEta>>, enabledIds: readonly string[]): StepEta {
  const entries = enabledIds.map((id) => perStep[id] ?? { gpuSeconds: 0, cpuSeconds: 0 });
  return {
    gpuSeconds: entries.reduce((total, entry) => total + entry.gpuSeconds, 0),
    cpuSeconds: entries.reduce((total, entry) => total + entry.cpuSeconds, 0),
  };
}

// Mismo criterio que photo_restore_chain.is_overprocessing.
export function isOverprocessing(
  state: RestoreSelectionState,
  enabledIds: readonly string[],
  denoiseLimit: number,
): boolean {
  if (!enabledIds.includes("descreen") || !enabledIds.includes("denoise")) {
    return false;
  }
  const halftone = stepOptions(state, "descreen").mode === "halftone";
  return halftone && Number(stepOptions(state, "denoise").strength) > denoiseLimit;
}

// Un preset tocado a mano ya no es ese preset: nombrarlo en el sidecar diria
// que corrio algo que no corrio.
export function restoreRequestOptions(state: RestoreSelectionState, enabledIds: readonly string[]): RestoreOptions {
  const perStep = Object.fromEntries(enabledIds.map((id) => [id, stepOptions(state, id)]));
  return state.customized ? perStep : { preset: state.presetId, ...perStep };
}

export function summaryKey(count: number): string {
  if (count === 0) return "restore.summary.none";
  if (count === 1) return "restore.summary.one";
  return "restore.summary.many";
}

const SECONDS_PER_MINUTE = 60;
// Por debajo de un minuto y medio, "1 min" o "2 min" esconde demasiado.
const MINUTES_FROM_SECONDS = 90;

export interface DurationText {
  key: string;
  count: number;
}

export function durationText(seconds: number): DurationText {
  if (seconds < MINUTES_FROM_SECONDS) {
    return { key: "restore.duration.seconds", count: Math.max(1, Math.round(seconds)) };
  }
  return { key: "restore.duration.minutes", count: Math.round(seconds / SECONDS_PER_MINUTE) };
}
