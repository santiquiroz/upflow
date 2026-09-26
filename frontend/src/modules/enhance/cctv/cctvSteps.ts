import type {
  CctvAnalysis,
  CctvFilterSchema,
  CctvLane,
  CctvParamSchema,
  CctvParamValue,
  CctvParams,
  CctvPreset,
  CctvPresetStep,
  CctvStepRequest,
  CctvStepSchema,
} from "../../../services/cctv";

export interface StepChoice {
  filter: string;
  params: CctvParams;
}

export type StepChoices = Readonly<Record<string, StepChoice>>;

export interface PresetContext {
  interlaced: boolean;
  sampleAspect: readonly [number, number] | null;
}

// El recorte, las cajas del OSD y la banda IA los arman sus propios controles y el backend.
const MANAGED_STEP_IDS: ReadonlySet<string> = new Set(["trim", "osd_protect", "ai_label"]);
const FILTER_PARAM = "filter";
const INTERPOLATED_SCALE_FLAGS: ReadonlySet<CctvParamValue> = new Set(["bicubic", "lanczos"]);

function isShownInLane(step: CctvStepSchema, lane: CctvLane): boolean {
  return lane === "ai" || step.category === "classic";
}

export function visibleSteps(catalog: readonly CctvStepSchema[], lane: CctvLane): CctvStepSchema[] {
  return catalog.filter((step) => !MANAGED_STEP_IDS.has(step.id) && isShownInLane(step, lane));
}

function parseSampleAspect(sar: string | undefined): readonly [number, number] | null {
  const [num, den] = (sar ?? "").split(":").map(Number);
  return Number.isInteger(num) && Number.isInteger(den) && num > 0 && den > 0 ? [num, den] : null;
}

export function presetContextOf(analysis: CctvAnalysis): PresetContext {
  return {
    interlaced: analysis.quality?.interlace?.interlaced ?? false,
    sampleAspect: parseSampleAspect(analysis.video.lite?.sar),
  };
}

function isAnamorphic(context: PresetContext): boolean {
  return context.sampleAspect !== null && context.sampleAspect[0] !== context.sampleAspect[1];
}

function presetStepApplies(step: CctvPresetStep, context: PresetContext): boolean {
  if (step.when === "interlaced") {
    return context.interlaced;
  }
  if (step.when === "anamorphic") {
    return isAnamorphic(context);
  }
  return true;
}

export function findFilter(step: CctvStepSchema, name: string): CctvFilterSchema | null {
  return step.filters.find((candidate) => candidate.name === name) ?? null;
}

function firstAvailableFilter(step: CctvStepSchema): CctvFilterSchema | null {
  return step.filters.find((candidate) => candidate.available) ?? null;
}

export function defaultChoice(step: CctvStepSchema): StepChoice | null {
  const filter = firstAvailableFilter(step);
  return filter ? { filter: filter.name, params: {} } : null;
}

function omitKey<T>(record: Readonly<Record<string, T>>, key: string): Record<string, T> {
  return Object.fromEntries(Object.entries(record).filter(([name]) => name !== key));
}

function presetParams(step: CctvPresetStep, context: PresetContext): CctvParams {
  if (step.id === "aspect" && context.sampleAspect !== null) {
    return { num: context.sampleAspect[0], den: context.sampleAspect[1] };
  }
  return omitKey(step.params, FILTER_PARAM);
}

function presetFilterName(presetStep: CctvPresetStep, step: CctvStepSchema): string | null {
  const named = presetStep.params[FILTER_PARAM];
  return typeof named === "string" ? named : firstAvailableFilter(step)?.name ?? null;
}

function presetChoice(presetStep: CctvPresetStep, step: CctvStepSchema, context: PresetContext): StepChoice | null {
  const filterName = presetFilterName(presetStep, step);
  const filter = filterName === null ? null : findFilter(step, filterName);
  if (!filter?.available) {
    return null;
  }
  return { filter: filter.name, params: presetParams(presetStep, context) };
}

export function choicesFromPreset(
  preset: CctvPreset,
  lane: CctvLane,
  context: PresetContext,
  catalog: readonly CctvStepSchema[],
): StepChoices {
  const visible = visibleSteps(catalog, lane);
  const entries = preset.lanes[lane].flatMap((presetStep) => {
    const step = visible.find((candidate) => candidate.id === presetStep.id);
    const choice = step && presetStepApplies(presetStep, context) ? presetChoice(presetStep, step, context) : null;
    return choice ? [[presetStep.id, choice] as const] : [];
  });
  return Object.fromEntries(entries);
}

export function withStepEnabled(choices: StepChoices, step: CctvStepSchema, enabled: boolean): StepChoices {
  const rest = omitKey(choices, step.id);
  const choice = enabled ? defaultChoice(step) : null;
  return choice ? { ...rest, [step.id]: choice } : rest;
}

export function withStepFilter(choices: StepChoices, stepId: string, filter: string): StepChoices {
  return { ...choices, [stepId]: { filter, params: {} } };
}

export function withStepParam(
  choices: StepChoices,
  stepId: string,
  name: string,
  value: CctvParamValue | null,
): StepChoices {
  const current = choices[stepId];
  if (!current) {
    return choices;
  }
  const others = omitKey(current.params, name);
  const params = value === null ? others : { ...others, [name]: value };
  return { ...choices, [stepId]: { ...current, params } };
}

export function paramValue(param: CctvParamSchema, choice: StepChoice): CctvParamValue | null {
  return choice.params[param.name] ?? param.default;
}

export function missingParams(step: CctvStepSchema, choice: StepChoice): string[] {
  const filter = findFilter(step, choice.filter);
  return (filter?.params ?? []).filter((param) => paramValue(param, choice) === null).map((param) => param.name);
}

export function incompleteStepIds(choices: StepChoices, catalog: readonly CctvStepSchema[]): string[] {
  return catalog
    .filter((step) => choices[step.id] && missingParams(step, choices[step.id]).length > 0)
    .map((step) => step.id);
}

export function stepRequests(choices: StepChoices, catalog: readonly CctvStepSchema[]): CctvStepRequest[] {
  return catalog
    .filter((step) => choices[step.id] !== undefined)
    .map((step) => ({ id: step.id, params: { [FILTER_PARAM]: choices[step.id].filter, ...choices[step.id].params } }));
}

export function stepWarningKey(stepId: string, choice: StepChoice): string | null {
  if (stepId === "sharpen") {
    return "cctv.sharpen.halos";
  }
  if (stepId === "scale" && INTERPOLATED_SCALE_FLAGS.has(choice.params.flags)) {
    return "cctv.scale.newPixels";
  }
  return null;
}
