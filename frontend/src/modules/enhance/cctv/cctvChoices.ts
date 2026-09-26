import type {
  CctvAnalysis,
  CctvBox,
  CctvJobRequest,
  CctvLane,
  CctvPreset,
  CctvPresetsResponse,
  CctvStepSchema,
  CctvTask,
} from "../../../services/cctv";
import { aiUpscaleFields, withAiUpscaleStep, type AiUpscaleChoice } from "./cctvAiUpscale";
import { suggestedOsdBoxes } from "./cctvBoxes";
import { caseRequestOf, EMPTY_CASE_DETAILS, type CaseDetails } from "./cctvCase";
import { storedSizeOf, type TrimRange } from "./cctvFrames";
import { defaultTask } from "./cctvLanes";
import { EMPTY_ROI, roiCatalog, roiRequest, type RoiChoice } from "./cctvRoi";
import { choicesFromPreset, presetContextOf, stepRequests, type StepChoices } from "./cctvSteps";

export interface CctvChoices {
  lane: CctvLane;
  task: CctvTask;
  presetId: string | null;
  steps: StepChoices;
  noOsd: boolean;
  osdBoxes: readonly CctvBox[];
  osdBoxesConfirmed: boolean;
  trim: TrimRange | null;
  caseDetails: CaseDetails;
  aiUpscale: AiUpscaleChoice | null;
  roi: RoiChoice;
}

type JobTarget = Pick<CctvJobRequest, "osdBoxes" | "osdBoxesConfirmed" | "noOsd" | "trim" | "roi">;

function findPreset(presets: CctvPresetsResponse, presetId: string | null): CctvPreset | null {
  return presets.presets.find((preset) => preset.id === presetId) ?? null;
}

function suggestedPresetId(analysis: CctvAnalysis, presets: CctvPresetsResponse): string | null {
  return findPreset(presets, analysis.suggestedPreset)?.id ?? presets.presets[0]?.id ?? null;
}

// La foto multi-cuadro es clasica en los dos carriles y solo desentrelaza y desbloquea.
function stepLaneOf(lane: CctvLane, task: CctvTask): CctvLane {
  return task === "roi_fusion" ? "classic" : lane;
}

export function catalogFor(presets: CctvPresetsResponse, lane: CctvLane, task: CctvTask): CctvStepSchema[] {
  const catalog = presets.steps[stepLaneOf(lane, task)];
  return task === "roi_fusion" ? roiCatalog(catalog) : catalog;
}

function presetSteps(
  presetId: string | null,
  lane: CctvLane,
  task: CctvTask,
  analysis: CctvAnalysis,
  presets: CctvPresetsResponse,
): StepChoices {
  const preset = findPreset(presets, presetId);
  const catalog = catalogFor(presets, lane, task);
  return preset ? choicesFromPreset(preset, stepLaneOf(lane, task), presetContextOf(analysis), catalog) : {};
}

export function initialChoices(analysis: CctvAnalysis, presets: CctvPresetsResponse): CctvChoices {
  const presetId = suggestedPresetId(analysis, presets);
  const task = defaultTask("classic");
  return {
    lane: "classic",
    task,
    presetId,
    steps: presetSteps(presetId, "classic", task, analysis, presets),
    noOsd: false,
    osdBoxes: suggestedOsdBoxes(storedSizeOf(analysis)),
    osdBoxesConfirmed: false,
    trim: null,
    caseDetails: EMPTY_CASE_DETAILS,
    aiUpscale: null,
    roi: EMPTY_ROI,
  };
}

export function withLane(
  choices: CctvChoices,
  lane: CctvLane,
  analysis: CctvAnalysis,
  presets: CctvPresetsResponse,
): CctvChoices {
  const task = defaultTask(lane);
  const steps = presetSteps(choices.presetId, lane, task, analysis, presets);
  return { ...choices, lane, task, steps, aiUpscale: null };
}

export function withTask(
  choices: CctvChoices,
  task: CctvTask,
  analysis: CctvAnalysis,
  presets: CctvPresetsResponse,
): CctvChoices {
  return { ...choices, task, steps: presetSteps(choices.presetId, choices.lane, task, analysis, presets) };
}

export function withPreset(
  choices: CctvChoices,
  presetId: string,
  analysis: CctvAnalysis,
  presets: CctvPresetsResponse,
): CctvChoices {
  return { ...choices, presetId, steps: presetSteps(presetId, choices.lane, choices.task, analysis, presets) };
}

// Mover o dibujar una caja invalida la confirmacion: hay que volver a mirarla sobre un cuadro con la hora.
export function withOsdBoxes(choices: CctvChoices, osdBoxes: readonly CctvBox[]): CctvChoices {
  return { ...choices, osdBoxes, osdBoxesConfirmed: false };
}

export function withOsdConfirmed(choices: CctvChoices): CctvChoices {
  return { ...choices, noOsd: false, osdBoxesConfirmed: choices.osdBoxes.length > 0 };
}

export function withNoOsd(choices: CctvChoices, noOsd: boolean): CctvChoices {
  return { ...choices, noOsd, osdBoxesConfirmed: false };
}

export function withTrim(choices: CctvChoices, trim: TrimRange | null): CctvChoices {
  return { ...choices, trim };
}

export function withCaseDetails(choices: CctvChoices, caseDetails: CaseDetails): CctvChoices {
  return { ...choices, caseDetails };
}

export function withAiUpscale(choices: CctvChoices, aiUpscale: AiUpscaleChoice | null): CctvChoices {
  return { ...choices, aiUpscale };
}

export function withRoi(choices: CctvChoices, roi: RoiChoice): CctvChoices {
  return { ...choices, roi };
}

function videoTarget(choices: CctvChoices): JobTarget {
  return {
    osdBoxes: choices.noOsd ? [] : [...choices.osdBoxes],
    osdBoxesConfirmed: !choices.noOsd && choices.osdBoxesConfirmed,
    noOsd: choices.noOsd,
    trim: choices.trim ? [choices.trim[0], choices.trim[1]] : null,
  };
}

// La foto usa su propio rango y no toca el OSD: el backend rechaza recorte y cajas en esta tarea.
function roiTarget(choices: CctvChoices): JobTarget {
  const roi = roiRequest(choices.roi);
  return { osdBoxes: [], osdBoxesConfirmed: false, noOsd: false, trim: null, ...(roi ? { roi } : {}) };
}

function jobTarget(choices: CctvChoices): JobTarget {
  return choices.task === "roi_fusion" ? roiTarget(choices) : videoTarget(choices);
}

function aiUpscaleOf(choices: CctvChoices): AiUpscaleChoice | null {
  return choices.task === "enhance" ? choices.aiUpscale : null;
}

export function buildCctvJobRequest(
  token: string,
  choices: CctvChoices,
  presets: CctvPresetsResponse,
): CctvJobRequest {
  const catalog = catalogFor(presets, choices.lane, choices.task);
  const aiUpscale = aiUpscaleOf(choices);
  return {
    token,
    task: choices.task,
    preset: choices.presetId,
    steps: stepRequests(withAiUpscaleStep(choices.steps, catalog, aiUpscale), catalog),
    ...jobTarget(choices),
    ...aiUpscaleFields(aiUpscale),
    ...caseRequestOf(choices.caseDetails),
  };
}
