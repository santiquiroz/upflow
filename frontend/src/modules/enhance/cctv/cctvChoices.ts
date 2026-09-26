import type {
  CctvAnalysis,
  CctvBox,
  CctvJobRequest,
  CctvLane,
  CctvPreset,
  CctvPresetsResponse,
  CctvTask,
} from "../../../services/cctv";
import { defaultTask } from "./cctvLanes";
import { choicesFromPreset, presetContextOf, stepRequests, type StepChoices } from "./cctvSteps";

export interface CctvChoices {
  lane: CctvLane;
  task: CctvTask;
  presetId: string | null;
  steps: StepChoices;
  noOsd: boolean;
  osdBoxes: readonly CctvBox[];
  osdBoxesConfirmed: boolean;
}

function findPreset(presets: CctvPresetsResponse, presetId: string | null): CctvPreset | null {
  return presets.presets.find((preset) => preset.id === presetId) ?? null;
}

function suggestedPresetId(analysis: CctvAnalysis, presets: CctvPresetsResponse): string | null {
  return findPreset(presets, analysis.suggestedPreset)?.id ?? presets.presets[0]?.id ?? null;
}

function presetSteps(
  presetId: string | null,
  lane: CctvLane,
  analysis: CctvAnalysis,
  presets: CctvPresetsResponse,
): StepChoices {
  const preset = findPreset(presets, presetId);
  return preset ? choicesFromPreset(preset, lane, presetContextOf(analysis), presets.steps[lane]) : {};
}

export function initialChoices(analysis: CctvAnalysis, presets: CctvPresetsResponse): CctvChoices {
  const presetId = suggestedPresetId(analysis, presets);
  return {
    lane: "classic",
    task: defaultTask("classic"),
    presetId,
    steps: presetSteps(presetId, "classic", analysis, presets),
    noOsd: false,
    osdBoxes: [],
    osdBoxesConfirmed: false,
  };
}

export function withLane(
  choices: CctvChoices,
  lane: CctvLane,
  analysis: CctvAnalysis,
  presets: CctvPresetsResponse,
): CctvChoices {
  return { ...choices, lane, task: defaultTask(lane), steps: presetSteps(choices.presetId, lane, analysis, presets) };
}

export function withPreset(
  choices: CctvChoices,
  presetId: string,
  analysis: CctvAnalysis,
  presets: CctvPresetsResponse,
): CctvChoices {
  return { ...choices, presetId, steps: presetSteps(presetId, choices.lane, analysis, presets) };
}

export function buildCctvJobRequest(
  token: string,
  choices: CctvChoices,
  presets: CctvPresetsResponse,
): CctvJobRequest {
  return {
    token,
    task: choices.task,
    preset: choices.presetId,
    steps: stepRequests(choices.steps, presets.steps[choices.lane]),
    osdBoxes: choices.noOsd ? [] : [...choices.osdBoxes],
    osdBoxesConfirmed: !choices.noOsd && choices.osdBoxesConfirmed,
    noOsd: choices.noOsd,
  };
}
