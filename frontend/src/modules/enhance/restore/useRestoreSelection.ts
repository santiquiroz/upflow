import { useMemo, useState } from "react";
import type {
  RestoreAnalysis,
  RestoreCapabilities,
  RestoreOptions,
  RestoreStepOptions,
  StepEta,
} from "../../../lib/restoreApiTypes";
import {
  enabledInChainOrder,
  initialSelection,
  isOverprocessing,
  restoreRequestOptions,
  selectionEta,
  selectPreset,
  setStepOption,
  stepOptions,
  toggleStep,
  type RestoreSelectionState,
} from "./restoreSteps";

export interface RestoreSelection {
  presetId: string;
  customized: boolean;
  enabledIds: string[];
  isEnabled: (stepId: string) => boolean;
  optionsOf: (stepId: string) => RestoreStepOptions;
  choosePreset: (presetId: string) => void;
  toggleStep: (stepId: string, enabled: boolean) => void;
  setOption: (stepId: string, option: string, value: unknown) => void;
  eta: StepEta;
  isOverprocessing: boolean;
  requestOptions: RestoreOptions;
}

// Una foto nueva (otro token) arranca de su propia propuesta; girar o recortar
// la misma foto conserva lo que el usuario ya eligio.
function currentState(state: RestoreSelectionState, analysis: RestoreAnalysis): RestoreSelectionState {
  return state.token === analysis.token ? state : initialSelection(analysis);
}

export function useRestoreSelection(analysis: RestoreAnalysis, capabilities: RestoreCapabilities): RestoreSelection {
  const [stored, setStored] = useState(() => initialSelection(analysis));
  const state = currentState(stored, analysis);

  const enabledIds = useMemo(() => enabledInChainOrder(state, capabilities.steps), [state, capabilities.steps]);

  function update(change: (current: RestoreSelectionState) => RestoreSelectionState): void {
    setStored((previous) => change(currentState(previous, analysis)));
  }

  return {
    presetId: state.presetId,
    customized: state.customized,
    enabledIds,
    isEnabled: (stepId) => enabledIds.includes(stepId),
    optionsOf: (stepId) => stepOptions(state, stepId),
    choosePreset: (presetId) => update(() => selectPreset(analysis, presetId)),
    toggleStep: (stepId, enabled) => update((current) => toggleStep(current, stepId, enabled)),
    setOption: (stepId, option, value) => update((current) => setStepOption(current, stepId, option, value)),
    eta: selectionEta(analysis.eta.perStep, enabledIds),
    isOverprocessing: isOverprocessing(state, enabledIds, capabilities.halftoneDenoiseLimit),
    requestOptions: restoreRequestOptions(state, enabledIds),
  };
}
