import type { CctvAiUpscaleModel, CctvStepSchema } from "../../../services/cctv";
import { defaultChoice, type StepChoices } from "./cctvSteps";

export interface AiUpscaleChoice {
  modelId: string;
  scale: number;
}

export interface AiUpscaleFields {
  modelId?: string;
  scale?: number;
}

export const AI_UPSCALE_STEP_ID = "ai_upscale";

export function generativeTagKey(generative: boolean): string {
  return generative ? "model.tag.generative" : "model.tag.nonGenerative";
}

export function scaleFor(model: CctvAiUpscaleModel, preferred: number | null): number {
  return preferred !== null && model.scales.includes(preferred) ? preferred : model.scales[0];
}

// "None" es el default: sin modelo no hay paso de reescalado IA.
export function aiUpscaleFor(
  models: readonly CctvAiUpscaleModel[],
  modelId: string | null,
  preferredScale: number | null,
): AiUpscaleChoice | null {
  const model = models.find((candidate) => candidate.id === modelId);
  return model ? { modelId: model.id, scale: scaleFor(model, preferredScale) } : null;
}

// El paso lo arma el selector de modelo y va en el orden del catalogo como cualquier otro.
export function withAiUpscaleStep(
  steps: StepChoices,
  catalog: readonly CctvStepSchema[],
  choice: AiUpscaleChoice | null,
): StepChoices {
  const step = catalog.find((candidate) => candidate.id === AI_UPSCALE_STEP_ID);
  const stepChoice = choice && step ? defaultChoice(step) : null;
  return stepChoice ? { ...steps, [AI_UPSCALE_STEP_ID]: stepChoice } : steps;
}

export function aiUpscaleFields(choice: AiUpscaleChoice | null): AiUpscaleFields {
  return choice ? { modelId: choice.modelId, scale: choice.scale } : {};
}
