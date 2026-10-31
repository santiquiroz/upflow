import type { GenerationModelSummary } from "../../lib/apiTypes";
import type { CreateGenerationJobParams } from "../../services/generation";
import type { EditMode } from "./InpaintModelSelect";

export function isReplaceModel(model: GenerationModelSummary): boolean {
  return model.status === "installed" &&
    model.supportsInpaint !== false &&
    !model.eraseOnly &&
    model.speed !== "turbo";
}

export function canReplace(models: GenerationModelSummary[]): boolean {
  return models.some(isReplaceModel);
}

export function effectiveMode(mode: EditMode, models: GenerationModelSummary[], loaded: boolean): EditMode {
  return mode === "replace" && loaded && !canReplace(models) ? "erase" : mode;
}

export function defaultEraseModel(models: GenerationModelSummary[]): string | null {
  return models.some((model) => model.id === "eraser-auto" && model.status === "installed")
    ? "eraser-auto"
    : null;
}

export function modelForMode(
  mode: EditMode,
  currentModelId: string | null,
  models: GenerationModelSummary[],
): string | null {
  const model = models.find((candidate) => candidate.id === currentModelId);
  if (mode === "replace") return model && isReplaceModel(model) ? model.id : null;
  if (model?.status === "installed" && model.supportsInpaint !== false && model.speed !== "turbo") {
    return model.id;
  }
  return defaultEraseModel(models);
}

export function shouldUseAutoErase(
  mode: EditMode,
  modelId: string | null,
  models: GenerationModelSummary[],
): boolean {
  if (mode !== "erase") return false;
  const model = models.find((candidate) => candidate.id === modelId);
  if (model?.status !== "installed") return false;
  if (!model.inpaintOnly || model.eraseOnly) return false;
  return models.some((candidate) => candidate.id === "eraser-lama" && candidate.status === "installed");
}

export function supportsEdgeAdjust(
  mode: EditMode,
  modelId: string | null,
  models: GenerationModelSummary[],
): boolean {
  return mode === "erase" && (modelId === "eraser-auto" || shouldUseAutoErase(mode, modelId, models));
}

export function replaceModelId(models: GenerationModelSummary[]): string | null {
  const candidates = models.filter(isReplaceModel);
  return (candidates.find((model) => model.inpaintOnly) ?? candidates[0])?.id ?? null;
}

export function hasEraserAuto(models: GenerationModelSummary[]): boolean {
  return defaultEraseModel(models) !== null;
}

export interface RedoJobInput {
  engine: "lama" | "osor" | "lama+osor" | "sdxl";
  prompt?: string;
  base: CreateGenerationJobParams;
  models: GenerationModelSummary[];
}

export function buildRedoJobParams({ engine, prompt, base, models }: RedoJobInput): CreateGenerationJobParams | null {
  const { eraseStrategy: _strategy, eraseEngine: _engine, ...params } = base;
  if (engine !== "sdxl") return { ...params, modelId: "eraser-auto", eraseEngine: engine };
  const modelId = replaceModelId(models);
  const text = prompt?.trim() ?? "";
  return modelId && text ? { ...params, modelId, prompt: text } : null;
}

export function buildEditorJobParams(
  params: CreateGenerationJobParams,
  mode: EditMode,
  models: GenerationModelSummary[],
): CreateGenerationJobParams {
  const { eraseStrategy: _eraseStrategy, ...jobParams } = params;
  if (!params.maskImageToken) return jobParams;
  if (params.modelId === "eraser-auto") return jobParams;
  if (!shouldUseAutoErase(mode, params.modelId, models)) return jobParams;
  return { ...jobParams, eraseStrategy: "auto" };
}
