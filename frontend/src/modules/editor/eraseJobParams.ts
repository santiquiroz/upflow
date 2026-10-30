import type { GenerationModelSummary } from "../../lib/apiTypes";
import type { CreateGenerationJobParams } from "../../services/generation";

export function isReplaceModel(model: GenerationModelSummary): boolean {
  return model.status === "installed" &&
    model.supportsInpaint !== false &&
    !model.eraseOnly &&
    model.speed !== "turbo";
}

export function canReplace(models: GenerationModelSummary[]): boolean {
  return models.some(isReplaceModel);
}

export function defaultEraseModel(models: GenerationModelSummary[]): string | null {
  return models.some((model) => model.id === "eraser-auto" && model.status === "installed")
    ? "eraser-auto"
    : null;
}

export function shouldUseAutoErase(
  mode: "erase" | "replace",
  modelId: string | null,
  models: GenerationModelSummary[],
): boolean {
  if (mode !== "erase") return false;
  const model = models.find((candidate) => candidate.id === modelId);
  if (model?.status !== "installed") return false;
  if (!model.inpaintOnly || model.eraseOnly) return false;
  return models.some((candidate) => candidate.id === "eraser-lama" && candidate.status === "installed");
}

export function buildEditorJobParams(
  params: CreateGenerationJobParams,
  mode: "erase" | "replace",
  models: GenerationModelSummary[],
): CreateGenerationJobParams {
  const { eraseStrategy: _eraseStrategy, ...jobParams } = params;
  if (!params.maskImageToken) return jobParams;
  if (params.modelId === "eraser-auto") return jobParams;
  if (!shouldUseAutoErase(mode, params.modelId, models)) return jobParams;
  return { ...jobParams, eraseStrategy: "auto" };
}
