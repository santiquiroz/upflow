import type { GenerationModelSummary } from "../../lib/apiTypes";
import type { CreateGenerationJobParams } from "../../services/generation";

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
  if (!shouldUseAutoErase(mode, params.modelId, models)) return jobParams;
  return { ...jobParams, eraseStrategy: "auto" };
}
