import { apiGet, apiPost, apiPostJson } from "../lib/api";
import type {
  CreateInstallResponse,
  GenerationCapabilities,
  GenerationJob,
  InstallStatusResponse,
} from "../lib/apiTypes";

export interface CreateGenerationJobParams {
  prompt: string;
  negativePrompt: string | null;
  modelId: string;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  seed: number | null;
  device: string | null;
  autoUpscale: boolean;
  upscaleModelName: string | null;
  upscaleScale: number | null;
  upscaleModelId: string | null;
}

function buildRequestBody(params: CreateGenerationJobParams): Record<string, unknown> {
  const body: Record<string, unknown> = {
    prompt: params.prompt,
    modelId: params.modelId,
    steps: params.steps,
    guidance: params.guidance,
    width: params.width,
    height: params.height,
    autoUpscale: params.autoUpscale,
  };
  if (params.negativePrompt) body.negativePrompt = params.negativePrompt;
  if (params.seed !== null) body.seed = params.seed;
  if (params.device) body.device = params.device;
  if (params.autoUpscale) {
    if (params.upscaleModelName) body.upscaleModelName = params.upscaleModelName;
    if (params.upscaleScale !== null) body.upscaleScale = params.upscaleScale;
    if (params.upscaleModelId) body.upscaleModelId = params.upscaleModelId;
  }
  return body;
}

// NOTA (contrato real, Task 9): POST /generation/jobs devuelve el GenerationJob
// completo (id, status, downloadUrl...) con 201 — NO el CreateJobResponse/202
// (jobId/statusUrl) de los otros kinds. Deviación intencional documentada; no
// "arreglarla" para parecerse a audio.
export function createGenerationJob(params: CreateGenerationJobParams): Promise<GenerationJob> {
  return apiPostJson<GenerationJob>("/generation/jobs", buildRequestBody(params));
}

export function getGenerationJob(jobId: string): Promise<GenerationJob> {
  return apiGet<GenerationJob>(`/generation/jobs/${jobId}`);
}

export function cancelGenerationJob(jobId: string): Promise<GenerationJob> {
  return apiPost<GenerationJob>(`/generation/jobs/${jobId}/cancel`);
}

export function fetchGenerationCapabilities(): Promise<GenerationCapabilities> {
  return apiGet<GenerationCapabilities>("/generation/capabilities");
}

export function installGenerationModel(repoId: string): Promise<CreateInstallResponse> {
  return apiPostJson<CreateInstallResponse>("/generation/models", { repoId });
}

export function getGenerationInstallStatus(installId: string): Promise<InstallStatusResponse> {
  return apiGet<InstallStatusResponse>(`/generation/models/install/${installId}`);
}
