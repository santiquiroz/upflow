import { apiGet, apiPost, apiPostForm, apiPostJson } from "../lib/api";
import type {
  ConversionStatusResponse,
  CreateInstallResponse,
  GenerationCapabilities,
  GenerationJob,
  InitImageResponse,
  InstallStatusResponse,
  ModelSearchResponse,
  Precision,
  PreflightResponse,
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
  initImageToken?: string;
  strength?: number;
  // Máscara de inpainting (blanco=editar, negro=conservar); requiere initImageToken.
  maskImageToken?: string;
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
  if (params.initImageToken) {
    body.initImageToken = params.initImageToken;
    if (params.strength !== undefined) body.strength = params.strength;
    if (params.maskImageToken) body.maskImageToken = params.maskImageToken;
  }
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

function buildInitImageFormData(file: File): FormData {
  const formData = new FormData();
  formData.append("file", file);
  return formData;
}

export function uploadGenerationInitImage(file: File): Promise<InitImageResponse> {
  return apiPostForm<InitImageResponse>("/generation/init-image", buildInitImageFormData(file));
}

export function getGenerationJob(jobId: string): Promise<GenerationJob> {
  return apiGet<GenerationJob>(`/generation/jobs/${jobId}`);
}

export function cancelGenerationJob(jobId: string): Promise<GenerationJob> {
  return apiPost<GenerationJob>(`/generation/jobs/${jobId}/cancel`);
}

export function listGenerationJobs(all: boolean): Promise<{ jobs: GenerationJob[] }> {
  return apiGet<{ jobs: GenerationJob[] }>(`/generation/jobs?all=${all}`);
}

export function fetchGenerationCapabilities(): Promise<GenerationCapabilities> {
  return apiGet<GenerationCapabilities>("/generation/capabilities");
}

export function searchGenerationModels(query: string): Promise<ModelSearchResponse> {
  return apiGet<ModelSearchResponse>(`/generation/models/search?q=${encodeURIComponent(query)}`);
}

export function preflightGenerationModel(
  repoId: string,
  width = 512,
  height = 512,
): Promise<PreflightResponse> {
  const params = new URLSearchParams({ repoId, width: String(width), height: String(height) });
  return apiGet<PreflightResponse>(`/generation/models/preflight?${params}`);
}

export function installGenerationModel(
  repoId: string,
  precision?: Precision,
  checkpointPath?: string,
): Promise<CreateInstallResponse> {
  const body: Record<string, unknown> = { repoId };
  if (precision) body.precision = precision;
  if (checkpointPath) body.checkpointPath = checkpointPath;
  return apiPostJson<CreateInstallResponse>("/generation/models", body);
}

// Vía rápida: el lane Vulkan corre el checkpoint tal cual, así que instalar es
// descargar. Sin exportación a ONNX y sin los ~40 min que eso cuesta.
export function installVulkanModel(
  repoId: string,
  filename: string,
): Promise<CreateInstallResponse> {
  return apiPostJson<CreateInstallResponse>("/generation/models/vulkan", { repoId, filename });
}

export interface VulkanInstallStatus {
  installId: string;
  repoId: string;
  status: "queued" | "downloading" | "installed" | "error";
  progressPct: number | null;
  modelId: string | null;
  error: string | null;
}

export function getVulkanInstallStatus(installId: string): Promise<VulkanInstallStatus> {
  return apiGet<VulkanInstallStatus>(`/generation/models/vulkan/${installId}`);
}

export function getGenerationInstallStatus(installId: string): Promise<InstallStatusResponse> {
  return apiGet<InstallStatusResponse>(`/generation/models/install/${installId}`);
}

export function convertGenerationModel(
  repoId: string,
): Promise<{ conversionId: string; statusUrl: string }> {
  return apiPostJson("/generation/models/convert", { repoId });
}

export function getConversionStatus(conversionId: string): Promise<ConversionStatusResponse> {
  return apiGet<ConversionStatusResponse>(`/generation/models/convert/${conversionId}`);
}
