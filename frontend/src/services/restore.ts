import { apiGet, apiPostForm, apiPostJson } from "../lib/api";
import type { JobResponse } from "../lib/apiTypes";
import type {
  RecomposeFaceChoice,
  RecomposeResponse,
  RestoreAnalysis,
  RestoreArtifactName,
  RestoreCapabilities,
  RestoreGeometry,
  RestoreMaskResponse,
  RestoreOptions,
} from "../lib/restoreApiTypes";
import type { UploadOptions } from "../lib/uploadRequest";

const MASK_FILE_NAME = "damage_mask.png";

export type RestoreJobSource = { token: string } | { file: File };

export interface CreateRestoreJobParams {
  source: RestoreJobSource;
  steps: string[];
  options: RestoreOptions;
  scale: number;
  modelId: string | null;
  device: string | null;
  outputFormat: string;
}

export function getRestoreCapabilities(): Promise<RestoreCapabilities> {
  return apiGet<RestoreCapabilities>("/restore/capabilities");
}

export function analyzePhoto(file: File, options: UploadOptions = {}): Promise<RestoreAnalysis> {
  const formData = new FormData();
  formData.append("file", file);
  return apiPostForm<RestoreAnalysis>("/restore/analyze", formData, options);
}

export function setPhotoGeometry(token: string, geometry: RestoreGeometry): Promise<RestoreAnalysis> {
  return apiPostJson<RestoreAnalysis>(`/restore/analysis/${encodeURIComponent(token)}/geometry`, geometry);
}

export function uploadDamageMask(token: string, mask: Blob): Promise<RestoreMaskResponse> {
  const formData = new FormData();
  formData.append("file", mask, MASK_FILE_NAME);
  return apiPostForm<RestoreMaskResponse>(`/restore/analysis/${encodeURIComponent(token)}/mask`, formData);
}

function appendSource(formData: FormData, source: RestoreJobSource): void {
  if ("token" in source) {
    formData.append("token", source.token);
    return;
  }
  formData.append("file", source.file);
}

// Sin modelo el backend usa su SR por defecto; mandar un id vacio lo
// confundiria con un modelo inexistente.
function appendModel(formData: FormData, modelId: string | null): void {
  if (!modelId) {
    return;
  }
  formData.append("model_name", modelId);
  formData.append("model_id", modelId);
}

function appendOptions(formData: FormData, options: RestoreOptions): void {
  if (Object.keys(options).length > 0) {
    formData.append("restore_options", JSON.stringify(options));
  }
}

export function buildRestoreJobFormData(params: CreateRestoreJobParams): FormData {
  const formData = new FormData();
  appendSource(formData, params.source);
  formData.append("restore_steps", params.steps.join(","));
  appendOptions(formData, params.options);
  formData.append("scale", String(params.scale));
  appendModel(formData, params.modelId);
  if (params.device) {
    formData.append("device", params.device);
  }
  formData.append("output_format", params.outputFormat);
  return formData;
}

export function createRestoreJob(
  params: CreateRestoreJobParams,
  options: UploadOptions = {},
): Promise<JobResponse> {
  return apiPostForm<JobResponse>("/restore/jobs", buildRestoreJobFormData(params), options);
}

export function recomposeFaces(
  jobId: string,
  faces: Record<number, RecomposeFaceChoice>,
): Promise<RecomposeResponse> {
  return apiPostJson<RecomposeResponse>(`/restore/jobs/${encodeURIComponent(jobId)}/recompose`, { faces });
}

export function restoreArtifactUrl(jobId: string, name: RestoreArtifactName): string {
  return `/api/v1/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeURIComponent(name)}`;
}
