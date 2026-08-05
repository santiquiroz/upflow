import { apiGet, apiPost, apiPostForm, apiPostJson } from "../lib/api";
import type { UploadOptions } from "../lib/uploadRequest";
import type {
  AsrInstallStatusResponse,
  CreateInstallResponse,
  CreateJobResponse,
  DevicesResponse,
  ModelResponse,
  ModelsResponse,
  ModelSearchResponse,
  TranscribeJob,
} from "../lib/apiTypes";

// "text" deja solo la transcripcion; "video" suma la pista de subtitulos al
// contenedor sin re-encodear; "video_burned" la pinta en la imagen.
export type TranscribeOutputMode = "text" | "video" | "video_burned";

export interface CreateTranscribeJobParams {
  file: File;
  modelId: string;
  language?: string;
  device?: string;
  outputMode?: TranscribeOutputMode;
}

function buildTranscribeJobFormData(params: CreateTranscribeJobParams): FormData {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("model_id", params.modelId);
  if (params.language) {
    formData.append("language", params.language);
  }
  if (params.device) {
    formData.append("device", params.device);
  }
  if (params.outputMode) {
    formData.append("output_mode", params.outputMode);
  }
  return formData;
}

export function createTranscribeJob(
  params: CreateTranscribeJobParams,
  options: UploadOptions = {},
): Promise<CreateJobResponse> {
  return apiPostForm<CreateJobResponse>(
    "/transcribe/jobs",
    buildTranscribeJobFormData(params),
    options,
  );
}

export function getTranscribeJob(jobId: string): Promise<TranscribeJob> {
  return apiGet<TranscribeJob>(`/transcribe/jobs/${jobId}`);
}

export function cancelTranscribeJob(jobId: string): Promise<TranscribeJob> {
  return apiPost<TranscribeJob>(`/transcribe/jobs/${jobId}/cancel`);
}

export function searchAsrModels(query: string): Promise<ModelSearchResponse> {
  return apiGet<ModelSearchResponse>(
    `/asr/models/search?q=${encodeURIComponent(query)}`,
  );
}

export function installAsrModel(repoId: string): Promise<CreateInstallResponse> {
  return apiPostJson<CreateInstallResponse>("/asr/models/install", { repoId });
}

export function getAsrInstallStatus(
  installId: string,
): Promise<AsrInstallStatusResponse> {
  return apiGet<AsrInstallStatusResponse>(
    `/asr/models/install/${installId}`,
  );
}

export async function fetchInstalledAsrModels(): Promise<ModelResponse[]> {
  const response = await apiGet<ModelsResponse>("/models");
  return response.models.filter((model) => model.kind === "asr-onnx");
}

export function fetchTranscribeDevices(): Promise<DevicesResponse> {
  return apiGet<DevicesResponse>("/devices");
}
