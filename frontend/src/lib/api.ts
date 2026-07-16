import type {
  CreateInstallResponse,
  CreateJobResponse,
  DevicesResponse,
  EngineInfoResponse,
  HealthResponse,
  InstallStatusResponse,
  JobResponse,
  ModelSearchResponse,
  ModelsResponse,
  VideoJobResponse,
} from "./apiTypes";

const API_BASE = "/api/v1";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function extractErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    if (typeof body.detail === "string" && body.detail.length > 0) {
      return body.detail;
    }
  } catch {
    // Body was not JSON (or empty) — fall through to statusText below.
  }
  return response.statusText || `Request failed with status ${response.status}`;
}

export async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { method: "GET" });
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }
  return (await response.json()) as T;
}

export async function apiPostForm<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { method: "POST", body: formData });
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }
  return (await response.json()) as T;
}

async function apiPostJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }
  return (await response.json()) as T;
}

async function apiDelete(path: string): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }
}

export function getHealth(): Promise<HealthResponse> {
  return apiGet<HealthResponse>("/health");
}

export function getEngineInfo(): Promise<EngineInfoResponse> {
  return apiGet<EngineInfoResponse>("/engine");
}

export function getDevices(): Promise<DevicesResponse> {
  return apiGet<DevicesResponse>("/devices");
}

export function getModels(): Promise<ModelsResponse> {
  return apiGet<ModelsResponse>("/models");
}

export interface CreateImageJobParams {
  file: File;
  modelId: string;
  device: string | null;
  scale: number;
  outputFormat: string;
}

function buildImageJobFormData(params: CreateImageJobParams): FormData {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("model_name", params.modelId);
  formData.append("model_id", params.modelId);
  formData.append("scale", String(params.scale));
  formData.append("output_format", params.outputFormat);
  if (params.device) {
    formData.append("device", params.device);
  }
  return formData;
}

export function createImageJob(params: CreateImageJobParams): Promise<CreateJobResponse> {
  return apiPostForm<CreateJobResponse>("/jobs", buildImageJobFormData(params));
}

export function getJob(jobId: string): Promise<JobResponse> {
  return apiGet<JobResponse>(`/jobs/${jobId}`);
}

export interface CreateVideoJobParams {
  file: File;
  profileKey: string;
  modelId: string | null;
  device: string | null;
  scale: number;
  outputContainer: string;
  videoCodec: string;
  videoPreset: string;
  crf: number;
  keepAudio: boolean;
  fpsMultiplier: number;
  targetFps: string | null;
  audioEnhance: string | null;
  audioRestore: string | null;
}

function appendVideoModelFields(formData: FormData, modelId: string | null): void {
  if (!modelId) {
    return;
  }
  formData.append("model_name", modelId);
  formData.append("model_id", modelId);
}

function buildVideoJobFormData(params: CreateVideoJobParams): FormData {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("profile_key", params.profileKey);
  formData.append("scale", String(params.scale));
  formData.append("output_container", params.outputContainer);
  formData.append("video_codec", params.videoCodec);
  formData.append("video_preset", params.videoPreset);
  formData.append("crf", String(params.crf));
  formData.append("keep_audio", String(params.keepAudio));
  formData.append("fps_multiplier", String(params.fpsMultiplier));
  appendVideoModelFields(formData, params.modelId);
  if (params.device) {
    formData.append("device", params.device);
  }
  if (params.targetFps) {
    formData.append("target_fps", params.targetFps);
  }
  if (params.audioEnhance) {
    formData.append("audio_enhance", params.audioEnhance);
  }
  if (params.audioRestore) {
    formData.append("audio_restore", params.audioRestore);
  }
  return formData;
}

export function createVideoJob(params: CreateVideoJobParams): Promise<CreateJobResponse> {
  return apiPostForm<CreateJobResponse>("/video/jobs", buildVideoJobFormData(params));
}

export function getVideoJob(jobId: string): Promise<VideoJobResponse> {
  return apiGet<VideoJobResponse>(`/video/jobs/${jobId}`);
}

export function searchHfModels(query: string): Promise<ModelSearchResponse> {
  return apiGet<ModelSearchResponse>(`/models/search?q=${encodeURIComponent(query)}`);
}

export function installModel(repoId: string): Promise<CreateInstallResponse> {
  return apiPostJson<CreateInstallResponse>("/models/install", { repoId });
}

export function getInstallStatus(installId: string): Promise<InstallStatusResponse> {
  return apiGet<InstallStatusResponse>(`/models/install/${installId}`);
}

export function deleteModel(modelId: string): Promise<void> {
  return apiDelete(`/models/${modelId}`);
}
