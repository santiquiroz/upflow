import type {
  AnalyzeVideoResponse,
  CreateInstallResponse,
  CreateJobResponse,
  DevicesResponse,
  EngineInfoResponse,
  HealthResponse,
  InstallStatusResponse,
  JobResponse,
  ModelSearchResponse,
  ModelsResponse,
  UpscaleBackend,
  VideoCapabilities,
  VideoEncoder,
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

export async function apiPost<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { method: "POST" });
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

export function cancelJob(jobId: string): Promise<JobResponse> {
  return apiPost<JobResponse>(`/jobs/${jobId}/cancel`);
}

export interface CreateVideoJobParams {
  // Exactly one of `file` / `uploadToken` must be set: `file` re-uploads the
  // raw video, `uploadToken` reuses a prior POST /video/analyze upload — the
  // backend rejects a request carrying both.
  file?: File;
  uploadToken?: string;
  // Original filename for display (job queue, accessibility labels). Required
  // when submitting via `uploadToken`, since `file` itself isn't sent in that
  // case and callers still know the name of the file they analyzed.
  fileName?: string;
  audioTrackIndices?: number[];
  keepSubtitles?: boolean;
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
  interpEngine: string;
  backend: UpscaleBackend;
  videoEncoder: VideoEncoder;
}

function appendVideoModelFields(formData: FormData, modelId: string | null): void {
  if (!modelId) {
    return;
  }
  formData.append("model_name", modelId);
  formData.append("model_id", modelId);
}

function appendVideoSourceFields(formData: FormData, params: CreateVideoJobParams): void {
  if (params.uploadToken) {
    formData.append("upload_token", params.uploadToken);
    return;
  }
  if (params.file) {
    formData.append("file", params.file);
  }
}

function buildVideoJobFormData(params: CreateVideoJobParams): FormData {
  const formData = new FormData();
  appendVideoSourceFields(formData, params);
  formData.append("profile_key", params.profileKey);
  formData.append("scale", String(params.scale));
  formData.append("output_container", params.outputContainer);
  formData.append("video_codec", params.videoCodec);
  formData.append("video_preset", params.videoPreset);
  formData.append("crf", String(params.crf));
  formData.append("keep_audio", String(params.keepAudio));
  formData.append("fps_multiplier", String(params.fpsMultiplier));
  formData.append("interp_engine", params.interpEngine);
  formData.append("backend", params.backend);
  formData.append("video_encoder", params.videoEncoder);
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
  if (params.audioTrackIndices && params.audioTrackIndices.length > 0) {
    formData.append("audio_track_indices", params.audioTrackIndices.join(","));
  }
  if (params.keepSubtitles) {
    formData.append("keep_subtitles", "true");
  }
  return formData;
}

export function createVideoJob(params: CreateVideoJobParams): Promise<CreateJobResponse> {
  return apiPostForm<CreateJobResponse>("/video/jobs", buildVideoJobFormData(params));
}

export function analyzeVideo(file: File): Promise<AnalyzeVideoResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return apiPostForm<AnalyzeVideoResponse>("/video/analyze", formData);
}

export function getVideoJob(jobId: string): Promise<VideoJobResponse> {
  return apiGet<VideoJobResponse>(`/video/jobs/${jobId}`);
}

export function cancelVideoJob(jobId: string): Promise<VideoJobResponse> {
  return apiPost<VideoJobResponse>(`/video/jobs/${jobId}/cancel`);
}

export function getVideoCapabilities(): Promise<VideoCapabilities> {
  return apiGet<VideoCapabilities>("/video/capabilities");
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
