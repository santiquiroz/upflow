import type {
  CreateJobResponse,
  DevicesResponse,
  EngineInfoResponse,
  HealthResponse,
  JobResponse,
  ModelsResponse,
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

async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { method: "GET" });
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }
  return (await response.json()) as T;
}

async function apiPostForm<T>(path: string, formData: FormData): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { method: "POST", body: formData });
  if (!response.ok) {
    throw new ApiError(response.status, await extractErrorMessage(response));
  }
  return (await response.json()) as T;
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
