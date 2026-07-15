import type {
  DevicesResponse,
  EngineInfoResponse,
  HealthResponse,
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
