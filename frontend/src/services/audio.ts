import { apiGet, apiPost, apiPostForm } from "../lib/api";
import type { AudioCapabilities, AudioJob, CreateJobResponse } from "../lib/apiTypes";

export interface CreateAudioJobParams {
  file: File;
  denoise: string | null;
  restore: string | null;
  device: string | null;
}

function buildAudioJobFormData(params: CreateAudioJobParams): FormData {
  const formData = new FormData();
  formData.append("file", params.file);
  if (params.denoise) {
    formData.append("denoise", params.denoise);
  }
  if (params.restore) {
    formData.append("restore", params.restore);
  }
  if (params.device) {
    formData.append("device", params.device);
  }
  return formData;
}

export function createAudioJob(params: CreateAudioJobParams): Promise<CreateJobResponse> {
  return apiPostForm<CreateJobResponse>("/audio/jobs", buildAudioJobFormData(params));
}

export function getAudioJob(jobId: string): Promise<AudioJob> {
  return apiGet<AudioJob>(`/audio/jobs/${jobId}`);
}

export function cancelAudioJob(jobId: string): Promise<AudioJob> {
  return apiPost<AudioJob>(`/audio/jobs/${jobId}/cancel`);
}

export function fetchAudioCapabilities(): Promise<AudioCapabilities> {
  return apiGet<AudioCapabilities>("/audio/capabilities");
}
