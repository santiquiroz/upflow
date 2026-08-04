import { apiGet, apiPost, apiPostForm } from "../lib/api";
import type {
  AudioCapabilities,
  AudioJob,
  CreateJobResponse,
  VoiceCatalog,
} from "../lib/apiTypes";

export interface CreateAudioJobParams {
  file: File;
  denoise: string | null;
  restore: string | null;
  outputFormat: string;
  device: string | null;
  voiceSteps?: string[];
  voiceDelivery?: string | null;
  voicePresenceDb?: number | null;
  master?: string | null;
}

function buildAudioJobFormData(params: CreateAudioJobParams): FormData {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("output_format", params.outputFormat);
  if (params.denoise) {
    formData.append("denoise", params.denoise);
  }
  if (params.restore) {
    formData.append("restore", params.restore);
  }
  if (params.device) {
    formData.append("device", params.device);
  }
  if (params.master) {
    formData.append("master", params.master);
  }
  appendVoiceFields(formData, params);
  return formData;
}

function appendVoiceFields(formData: FormData, params: CreateAudioJobParams): void {
  // El backend parsea voice_steps como lista separada por comas. Una seleccion
  // vacia NO se manda: el campo ausente significa "sin mejora de voz", mientras
  // que un string vacio pediria una cadena de cero pasos.
  if (params.voiceSteps && params.voiceSteps.length > 0) {
    formData.append("voice_steps", params.voiceSteps.join(","));
  }
  if (params.voiceDelivery) {
    formData.append("voice_delivery", params.voiceDelivery);
  }
  if (typeof params.voicePresenceDb === "number") {
    formData.append("voice_presence_db", String(params.voicePresenceDb));
  }
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

export function listAudioJobs(all: boolean): Promise<{ jobs: AudioJob[] }> {
  return apiGet<{ jobs: AudioJob[] }>(`/audio/jobs?all=${all}`);
}

export function fetchAudioCapabilities(): Promise<AudioCapabilities> {
  return apiGet<AudioCapabilities>("/audio/capabilities");
}

export function fetchVoiceCatalog(): Promise<VoiceCatalog> {
  return apiGet<VoiceCatalog>("/audio/voice-catalog");
}
