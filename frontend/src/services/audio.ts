import { apiGet, apiPost, apiPostForm } from "../lib/api";
import type { UploadOptions } from "../lib/uploadRequest";
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
  /** Escalón de calidad para mp3/m4a. Ausente = el default del backend. */
  lossyQuality?: string | null;
  device: string | null;
  voiceSteps?: string[];
  voiceDelivery?: string | null;
  voicePresenceDb?: number | null;
  master?: string | null;
  /**
   * Cadena de limpieza: ids del catálogo. El orden que se mande da igual — lo
   * fija el catálogo del backend, igual que en la cadena de voz.
   */
  cleanupSteps?: string[];
  /** Modo karaoke: exclusivo, el backend rechaza combinarlo con otros pasos. */
  separate?: boolean;
  separationModel?: string | null;
}

function buildAudioJobFormData(params: CreateAudioJobParams): FormData {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("output_format", params.outputFormat);
  if (params.lossyQuality) {
    formData.append("lossy_quality", params.lossyQuality);
  }
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
  if (params.separate) {
    formData.append("separate", "true");
    if (params.separationModel) {
      formData.append("separation_model", params.separationModel);
    }
  }
  // Misma regla que voice_steps: una selección vacía NO se manda, porque el
  // campo ausente significa "sin cadena de limpieza".
  if (params.cleanupSteps && params.cleanupSteps.length > 0) {
    formData.append("cleanup_steps", params.cleanupSteps.join(","));
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

export function createAudioJob(
  params: CreateAudioJobParams,
  options: UploadOptions = {},
): Promise<CreateJobResponse> {
  return apiPostForm<CreateJobResponse>("/audio/jobs", buildAudioJobFormData(params), options);
}

export interface CompareModelsParams {
  file: File;
  models: string[];
  excerptSeconds?: number;
  offsetSeconds?: number | null;
}

export interface ComparisonEntry {
  modelId: string;
  jobId: string;
}

export interface AudioComparison {
  entries: ComparisonEntry[];
  offsetSeconds: number;
  excerptSeconds: number;
}

/** Corre varios separadores sobre el mismo fragmento del archivo del usuario. */
export function compareSeparationModels(
  params: CompareModelsParams,
  options: UploadOptions = {},
): Promise<AudioComparison> {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("models", params.models.join(","));
  if (params.excerptSeconds !== undefined) {
    formData.append("excerpt_seconds", String(params.excerptSeconds));
  }
  if (params.offsetSeconds !== undefined && params.offsetSeconds !== null) {
    formData.append("offset_seconds", String(params.offsetSeconds));
  }
  return apiPostForm<AudioComparison>("/audio/compare", formData, options);
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
