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
  /** Modelos EXTRA a combinar con el principal. Vacío = un solo modelo. */
  ensembleModels?: string[];
  /** Stems a quitar en pistas minus-one. Vacío = no se piden. */
  practiceStems?: string[];
  /** Volumen de guía del instrumento quitado (0-30). 0 = sin guía. */
  practiceGuidePercent?: number;
  /** Stems CON altura a transcribir (MIDI+MusicXML, +tab si guitar/bass). Vacío = no se piden. */
  transcribeStems?: string[];
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
    // Vacío NO se manda: el campo ausente significa "un solo modelo", misma
    // regla que las demás listas de esta API.
    if (params.ensembleModels && params.ensembleModels.length > 0) {
      formData.append("ensemble_models", params.ensembleModels.join(","));
    }
    appendPracticeFields(formData, params);
    appendTranscribeFields(formData, params);
  }
  // Misma regla que voice_steps: una selección vacía NO se manda, porque el
  // campo ausente significa "sin cadena de limpieza".
  if (params.cleanupSteps && params.cleanupSteps.length > 0) {
    formData.append("cleanup_steps", params.cleanupSteps.join(","));
  }
  appendVoiceFields(formData, params);
  return formData;
}

function appendPracticeFields(formData: FormData, params: CreateAudioJobParams): void {
  // Misma regla CSV que las demás listas: la selección vacía NO se manda. Y la
  // guía solo viaja acompañando stems — sin stems no hay minus-one que hornear,
  // así que una guía suelta sería un pedido sin sujeto.
  if (!params.practiceStems || params.practiceStems.length === 0) {
    return;
  }
  formData.append("practice_stems", params.practiceStems.join(","));
  // 0 es el default del backend: mandarlo solo agrega ruido al form.
  if (params.practiceGuidePercent) {
    formData.append("practice_guide_percent", String(params.practiceGuidePercent));
  }
}

function appendTranscribeFields(formData: FormData, params: CreateAudioJobParams): void {
  // Misma regla CSV que practice_stems: la selección vacía NO se manda, porque
  // el campo ausente significa "sin transcripción pedida".
  if (!params.transcribeStems || params.transcribeStems.length === 0) {
    return;
  }
  formData.append("transcribe_stems", params.transcribeStems.join(","));
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
