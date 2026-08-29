import { apiGet, apiPost, apiPostForm, apiPutJson } from "../lib/api";
import type { UploadOptions } from "../lib/uploadRequest";
import type { CreateJobResponse } from "../lib/apiTypes";

// La fase del NEGOCIO: `status` solo cuenta la pasada actual por la cola.
export type KaraokePhase =
  | "preparing"
  | "review"
  | "rendering"
  | "completed"
  | "failed"
  | "cancelled";

export type KaraokeBackgroundKind = "source" | "image" | "video" | "generated";

export interface KaraokeLyricLine {
  index: number;
  start: number;
  end: number;
  text: string;
  translation: string;
  // null cuando el trabajo no pidio deteccion de cantantes.
  singer: string | null;
}

export interface KaraokeSinger {
  id: string;
  label: string;
}

export interface KaraokeJob {
  id: string;
  status: string;
  phase: KaraokePhase;
  originalFilename: string;
  asrModelId: string;
  separationModelId: string | null;
  cleanupSteps: string[];
  restoreMode: string | null;
  language: string | null;
  romanize: boolean;
  translateTo: string | null;
  device: string | null;
  backgroundKind: KaraokeBackgroundKind;
  progressPct: number | null;
  error: string | null;
  lines: KaraokeLyricLine[];
  // Vacio cuando el trabajo no pidio deteccion de cantantes.
  singers: KaraokeSinger[];
  instrumentalUrl: string | null;
  sourceHasPicture: boolean | null;
  downloadUrl: string | null;
  // Solo en renders completados que mutearon a un cantante: la mezcla de
  // practica (instrumental + las voces que quedaron) como audio descargable.
  practiceMixUrl: string | null;
}

export interface CreateKaraokeJobParams {
  file: File;
  asrModelId: string;
  separationModelId?: string | null;
  cleanupSteps?: string[];
  restoreMode?: string | null;
  language?: string;
  romanize?: boolean;
  translateTo?: string | null;
  device?: string;
  detectSingers?: boolean;
  singerCount?: number;
}

function buildCreateFormData(params: CreateKaraokeJobParams): FormData {
  const formData = new FormData();
  formData.append("file", params.file);
  formData.append("asr_model_id", params.asrModelId);
  if (params.separationModelId) {
    formData.append("separation_model_id", params.separationModelId);
  }
  for (const step of params.cleanupSteps ?? []) {
    formData.append("cleanup_steps", step);
  }
  if (params.restoreMode) {
    formData.append("restore_mode", params.restoreMode);
  }
  if (params.language) {
    formData.append("language", params.language);
  }
  if (params.romanize) {
    formData.append("romanize", "true");
  }
  if (params.translateTo) {
    formData.append("translate_to", params.translateTo);
  }
  if (params.device) {
    formData.append("device", params.device);
  }
  // singer_count sin detect_singers es invalido para el backend: los dos
  // campos viajan juntos o no viajan.
  if (params.detectSingers) {
    formData.append("detect_singers", "true");
    if (params.singerCount) {
      formData.append("singer_count", String(params.singerCount));
    }
  }
  return formData;
}

export function createKaraokeJob(
  params: CreateKaraokeJobParams,
  options: UploadOptions = {},
): Promise<CreateJobResponse> {
  return apiPostForm<CreateJobResponse>(
    "/karaoke/jobs",
    buildCreateFormData(params),
    options,
  );
}

export function getKaraokeJob(jobId: string): Promise<KaraokeJob> {
  return apiGet<KaraokeJob>(`/karaoke/jobs/${jobId}`);
}

export interface KaraokeLyricEdit {
  index: number;
  text?: string;
  translation?: string;
  singer?: string;
}

export function updateKaraokeLyrics(
  jobId: string,
  lines: KaraokeLyricEdit[],
  singers?: KaraokeSinger[],
): Promise<KaraokeJob> {
  // `singers` ausente = "no los toques"; una lista presente reemplaza los
  // labels completos. Nunca se manda vacia.
  const body: { lines: KaraokeLyricEdit[]; singers?: KaraokeSinger[] } = { lines };
  if (singers && singers.length > 0) {
    body.singers = singers;
  }
  return apiPutJson<KaraokeJob>(`/karaoke/jobs/${jobId}/lyrics`, body);
}

export interface RenderKaraokeParams {
  backgroundKind: KaraokeBackgroundKind;
  background?: File | null;
  subtitleSize: string;
  subtitlePosition: string;
  subtitleColor: string;
  subtitleHighlightColor: string;
  singerColors?: Record<string, string>;
  muteSinger?: string | null;
}

export function renderKaraokeJob(
  jobId: string,
  params: RenderKaraokeParams,
  options: UploadOptions = {},
): Promise<KaraokeJob> {
  const formData = new FormData();
  formData.append("background_kind", params.backgroundKind);
  formData.append("subtitle_size", params.subtitleSize);
  formData.append("subtitle_position", params.subtitlePosition);
  formData.append("subtitle_color", params.subtitleColor);
  formData.append("subtitle_highlight_color", params.subtitleHighlightColor);
  // Convencion de campo repetido de la familia (como cleanup_steps): una
  // entrada "id:hex" por cantante.
  for (const [singerId, color] of Object.entries(params.singerColors ?? {})) {
    formData.append("singer_colors", `${singerId}:${color}`);
  }
  if (params.muteSinger) {
    formData.append("mute_singer", params.muteSinger);
  }
  if (params.background) {
    formData.append("background", params.background);
  }
  return apiPostForm<KaraokeJob>(
    `/karaoke/jobs/${jobId}/render`,
    formData,
    options,
  );
}

export function cancelKaraokeJob(jobId: string): Promise<KaraokeJob> {
  return apiPost<KaraokeJob>(`/karaoke/jobs/${jobId}/cancel`);
}
