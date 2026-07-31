import type { MediaProbe } from "../../lib/apiTypes";

// La lógica de decisión de la pantalla de descargas, separada del componente para
// poder probarla sin renderizar nada.

export const HEIGHT_OPTIONS: readonly number[] = [360, 480, 720, 1080, 1440, 2160];

/** 1080p y no 4K: el pedido caro tiene que ser una elección, no un descuido. */
export const DEFAULT_HEIGHT = 1080;

/** El backend rechaza por encima de esto; la UI no debería dejar llegar. */
export const MAX_PLAYLIST_ITEMS = 50;

/** A partir de acá una playlist merece una confirmación explícita. */
const PLAYLIST_WARNING_THRESHOLD = 10;

export interface PlaylistNotice {
  kind: "playlist";
  entryCount: number;
  /** Cuántos se van a bajar realmente con el límite actual. */
  willDownload: number;
  needsConfirmation: boolean;
}

export function isProbablyUrl(value: string): boolean {
  const trimmed = value.trim();
  return /^https?:\/\/\S+\.\S+/i.test(trimmed);
}

export function clampPlaylistLimit(limit: number): number {
  if (!Number.isFinite(limit) || limit < 1) {
    return 1;
  }
  return Math.min(Math.floor(limit), MAX_PLAYLIST_ITEMS);
}

/**
 * Qué avisar sobre una playlist.
 *
 * Es la queja más repetida de los descargadores: pegar un link y que arranquen 200
 * descargas. El aviso existe para que eso sea imposible por accidente.
 */
export function playlistNotice(
  probe: MediaProbe | null,
  includePlaylist: boolean,
  playlistLimit: number,
): PlaylistNotice | null {
  if (!probe?.isPlaylist) {
    return null;
  }
  const willDownload = includePlaylist
    ? Math.min(probe.entryCount, clampPlaylistLimit(playlistLimit))
    : 1;
  return {
    kind: "playlist",
    entryCount: probe.entryCount,
    willDownload,
    needsConfirmation: includePlaylist && willDownload > PLAYLIST_WARNING_THRESHOLD,
  };
}

/**
 * Las alturas que tiene sentido ofrecer para ESTA URL.
 *
 * Ofrecer 4K sobre un video que sólo existe en 720p produce un pedido que no se puede
 * cumplir y una expectativa que no se cumple.
 */
export function offeredHeights(probe: MediaProbe | null): readonly number[] {
  if (!probe || probe.availableHeights.length === 0) {
    return HEIGHT_OPTIONS;
  }
  const best = Math.max(...probe.availableHeights);
  const offered = HEIGHT_OPTIONS.filter((height) => height <= best);
  // Si el video es más chico que la opción más baja, igual hay que poder pedirlo.
  return offered.length > 0 ? offered : [HEIGHT_OPTIONS[0]];
}

export const AUDIO_FORMAT_OPTIONS = ["mp3", "m4a", "opus", "flac", "wav"] as const;
export type AudioFormat = (typeof AUDIO_FORMAT_OPTIONS)[number];

/** null = mejor calidad variable (VBR). */
export const AUDIO_BITRATE_OPTIONS: readonly (number | null)[] = [null, 320, 256, 192, 128];

export const VIDEO_CONTAINER_OPTIONS = ["mp4", "mkv"] as const;
export type VideoContainer = (typeof VIDEO_CONTAINER_OPTIONS)[number];

const LOSSLESS_FORMATS: readonly AudioFormat[] = ["flac", "wav"];

/** FLAC/WAV no tienen bitrate elegible: ofrecerlo sería un knob mudo. */
export function bitrateSelectable(format: AudioFormat): boolean {
  return !LOSSLESS_FORMATS.includes(format);
}

/**
 * El link para bajar el archivo producido, servido por el backend.
 *
 * Sin esto un usuario remoto ve el nombre del archivo y una ruta de otra máquina:
 * la descarga termina y el resultado queda inalcanzable.
 */
export function downloadFileHref(jobId: string, index: number): string {
  return `/api/v1/download/jobs/${jobId}/download?index=${index}`;
}

export function formatBytes(bytes: number | null): string {
  if (bytes === null || bytes <= 0) {
    return "—";
  }
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || seconds <= 0) {
    return "—";
  }
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return hours > 0 ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${minutes}:${pad(secs)}`;
}
