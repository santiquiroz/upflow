import { apiGet, apiPostJson } from "../lib/api";
import type { DownloadJob, MediaProbe } from "../lib/apiTypes";

export interface DownloadRequest {
  url: string;
  maxHeight?: number;
  audioOnly?: boolean;
  audioFormat?: string;
  audioBitrateKbps?: number | null;
  videoContainer?: string;
  includePlaylist?: boolean;
  playlistLimit?: number;
  subtitleLanguages?: string[];
  thenSeparate?: boolean;
  thenSeparationModel?: string | null;
}

/** Qué hay en la URL, sin descargar. Es lo que deja ver una playlist de 200 antes de disparar 200. */
export function probeMedia(request: DownloadRequest): Promise<MediaProbe> {
  return apiPostJson<MediaProbe>("/download/probe", request);
}

export function createDownloadJob(request: DownloadRequest): Promise<DownloadJob> {
  return apiPostJson<DownloadJob>("/download/jobs", request);
}

export function getDownloadJob(jobId: string): Promise<DownloadJob> {
  return apiGet<DownloadJob>(`/download/jobs/${jobId}`);
}

export function cancelDownloadJob(jobId: string): Promise<DownloadJob> {
  return apiPostJson<DownloadJob>(`/download/jobs/${jobId}/cancel`, {});
}
