import type { JobStatus } from "./apiTypes";
import type { JobKind } from "./jobTypeGuards";

const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ["completed", "failed", "cancelled"];

export function isTerminalJobStatus(status: JobStatus): boolean {
  return TERMINAL_JOB_STATUSES.includes(status);
}

const CANCELLABLE_JOB_STATUSES: readonly JobStatus[] = ["queued", "running"];

export function isCancellableJobStatus(status: JobStatus): boolean {
  return CANCELLABLE_JOB_STATUSES.includes(status);
}

export type { JobKind };

// Devuelve la CLAVE y no el texto: el nombre de la familia se lee en el idioma
// de la pantalla, igual que el resto de la copia.
export function jobKindLabelKey(kind: JobKind): string {
  return `job.kind.${kind}`;
}

const JOB_STATUS_LABEL_KEYS: Record<JobStatus, string> = {
  queued: "job.status.queued",
  running: "job.status.processing",
  completed: "job.status.completed",
  failed: "job.status.failed",
  cancelled: "job.status.cancelled",
};

export function jobStatusLabelKey(status: JobStatus): string {
  return JOB_STATUS_LABEL_KEYS[status];
}
