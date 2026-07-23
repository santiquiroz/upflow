import type { JobStatus } from "./apiTypes";

const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ["completed", "failed", "cancelled"];

export function isTerminalJobStatus(status: JobStatus): boolean {
  return TERMINAL_JOB_STATUSES.includes(status);
}

const CANCELLABLE_JOB_STATUSES: readonly JobStatus[] = ["queued", "running"];

export function isCancellableJobStatus(status: JobStatus): boolean {
  return CANCELLABLE_JOB_STATUSES.includes(status);
}

export type JobKind = "image" | "video" | "audio" | "generation";

const JOB_KIND_LABELS: Record<JobKind, string> = {
  image: "Image",
  video: "Video",
  audio: "Audio",
  generation: "Generation",
};

export function jobKindLabel(kind: JobKind): string {
  return JOB_KIND_LABELS[kind];
}
