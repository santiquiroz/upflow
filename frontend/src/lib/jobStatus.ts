import type { JobStatus } from "./apiTypes";

const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ["completed", "failed"];

export function isTerminalJobStatus(status: JobStatus): boolean {
  return TERMINAL_JOB_STATUSES.includes(status);
}

export type JobKind = "image" | "video" | "audio";

const JOB_KIND_LABELS: Record<JobKind, string> = {
  image: "Image",
  video: "Video",
  audio: "Audio",
};

export function jobKindLabel(kind: JobKind): string {
  return JOB_KIND_LABELS[kind];
}
