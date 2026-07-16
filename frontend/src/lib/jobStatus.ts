import type { JobStatus } from "./apiTypes";

const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ["completed", "failed"];

export function isTerminalJobStatus(status: JobStatus): boolean {
  return TERMINAL_JOB_STATUSES.includes(status);
}

export type JobKind = "image" | "video";

export function jobKindLabel(kind: JobKind): string {
  return kind === "image" ? "Image" : "Video";
}
