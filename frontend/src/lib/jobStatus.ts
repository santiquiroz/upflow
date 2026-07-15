import type { JobStatus } from "./apiTypes";

const TERMINAL_JOB_STATUSES: readonly JobStatus[] = ["completed", "failed"];

export function isTerminalJobStatus(status: JobStatus): boolean {
  return TERMINAL_JOB_STATUSES.includes(status);
}
