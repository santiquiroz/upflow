import type { JobResponse } from "../../../lib/apiTypes";
import { isTerminalJobStatus } from "../../../lib/jobStatus";
import { readRestoreSummary, type RestoreResultSummary } from "./restoreResultModel";

export type BatchRowState =
  | { kind: "waiting"; job: JobResponse | undefined }
  | { kind: "ended"; job: JobResponse }
  | { kind: "done"; job: JobResponse; summary: RestoreResultSummary };

export function batchRowState(job: JobResponse | undefined): BatchRowState {
  if (job === undefined || !isTerminalJobStatus(job.status)) {
    return { kind: "waiting", job };
  }
  const summary = readRestoreSummary(job);
  return summary === null ? { kind: "ended", job } : { kind: "done", job, summary };
}

export function ranFaces(job: JobResponse): boolean {
  return job.restoreSteps?.includes("faces") ?? false;
}

export function hasRestoredFaces(summary: RestoreResultSummary): boolean {
  return summary.faces.length > 0;
}

export function enabledFaceCount(summary: RestoreResultSummary): number {
  return summary.faces.filter((face) => face.enabled).length;
}

export function faceCountKey(summary: RestoreResultSummary): string {
  const enabled = enabledFaceCount(summary);
  if (!hasRestoredFaces(summary)) {
    return "restore.batch.results.noFaces";
  }
  if (enabled === 0) {
    return "restore.batch.results.facesOff";
  }
  return enabled === 1 ? "restore.batch.results.facesRestoredOne" : "restore.batch.results.facesRestored";
}

export function needsFaceReview(state: BatchRowState, reviewed: boolean): boolean {
  return state.kind === "done" && hasRestoredFaces(state.summary) && !reviewed;
}

export function countToReview(states: readonly BatchRowState[], reviewed: ReadonlySet<string>): number {
  return states.filter((state) => state.job !== undefined && needsFaceReview(state, reviewed.has(state.job.jobId)))
    .length;
}
