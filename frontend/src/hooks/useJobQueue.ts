import { useQueries } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";
import { getJob, getVideoJob } from "../lib/api";
import type { JobResponse, JobStatus, VideoJobResponse } from "../lib/apiTypes";
import { isTerminalJobStatus } from "../lib/jobStatus";
import { jobQueueStore, type JobQueueStore, type TrackedJob } from "../lib/jobQueueStore";

export const DEFAULT_QUEUE_POLL_INTERVAL_MS = 1500;

export interface JobQueueEntry {
  id: string;
  kind: TrackedJob["kind"];
  fileName: string;
  createdAt: number;
  status: JobStatus;
  downloadUrl: string | null;
  errorMessage: string | null;
}

export interface UseJobQueueResult {
  entries: JobQueueEntry[];
  dismiss: (id: string) => void;
  clearCompleted: () => void;
}

function fetchTrackedJob(tracked: TrackedJob): Promise<JobResponse | VideoJobResponse> {
  return tracked.kind === "image" ? getJob(tracked.id) : getVideoJob(tracked.id);
}

function resolveEntryError(data: JobResponse | VideoJobResponse | undefined, queryError: unknown): string | null {
  if (queryError instanceof Error) {
    return queryError.message;
  }
  if (data?.status === "failed") {
    return data.error ?? "The job failed.";
  }
  return null;
}

function toQueueEntry(
  tracked: TrackedJob,
  data: JobResponse | VideoJobResponse | undefined,
  queryError: unknown,
): JobQueueEntry {
  return {
    id: tracked.id,
    kind: tracked.kind,
    fileName: tracked.fileName,
    createdAt: tracked.createdAt,
    status: data?.status ?? "queued",
    downloadUrl: data?.downloadUrl ?? null,
    errorMessage: resolveEntryError(data, queryError),
  };
}

function byNewestFirst(a: JobQueueEntry, b: JobQueueEntry): number {
  return b.createdAt - a.createdAt;
}

export function useJobQueue(
  store: JobQueueStore = jobQueueStore,
  pollIntervalMs: number = DEFAULT_QUEUE_POLL_INTERVAL_MS,
): UseJobQueueResult {
  const trackedJobs = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);

  const results = useQueries({
    queries: trackedJobs.map((tracked) => ({
      queryKey: [tracked.kind === "image" ? "job" : "videoJob", tracked.id],
      queryFn: () => fetchTrackedJob(tracked),
      refetchInterval: (query: { state: { data?: { status?: JobStatus } } }) =>
        isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : pollIntervalMs,
    })),
  });

  const entries = trackedJobs
    .map((tracked, index) => toQueueEntry(tracked, results[index]?.data, results[index]?.error))
    .sort(byNewestFirst);

  function dismiss(id: string): void {
    store.removeTrackedJob(id);
  }

  function clearCompleted(): void {
    entries.filter((entry) => isTerminalJobStatus(entry.status)).forEach((entry) => store.removeTrackedJob(entry.id));
  }

  return { entries, dismiss, clearCompleted };
}
