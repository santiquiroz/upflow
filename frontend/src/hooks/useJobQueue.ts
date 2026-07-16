import { useQueries } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";
import { getJob, getVideoJob } from "../lib/api";
import type { AudioJob, JobResponse, JobStatus, VideoJobResponse } from "../lib/apiTypes";
import { isTerminalJobStatus } from "../lib/jobStatus";
import { jobQueueStore, type JobQueueStore, type TrackedJob } from "../lib/jobQueueStore";
import { getAudioJob } from "../services/audio";

export const DEFAULT_QUEUE_POLL_INTERVAL_MS = 1500;

export type TrackedJobResponse = JobResponse | VideoJobResponse | AudioJob;

export interface JobQueueEntry {
  id: string;
  kind: TrackedJob["kind"];
  fileName: string;
  createdAt: number;
  status: JobStatus;
  downloadUrl: string | null;
  errorMessage: string | null;
  job: TrackedJobResponse | undefined;
}

export interface UseJobQueueResult {
  entries: JobQueueEntry[];
  dismiss: (id: string) => void;
  clearCompleted: () => void;
}

const QUERY_KEY_BY_KIND: Record<TrackedJob["kind"], string> = {
  image: "job",
  video: "videoJob",
  audio: "audioJob",
};

function fetchTrackedJob(tracked: TrackedJob): Promise<TrackedJobResponse> {
  if (tracked.kind === "image") {
    return getJob(tracked.id);
  }
  if (tracked.kind === "audio") {
    return getAudioJob(tracked.id);
  }
  return getVideoJob(tracked.id);
}

function resolveEntryError(data: TrackedJobResponse | undefined, queryError: unknown): string | null {
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
  data: TrackedJobResponse | undefined,
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
    job: data,
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
      queryKey: [QUERY_KEY_BY_KIND[tracked.kind], tracked.id],
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
