import { useMutation, useQuery } from "@tanstack/react-query";
import { useRef, useState } from "react";
import type { CreateVideoJobParams } from "../lib/api";
import { createVideoJob, getVideoJob } from "../lib/api";
import type { JobStatus, VideoJobResponse } from "../lib/apiTypes";
import { isTerminalJobStatus } from "../lib/jobStatus";
import { jobQueueStore, type JobQueueStore } from "../lib/jobQueueStore";

export const DEFAULT_POLL_INTERVAL_MS = 1500;

export type VideoJobPhase = "idle" | "uploading" | JobStatus;

export interface UseVideoJobResult {
  phase: VideoJobPhase;
  job: VideoJobResponse | undefined;
  errorMessage: string | null;
  submit: (params: CreateVideoJobParams) => void;
  reset: () => void;
}

function resolvePhase(
  isUploading: boolean,
  initialStatus: JobStatus | undefined,
  job: VideoJobResponse | undefined,
): VideoJobPhase {
  if (isUploading) {
    return "uploading";
  }
  if (job) {
    return job.status;
  }
  if (initialStatus) {
    return initialStatus;
  }
  return "idle";
}

function resolveErrorMessage(
  uploadError: unknown,
  jobError: unknown,
  job: VideoJobResponse | undefined,
): string | null {
  if (uploadError instanceof Error) {
    return uploadError.message;
  }
  if (jobError instanceof Error) {
    return jobError.message;
  }
  if (job?.status === "failed") {
    return job.error ?? "The job failed.";
  }
  return null;
}

export function useVideoJob(
  pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS,
  queue: JobQueueStore = jobQueueStore,
): UseVideoJobResult {
  const [jobId, setJobId] = useState<string | null>(null);
  const pendingFileNameRef = useRef<string>("video");

  const uploadMutation = useMutation({
    mutationFn: createVideoJob,
    onSuccess: (data) => {
      setJobId(data.jobId);
      queue.addTrackedJob({
        id: data.jobId,
        kind: "video",
        fileName: pendingFileNameRef.current,
        createdAt: Date.now(),
      });
    },
  });

  const jobQuery = useQuery({
    queryKey: ["videoJob", jobId],
    queryFn: () => getVideoJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => (isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : pollIntervalMs),
  });

  function submit(params: CreateVideoJobParams): void {
    setJobId(null);
    pendingFileNameRef.current = params.file.name;
    uploadMutation.mutate(params);
  }

  function reset(): void {
    setJobId(null);
    uploadMutation.reset();
  }

  return {
    phase: resolvePhase(uploadMutation.isPending, uploadMutation.data?.status, jobQuery.data),
    job: jobQuery.data,
    errorMessage: resolveErrorMessage(uploadMutation.error, jobQuery.error, jobQuery.data),
    submit,
    reset,
  };
}
