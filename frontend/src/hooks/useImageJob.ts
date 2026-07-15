import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import type { CreateImageJobParams } from "../lib/api";
import { createImageJob, getJob } from "../lib/api";
import type { JobResponse, JobStatus } from "../lib/apiTypes";
import { isTerminalJobStatus } from "../lib/jobStatus";

export const DEFAULT_POLL_INTERVAL_MS = 1500;

export type ImageJobPhase = "idle" | "uploading" | JobStatus;

export interface UseImageJobResult {
  phase: ImageJobPhase;
  job: JobResponse | undefined;
  errorMessage: string | null;
  submit: (params: CreateImageJobParams) => void;
  reset: () => void;
}

function resolvePhase(
  isUploading: boolean,
  initialStatus: JobStatus | undefined,
  job: JobResponse | undefined,
): ImageJobPhase {
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
  job: JobResponse | undefined,
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

export function useImageJob(pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS): UseImageJobResult {
  const [jobId, setJobId] = useState<string | null>(null);

  const uploadMutation = useMutation({
    mutationFn: createImageJob,
    onSuccess: (data) => setJobId(data.jobId),
  });

  const jobQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => (isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : pollIntervalMs),
  });

  function submit(params: CreateImageJobParams): void {
    setJobId(null);
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
