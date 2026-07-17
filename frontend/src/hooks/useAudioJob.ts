import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import type { AudioCapabilities, AudioJob, JobStatus } from "../lib/apiTypes";
import { isTerminalJobStatus } from "../lib/jobStatus";
import { jobQueueStore, type JobQueueStore } from "../lib/jobQueueStore";
import {
  cancelAudioJob,
  createAudioJob,
  fetchAudioCapabilities,
  getAudioJob,
  type CreateAudioJobParams,
} from "../services/audio";

export const DEFAULT_POLL_INTERVAL_MS = 1500;

export type AudioJobPhase = "idle" | "uploading" | JobStatus;

export interface UseAudioJobResult {
  phase: AudioJobPhase;
  job: AudioJob | undefined;
  errorMessage: string | null;
  submit: (params: CreateAudioJobParams) => void;
  cancel: () => void;
  reset: () => void;
}

function resolvePhase(
  isUploading: boolean,
  initialStatus: JobStatus | undefined,
  job: AudioJob | undefined,
): AudioJobPhase {
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
  job: AudioJob | undefined,
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

export function useAudioJob(
  pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS,
  queue: JobQueueStore = jobQueueStore,
): UseAudioJobResult {
  const [jobId, setJobId] = useState<string | null>(null);
  const pendingFileNameRef = useRef<string>("audio");
  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: createAudioJob,
    onSuccess: (data) => {
      setJobId(data.jobId);
      queue.addTrackedJob({
        id: data.jobId,
        kind: "audio",
        fileName: pendingFileNameRef.current,
        createdAt: Date.now(),
      });
    },
  });

  const jobQuery = useQuery({
    queryKey: ["audioJob", jobId],
    queryFn: () => getAudioJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => (isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : pollIntervalMs),
  });

  function submit(params: CreateAudioJobParams): void {
    setJobId(null);
    pendingFileNameRef.current = params.file.name;
    uploadMutation.mutate(params);
  }

  // Best-effort: a 409 (job already finished) needs no surfaced error since the
  // running poll is the source of truth and reconciles the status on refetch.
  function cancel(): void {
    if (jobId === null) {
      return;
    }
    void cancelAudioJob(jobId)
      .then(() => queryClient.invalidateQueries({ queryKey: ["audioJob", jobId] }))
      .catch(() => undefined);
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
    cancel,
    reset,
  };
}

export function useAudioCapabilities() {
  return useQuery<AudioCapabilities>({ queryKey: ["audioCapabilities"], queryFn: fetchAudioCapabilities });
}
