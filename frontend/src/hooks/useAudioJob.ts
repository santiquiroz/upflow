import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "../i18n/LocaleProvider";
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
  // `null` mientras no se este subiendo, o cuando el total no es computable:
  // dibujar un porcentaje inventado seria mentir sobre lo que falta.
  uploadPercent: number | null;
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

  t: (key: string) => string,
): string | null {
  if (uploadError instanceof Error) {
    return uploadError.message;
  }
  if (jobError instanceof Error) {
    return jobError.message;
  }
  if (job?.status === "failed") {
    return job.error ?? t("job.failed");
  }
  return null;
}

export function useAudioJob(
  pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS,
  queue: JobQueueStore = jobQueueStore,
): UseAudioJobResult {
  const { t } = useTranslation();
  const [jobId, setJobId] = useState<string | null>(null);
  const pendingFileNameRef = useRef<string>("audio");
  const queryClient = useQueryClient();

  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  // Mientras se sube todavia no hay jobId: cortar el envio es lo unico que
  // "cancelar" puede significar en ese momento.
  const uploadAbortRef = useRef<AbortController | null>(null);
  const uploadMutation = useMutation({
    mutationFn: (params: Parameters<typeof createAudioJob>[0]) => {
      const controller = new AbortController();
      uploadAbortRef.current = controller;
      return createAudioJob(params, {
        onProgress: setUploadPercent,
        signal: controller.signal,
      });
    },
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
      // Cortar la subida no es un error: el usuario pidio que se cortara.
      uploadAbortRef.current?.abort();
      uploadAbortRef.current = null;
      uploadMutation.reset();
      setUploadPercent(null);
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
    uploadPercent,
    phase: resolvePhase(uploadMutation.isPending, uploadMutation.data?.status, jobQuery.data),
    job: jobQuery.data,
    errorMessage: resolveErrorMessage(uploadMutation.error, jobQuery.error, jobQuery.data, t),
    submit,
    cancel,
    reset,
  };
}

export function useAudioCapabilities() {
  return useQuery<AudioCapabilities>({ queryKey: ["audioCapabilities"], queryFn: fetchAudioCapabilities });
}
