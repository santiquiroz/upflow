import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "../i18n/LocaleProvider";
import { useRef, useState } from "react";
import type { CreateVideoJobParams } from "../lib/api";
import { cancelVideoJob, createVideoJob, getVideoCapabilities, getVideoJob } from "../lib/api";
import type { JobStatus, VideoCapabilities, VideoJobResponse } from "../lib/apiTypes";
import { isTerminalJobStatus } from "../lib/jobStatus";
import { jobQueueStore, type JobQueueStore } from "../lib/jobQueueStore";

export const DEFAULT_POLL_INTERVAL_MS = 1500;

export type VideoJobPhase = "idle" | "uploading" | JobStatus;

export interface UseVideoJobResult {
  phase: VideoJobPhase;
  job: VideoJobResponse | undefined;
  errorMessage: string | null;
  // `null` mientras no se este subiendo, o cuando el total no es computable:
  // dibujar un porcentaje inventado seria mentir sobre lo que falta.
  uploadPercent: number | null;
  submit: (params: CreateVideoJobParams) => void;
  cancel: () => void;
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

export function useVideoJob(
  pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS,
  queue: JobQueueStore = jobQueueStore,
): UseVideoJobResult {
  const { t } = useTranslation();
  const [jobId, setJobId] = useState<string | null>(null);
  const pendingFileNameRef = useRef<string>("video");
  const queryClient = useQueryClient();

  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const uploadMutation = useMutation({
    mutationFn: (params: Parameters<typeof createVideoJob>[0]) =>
      createVideoJob(params, { onProgress: setUploadPercent }),
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
    pendingFileNameRef.current = params.fileName ?? params.file?.name ?? pendingFileNameRef.current;
    uploadMutation.mutate(params);
  }

  // Best-effort: a 409 (job already finished) needs no surfaced error since the
  // running poll is the source of truth and reconciles the status on refetch.
  function cancel(): void {
    if (jobId === null) {
      return;
    }
    void cancelVideoJob(jobId)
      .then(() => queryClient.invalidateQueries({ queryKey: ["videoJob", jobId] }))
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

export function useVideoCapabilities() {
  return useQuery<VideoCapabilities>({ queryKey: ["videoCapabilities"], queryFn: getVideoCapabilities });
}
