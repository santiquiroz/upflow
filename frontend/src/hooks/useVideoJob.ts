import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "../i18n/LocaleProvider";
import { useRef, useState } from "react";
import type { CreateVideoJobParams } from "../lib/api";
import { cancelVideoJob, createVideoJob, getVideoCapabilities, getVideoJob } from "../lib/api";
import { submitBatch } from "../lib/batchSubmit";
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
  submitMany: (paramsList: CreateVideoJobParams[]) => void;
  pendingUploads: number;
  failedUploads: number;
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
  const [pendingUploads, setPendingUploads] = useState(0);
  const [failedUploads, setFailedUploads] = useState(0);
  // Mientras se sube todavia no hay jobId: cortar el envio es lo unico que
  // "cancelar" puede significar en ese momento.
  const uploadAbortRef = useRef<AbortController | null>(null);
  const uploadMutation = useMutation({
    mutationFn: (params: Parameters<typeof createVideoJob>[0]) => {
      const controller = new AbortController();
      uploadAbortRef.current = controller;
      return createVideoJob(params, {
        onProgress: setUploadPercent,
        signal: controller.signal,
      });
    },
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

  async function submitMany(paramsList: CreateVideoJobParams[]): Promise<void> {
    setJobId(null);
    await submitBatch({
      paramsList,
      kind: "video",
      queue,
      // Un lote sube archivos, nunca tokens de analisis: el analisis es por
      // archivo y solo existe cuando el usuario eligio uno solo.
      fileNameOf: (params) => params.fileName ?? params.file?.name ?? "",
      uploadFirst: (params) => uploadMutation.mutateAsync(params),
      createRest: async (params) => (await createVideoJob(params)).jobId,
      onPendingChange: setPendingUploads,
      onFailedChange: setFailedUploads,
      onFirstStarted: (nombre) => {
        pendingFileNameRef.current = nombre;
      },
    });
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
    submitMany: (paramsList) => void submitMany(paramsList),
    pendingUploads,
    failedUploads,
    cancel,
    reset,
  };
}

export function useVideoCapabilities() {
  return useQuery<VideoCapabilities>({ queryKey: ["videoCapabilities"], queryFn: getVideoCapabilities });
}
