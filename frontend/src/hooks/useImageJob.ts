import { submitBatch } from "../lib/batchSubmit";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "../i18n/LocaleProvider";
import { useRef, useState } from "react";
import type { CreateImageJobParams } from "../lib/api";
import { cancelJob, createImageJob, getJob } from "../lib/api";
import type { JobResponse, JobStatus } from "../lib/apiTypes";
import { isTerminalJobStatus } from "../lib/jobStatus";
import { jobQueueStore, type JobQueueStore } from "../lib/jobQueueStore";

export const DEFAULT_POLL_INTERVAL_MS = 1500;

export type ImageJobPhase = "idle" | "uploading" | JobStatus;

export interface UseImageJobResult {
  phase: ImageJobPhase;
  job: JobResponse | undefined;
  errorMessage: string | null;
  // `null` mientras no se este subiendo, o cuando el total no es computable:
  // dibujar un porcentaje inventado seria mentir sobre lo que falta.
  uploadPercent: number | null;
  submit: (params: CreateImageJobParams) => void;
  /** Un trabajo por archivo, subidos de a uno. La tarjeta sigue al primero —
   *  es el que termina primero — y el resto se ve en la cola global. */
  submitMany: (paramsList: CreateImageJobParams[]) => void;
  /** Cuantos archivos del lote todavia no se subieron. */
  pendingUploads: number;
  /** Cuantos del lote no llegaron al servidor. Se cuentan para decirlo: un lote
   *  al que le faltan tres archivos y no avisa es una falla silenciosa. */
  failedUploads: number;
  cancel: () => void;
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

export function useImageJob(
  pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS,
  queue: JobQueueStore = jobQueueStore,
): UseImageJobResult {
  const { t } = useTranslation();
  const [jobId, setJobId] = useState<string | null>(null);
  const pendingFileNameRef = useRef<string>("image");
  const queryClient = useQueryClient();

  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  // Mientras se sube todavia no hay jobId: cortar el envio es lo unico que
  // "cancelar" puede significar en ese momento.
  const uploadAbortRef = useRef<AbortController | null>(null);
  const [pendingUploads, setPendingUploads] = useState(0);
  const [failedUploads, setFailedUploads] = useState(0);
  const uploadMutation = useMutation({
    mutationFn: (params: Parameters<typeof createImageJob>[0]) => {
      const controller = new AbortController();
      uploadAbortRef.current = controller;
      return createImageJob(params, {
        onProgress: setUploadPercent,
        signal: controller.signal,
      });
    },
    onSuccess: (data) => {
      setJobId(data.jobId);
      queue.addTrackedJob({
        id: data.jobId,
        kind: "image",
        fileName: pendingFileNameRef.current,
        createdAt: Date.now(),
      });
    },
  });

  const jobQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => (isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : pollIntervalMs),
  });

  function submit(params: CreateImageJobParams): void {
    setJobId(null);
    pendingFileNameRef.current = params.file.name;
    uploadMutation.mutate(params);
  }

  async function submitMany(paramsList: CreateImageJobParams[]): Promise<void> {
    setJobId(null);
    await submitBatch({
      paramsList,
      kind: "image",
      queue,
      fileNameOf: (params) => params.file.name,
      uploadFirst: (params) => uploadMutation.mutateAsync(params),
      createRest: async (params) => (await createImageJob(params)).jobId,
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
      // Cortar la subida no es un error: el usuario pidio que se cortara, y
      // dejar la tarjeta en rojo diria que algo salio mal.
      uploadAbortRef.current?.abort();
      uploadAbortRef.current = null;
      uploadMutation.reset();
      setUploadPercent(null);
      return;
    }
    void cancelJob(jobId)
      .then(() => queryClient.invalidateQueries({ queryKey: ["job", jobId] }))
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
