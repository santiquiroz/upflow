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
  const [pendingUploads, setPendingUploads] = useState(0);
  const [failedUploads, setFailedUploads] = useState(0);
  const uploadMutation = useMutation({
    mutationFn: (params: Parameters<typeof createImageJob>[0]) =>
      createImageJob(params, { onProgress: setUploadPercent }),
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

  // De a uno y no todos juntos: en paralelo compiten por el ancho de banda, el
  // porcentaje de subida deja de significar algo y la cola del servidor
  // (acotada) responde 429 a los ultimos.
  async function submitMany(paramsList: CreateImageJobParams[]): Promise<void> {
    if (paramsList.length === 0) {
      return;
    }
    const [primero, ...resto] = paramsList;
    setPendingUploads(resto.length);
    setFailedUploads(0);
    setJobId(null);
    pendingFileNameRef.current = primero.file.name;
    // Se espera al primero antes de arrancar el resto: sin esto los archivos
    // llegan al servidor en un orden distinto del que eligio el usuario.
    await uploadMutation.mutateAsync(primero).catch(() => undefined);
    for (const params of resto) {
      try {
        const creado = await createImageJob(params);
        queue.addTrackedJob({
          id: creado.jobId,
          kind: "image",
          fileName: params.file.name,
          createdAt: Date.now(),
        });
      } catch {
        // No se corta el lote: que un archivo no entre no es razon para no
        // intentar los que siguen. Se cuenta y se dice al final.
        setFailedUploads((fallados) => fallados + 1);
      } finally {
        setPendingUploads((quedan) => Math.max(0, quedan - 1));
      }
    }
  }

  // Best-effort: a 409 (job already finished) needs no surfaced error since the
  // running poll is the source of truth and reconciles the status on refetch.
  function cancel(): void {
    if (jobId === null) {
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
