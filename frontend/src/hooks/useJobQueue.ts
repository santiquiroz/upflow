import { useQueries, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "../i18n/LocaleProvider";
import { useEffect, useSyncExternalStore } from "react";
import { cancelJob, cancelVideoJob, getJob, getVideoJob } from "../lib/api";
import type { DownloadJob, JobStatus } from "../lib/apiTypes";
import { isTerminalJobStatus } from "../lib/jobStatus";
import type { AnyQueuedJob } from "../lib/jobTypeGuards";
import { jobQueueStore, type JobQueueStore, type TrackedJob } from "../lib/jobQueueStore";
import { cancelAudioJob, getAudioJob } from "../services/audio";
import { cancelDownloadJob, getDownloadJob } from "../services/download";
import { cancelGenerationJob, getGenerationJob } from "../services/generation";
import { cancelShape3dJob, getShape3dJob } from "../services/print";
import { cancelTranscribeJob, getTranscribeJob } from "../services/transcribe";

export const DEFAULT_QUEUE_POLL_INTERVAL_MS = 1500;

export type TrackedJobResponse = AnyQueuedJob;

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
  cancel: (id: string) => void;
  clearCompleted: () => void;
}

const QUERY_KEY_BY_KIND: Record<TrackedJob["kind"], string> = {
  image: "job",
  video: "videoJob",
  audio: "audioJob",
  generation: "generationJob",
  transcribe: "transcribe-job",
  download: "download-job",
  shape3d: "shape3dJob",
};

const CANCEL_BY_KIND: Record<TrackedJob["kind"], (id: string) => Promise<TrackedJobResponse>> = {
  image: cancelJob,
  video: cancelVideoJob,
  audio: cancelAudioJob,
  generation: cancelGenerationJob,
  transcribe: cancelTranscribeJob,
  download: cancelDownloadJob,
  shape3d: cancelShape3dJob,
};

const FETCH_BY_KIND: Record<TrackedJob["kind"], (id: string) => Promise<TrackedJobResponse>> = {
  image: getJob,
  video: getVideoJob,
  audio: getAudioJob,
  generation: getGenerationJob,
  transcribe: getTranscribeJob,
  download: getDownloadJob,
  shape3d: getShape3dJob,
};

function fetchTrackedJob(tracked: TrackedJob): Promise<TrackedJobResponse> {
  return FETCH_BY_KIND[tracked.kind](tracked.id);
}

function resolveEntryError(data: TrackedJobResponse | undefined, queryError: unknown,
  t: (key: string) => string,
): string | null {
  if (queryError instanceof Error) {
    return queryError.message;
  }
  if (data?.status === "failed") {
    return data.error ?? t("job.failed");
  }
  return null;
}

// Una descarga deja los archivos en disco: no tiene URL de descarga, asi que la
// propiedad no existe en su respuesta en vez de venir en null.
function readDownloadUrl(data: TrackedJobResponse | undefined): string | null {
  if (!data || !("downloadUrl" in data)) {
    return null;
  }
  return data.downloadUrl ?? null;
}

/** Los trabajos que una descarga encadeno, listos para entrar a la cola.
 *
 * Una descarga con "separar al terminar" produce trabajos de audio que nadie
 * pidio por pantalla: sin registrarlos, las pistas salen en silencio y el
 * usuario no tiene donde ver el progreso ni bajarlas.
 *
 * Los nombres salen de los archivos bajados, en el mismo orden: el servidor
 * dispara un trabajo por archivo. Si faltara alguno, el titulo del medio es
 * mejor etiqueta que un id.
 */
function followUpJobsOf(
  tracked: TrackedJob,
  data: TrackedJobResponse | undefined,
): TrackedJob[] {
  if (!data || !("followupJobIds" in data)) {
    return [];
  }
  const job = data as DownloadJob;
  return job.followupJobIds.map((id, index) => ({
    id,
    kind: "audio" as const,
    fileName: job.outputFiles[index] ?? job.mediaTitle ?? tracked.fileName,
    createdAt: Date.now(),
  }));
}

function toQueueEntry(
  tracked: TrackedJob,
  data: TrackedJobResponse | undefined,
  queryError: unknown,
  t: (key: string) => string,
): JobQueueEntry {
  return {
    id: tracked.id,
    kind: tracked.kind,
    fileName: tracked.fileName,
    createdAt: tracked.createdAt,
    status: data?.status ?? "queued",
    downloadUrl: readDownloadUrl(data),
    errorMessage: resolveEntryError(data, queryError, t),
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
  const { t } = useTranslation();
  const trackedJobs = useSyncExternalStore(store.subscribe, store.getSnapshot, store.getSnapshot);
  const queryClient = useQueryClient();

  const results = useQueries({
    queries: trackedJobs.map((tracked) => ({
      queryKey: [QUERY_KEY_BY_KIND[tracked.kind], tracked.id],
      queryFn: () => fetchTrackedJob(tracked),
      refetchInterval: (query: { state: { data?: { status?: JobStatus } } }) =>
        isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : pollIntervalMs,
    })),
  });

  const entries = trackedJobs
    .map((tracked, index) => toQueueEntry(tracked, results[index]?.data, results[index]?.error, t))
    .sort(byNewestFirst);

  // En un efecto y no al vuelo: agregar a la cola durante el render es escribir
  // en un store mientras se lee. `addTrackedJob` ignora ids repetidos, asi que
  // reejecutarlo en cada sondeo no duplica nada.
  useEffect(() => {
    trackedJobs.forEach((tracked, index) => {
      followUpJobsOf(tracked, results[index]?.data).forEach(store.addTrackedJob);
    });
  });

  function dismiss(id: string): void {
    store.removeTrackedJob(id);
  }

  // Best-effort: the server may answer 409 if the job just finished, but the
  // running poll is the source of truth for the displayed status, so a rejected
  // cancel needs no surfaced error -- the next refetch reconciles the state.
  function cancel(id: string): void {
    const tracked = trackedJobs.find((job) => job.id === id);
    if (!tracked) {
      return;
    }
    void CANCEL_BY_KIND[tracked.kind](id)
      .then(() => queryClient.invalidateQueries({ queryKey: [QUERY_KEY_BY_KIND[tracked.kind], id] }))
      .catch(() => undefined);
  }

  function clearCompleted(): void {
    entries.filter((entry) => isTerminalJobStatus(entry.status)).forEach((entry) => store.removeTrackedJob(entry.id));
  }

  return { entries, dismiss, cancel, clearCompleted };
}
