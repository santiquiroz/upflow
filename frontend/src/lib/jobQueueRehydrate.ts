import { apiGet } from "./api";
import type { JobStatus } from "./apiTypes";
import { isTerminalJobStatus } from "./jobStatus";
import type { JobQueueStore, TrackedJob, TrackedJobKind } from "./jobQueueStore";

// La cola vivia solo en memoria: recargar el navegador la borraba aunque los
// trabajos siguieran corriendo en el servidor. Se rehidrata desde el SERVIDOR y
// no desde localStorage a proposito — el servidor es la fuente real, asi que no
// pueden aparecer fantasmas de trabajos que ya no existen, y ademas ya trae el
// nombre del archivo, que era lo unico que localStorage habria aportado.

interface RawJob {
  id?: string;
  jobId?: string;
  status: JobStatus;
  originalFilename?: string;
  prompt?: string;
  mediaTitle?: string | null;
  url?: string;
}

export interface JobFetchers {
  fetchImageJobs: () => Promise<{ jobs: RawJob[] }>;
  fetchVideoJobs: () => Promise<{ jobs: RawJob[] }>;
  fetchAudioJobs: () => Promise<{ jobs: RawJob[] }>;
  fetchGenerationJobs: () => Promise<{ jobs: RawJob[] }>;
  fetchTranscribeJobs: () => Promise<{ jobs: RawJob[] }>;
  fetchDownloadJobs: () => Promise<{ jobs: RawJob[] }>;
  fetchShape3dJobs: () => Promise<{ jobs: RawJob[] }>;
}

export const defaultJobFetchers: JobFetchers = {
  fetchImageJobs: () => apiGet<{ jobs: RawJob[] }>("/jobs?all=false"),
  fetchVideoJobs: () => apiGet<{ jobs: RawJob[] }>("/video/jobs?all=false"),
  fetchAudioJobs: () => apiGet<{ jobs: RawJob[] }>("/audio/jobs?all=false"),
  fetchGenerationJobs: () => apiGet<{ jobs: RawJob[] }>("/generation/jobs?all=false"),
  fetchTranscribeJobs: () => apiGet<{ jobs: RawJob[] }>("/transcribe/jobs?all=false"),
  fetchDownloadJobs: () => apiGet<{ jobs: RawJob[] }>("/download/jobs?all=false"),
  fetchShape3dJobs: () => apiGet<{ jobs: RawJob[] }>("/print/generate?all=false"),
};

// Cada familia nombra su trabajo con lo que el usuario reconoce: el archivo que
// subio, el prompt que escribio o el titulo (o la URL) de lo que pidio bajar. Un
// prompt vacio es real — el modo foto del 3D no tiene texto — asi que se descarta
// como candidato en vez de mostrarse en blanco.
export function displayName(raw: RawJob, id: string): string {
  const candidates = [raw.originalFilename, raw.prompt, raw.mediaTitle, raw.url];
  return candidates.find((name) => typeof name === "string" && name.trim() !== "") ?? id;
}

function toTracked(raw: RawJob, kind: TrackedJobKind, index: number): TrackedJob | null {
  const id = raw.jobId ?? raw.id;
  if (!id) {
    return null;
  }
  return {
    id,
    kind,
    fileName: displayName(raw, id),
    createdAt: index,
  };
}

async function pending(
  load: () => Promise<{ jobs: RawJob[] }>,
  kind: TrackedJobKind,
): Promise<TrackedJob[]> {
  try {
    const { jobs } = await load();
    return jobs
      .filter((job) => !isTerminalJobStatus(job.status))
      .map((job, index) => toTracked(job, kind, index))
      .filter((job): job is TrackedJob => job !== null);
  } catch {
    // Un endpoint caido no puede vaciar el resto de la cola ni tumbar el arranque.
    return [];
  }
}

export async function rehydrateJobQueue(
  store: JobQueueStore,
  fetchers: JobFetchers = defaultJobFetchers,
): Promise<void> {
  const groups = await Promise.all([
    pending(fetchers.fetchImageJobs, "image"),
    pending(fetchers.fetchVideoJobs, "video"),
    pending(fetchers.fetchAudioJobs, "audio"),
    pending(fetchers.fetchGenerationJobs, "generation"),
    pending(fetchers.fetchTranscribeJobs, "transcribe"),
    pending(fetchers.fetchDownloadJobs, "download"),
    pending(fetchers.fetchShape3dJobs, "shape3d"),
  ]);

  const yaSeguidos = new Set(store.getSnapshot().map((job) => job.id));
  // El store antepone cada trabajo (el mas nuevo primero), asi que se recorre al
  // reves para que la cola quede en el mismo orden en que la devolvio el servidor.
  for (const job of groups.flat().reverse()) {
    if (!yaSeguidos.has(job.id)) {
      store.addTrackedJob(job);
      yaSeguidos.add(job.id);
    }
  }
}
