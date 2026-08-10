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
}

// Transcripcion, descarga y 3D NO se rehidratan: su API no expone un endpoint de
// lista (ver app/mcp/jobs.py::FAMILIES, has_list=False), asi que tras recargar
// el navegador no hay forma de preguntar "que quedo corriendo" sin inventar
// rutas nuevas. Se siguen en la cola durante la sesion en que se crean.
export interface JobFetchers {
  fetchImageJobs: () => Promise<{ jobs: RawJob[] }>;
  fetchVideoJobs: () => Promise<{ jobs: RawJob[] }>;
  fetchAudioJobs: () => Promise<{ jobs: RawJob[] }>;
  fetchGenerationJobs: () => Promise<{ jobs: RawJob[] }>;
}

export const defaultJobFetchers: JobFetchers = {
  fetchImageJobs: () => apiGet<{ jobs: RawJob[] }>("/jobs?all=false"),
  fetchVideoJobs: () => apiGet<{ jobs: RawJob[] }>("/video/jobs?all=false"),
  fetchAudioJobs: () => apiGet<{ jobs: RawJob[] }>("/audio/jobs?all=false"),
  fetchGenerationJobs: () => apiGet<{ jobs: RawJob[] }>("/generation/jobs?all=false"),
};

function toTracked(raw: RawJob, kind: TrackedJobKind, index: number): TrackedJob | null {
  const id = raw.jobId ?? raw.id;
  if (!id) {
    return null;
  }
  return {
    id,
    kind,
    // Un trabajo de generacion no tiene archivo de entrada: su nombre es el prompt.
    fileName: raw.originalFilename ?? raw.prompt ?? id,
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
