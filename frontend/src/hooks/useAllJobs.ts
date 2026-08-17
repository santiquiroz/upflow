import { useQueries } from "@tanstack/react-query";
import { apiGet, listJobs, listVideoJobs } from "../lib/api";
import type { JobStatus } from "../lib/apiTypes";
import type { JobKind } from "../lib/jobTypeGuards";
import { listAudioJobs } from "../services/audio";
import { listGenerationJobs } from "../services/generation";

const POLL_INTERVAL_MS = 2000;

export interface AllJobsEntry {
  id: string;
  kind: JobKind;
  fileName: string;
  createdAt: number;
  status: JobStatus;
  ownerId: string | null;
  downloadUrl: string | null;
}

// La vista de admin mira las MISMAS siete familias que la cola. Mientras cubrio
// solo cuatro, un admin veia media app: los trabajos de transcripcion, descarga
// y 3D de otros usuarios no existian para el, aunque el servidor los listara.
interface RawAllJob {
  id?: string;
  jobId?: string;
  status: JobStatus;
  createdAt: string;
  ownerId?: string | null;
  downloadUrl?: string | null;
  originalFilename?: string;
  prompt?: string;
  mediaTitle?: string | null;
  url?: string;
}

// El nombre que el usuario reconoce depende de la familia: el archivo que
// subio, el prompt que escribio, el titulo de lo que pidio bajar. Un prompt
// vacio es real (el modo foto del 3D no tiene texto) y por eso se descarta como
// candidato en vez de mostrarse en blanco. Misma regla que la rehidratacion.
function displayName(job: RawAllJob, id: string): string {
  const candidates = [job.originalFilename, job.prompt, job.mediaTitle, job.url];
  return candidates.find((name) => typeof name === "string" && name.trim() !== "") ?? id;
}

function toEntry(job: RawAllJob, kind: JobKind): AllJobsEntry | null {
  const id = job.jobId ?? job.id;
  if (!id) {
    return null;
  }
  return {
    id,
    kind,
    fileName: displayName(job, id),
    createdAt: Date.parse(job.createdAt),
    status: job.status,
    ownerId: job.ownerId ?? null,
    downloadUrl: job.downloadUrl ?? null,
  };
}

interface FamilyQuery {
  kind: JobKind;
  fetch: () => Promise<{ jobs: RawAllJob[] }>;
}

const FAMILIES: readonly FamilyQuery[] = [
  { kind: "image", fetch: () => listJobs(true) as Promise<{ jobs: RawAllJob[] }> },
  { kind: "video", fetch: () => listVideoJobs(true) as Promise<{ jobs: RawAllJob[] }> },
  { kind: "audio", fetch: () => listAudioJobs(true) as Promise<{ jobs: RawAllJob[] }> },
  { kind: "generation", fetch: () => listGenerationJobs(true) as Promise<{ jobs: RawAllJob[] }> },
  { kind: "transcribe", fetch: () => apiGet<{ jobs: RawAllJob[] }>("/transcribe/jobs?all=true") },
  { kind: "download", fetch: () => apiGet<{ jobs: RawAllJob[] }>("/download/jobs?all=true") },
  { kind: "shape3d", fetch: () => apiGet<{ jobs: RawAllJob[] }>("/print/generate?all=true") },
];

export function useAllJobsView(enabled: boolean): AllJobsEntry[] {
  const results = useQueries({
    queries: FAMILIES.map((family) => ({
      queryKey: ["allJobs", family.kind],
      queryFn: family.fetch,
      enabled,
      refetchInterval: enabled ? POLL_INTERVAL_MS : false,
    })),
  });

  if (!enabled) {
    return [];
  }

  const entries = results.flatMap((result, index) =>
    (result.data?.jobs ?? [])
      .map((job) => toEntry(job, FAMILIES[index].kind))
      .filter((entry): entry is AllJobsEntry => entry !== null),
  );
  return entries.sort((a, b) => b.createdAt - a.createdAt);
}
