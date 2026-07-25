import { useQueries } from "@tanstack/react-query";
import { listJobs, listVideoJobs } from "../lib/api";
import type { AudioJob, GenerationJob, JobResponse, JobStatus, VideoJobResponse } from "../lib/apiTypes";
import { listAudioJobs } from "../services/audio";
import { listGenerationJobs } from "../services/generation";

const POLL_INTERVAL_MS = 2000;

export interface AllJobsEntry {
  id: string;
  kind: "image" | "video" | "audio" | "generation";
  fileName: string;
  createdAt: number;
  status: JobStatus;
  ownerId: string | null;
  downloadUrl: string | null;
}

function imageEntry(job: JobResponse): AllJobsEntry {
  return {
    id: job.jobId, kind: "image", fileName: job.originalFilename, createdAt: Date.parse(job.createdAt),
    status: job.status, ownerId: job.ownerId, downloadUrl: job.downloadUrl,
  };
}

function videoEntry(job: VideoJobResponse): AllJobsEntry {
  return {
    id: job.jobId, kind: "video", fileName: job.originalFilename, createdAt: Date.parse(job.createdAt),
    status: job.status, ownerId: job.ownerId, downloadUrl: job.downloadUrl,
  };
}

function audioEntry(job: AudioJob): AllJobsEntry {
  return {
    id: job.id, kind: "audio", fileName: job.originalFilename, createdAt: Date.parse(job.createdAt),
    status: job.status, ownerId: job.ownerId, downloadUrl: job.downloadUrl,
  };
}

function generationEntry(job: GenerationJob): AllJobsEntry {
  return {
    id: job.id, kind: "generation", fileName: job.prompt, createdAt: Date.parse(job.createdAt),
    status: job.status, ownerId: job.ownerId, downloadUrl: job.downloadUrl,
  };
}

export function useAllJobsView(enabled: boolean): AllJobsEntry[] {
  const [imageResult, videoResult, audioResult, generationResult] = useQueries({
    queries: [
      {
        queryKey: ["allJobs", "image"], queryFn: () => listJobs(true), enabled,
        refetchInterval: enabled ? POLL_INTERVAL_MS : false,
      },
      {
        queryKey: ["allJobs", "video"], queryFn: () => listVideoJobs(true), enabled,
        refetchInterval: enabled ? POLL_INTERVAL_MS : false,
      },
      {
        queryKey: ["allJobs", "audio"], queryFn: () => listAudioJobs(true), enabled,
        refetchInterval: enabled ? POLL_INTERVAL_MS : false,
      },
      {
        queryKey: ["allJobs", "generation"], queryFn: () => listGenerationJobs(true), enabled,
        refetchInterval: enabled ? POLL_INTERVAL_MS : false,
      },
    ],
  });

  if (!enabled) {
    return [];
  }

  const entries = [
    ...(imageResult.data?.jobs ?? []).map(imageEntry),
    ...(videoResult.data?.jobs ?? []).map(videoEntry),
    ...(audioResult.data?.jobs ?? []).map(audioEntry),
    ...(generationResult.data?.jobs ?? []).map(generationEntry),
  ];
  return entries.sort((a, b) => b.createdAt - a.createdAt);
}
