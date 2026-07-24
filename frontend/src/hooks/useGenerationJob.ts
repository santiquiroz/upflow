import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import type { GenerationCapabilities, GenerationJob, JobStatus } from "../lib/apiTypes";
import { isTerminalInstallStatus } from "../lib/installStatus";
import { isTerminalJobStatus } from "../lib/jobStatus";
import { jobQueueStore, type JobQueueStore } from "../lib/jobQueueStore";
import {
  cancelGenerationJob,
  createGenerationJob,
  fetchGenerationCapabilities,
  getGenerationInstallStatus,
  getGenerationJob,
  installGenerationModel,
  type CreateGenerationJobParams,
} from "../services/generation";
import {
  DEFAULT_INSTALL_POLL_INTERVAL_MS,
  MODELS_QUERY_KEY,
  isAwaitingFirstStatus,
  resolveInstallErrorMessage,
  resolveInstallPhase,
  type InstallState,
  type ModelInstallPhase,
} from "./useModels";

export const DEFAULT_POLL_INTERVAL_MS = 1500;

export type GenerationJobPhase = "idle" | "uploading" | JobStatus;

export interface UseGenerationJobResult {
  phase: GenerationJobPhase;
  job: GenerationJob | undefined;
  errorMessage: string | null;
  submit: (params: CreateGenerationJobParams) => void;
  cancel: () => void;
  reset: () => void;
}

function resolvePhase(
  isUploading: boolean,
  initialStatus: JobStatus | undefined,
  job: GenerationJob | undefined,
): GenerationJobPhase {
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
  job: GenerationJob | undefined,
): string | null {
  if (uploadError instanceof Error) {
    return uploadError.message;
  }
  if (jobError instanceof Error) {
    return jobError.message;
  }
  if (job?.status === "failed") {
    return job.error ?? "The job failed.";
  }
  return null;
}

export function useGenerationJob(
  pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS,
  queue: JobQueueStore = jobQueueStore,
): UseGenerationJobResult {
  const [jobId, setJobId] = useState<string | null>(null);
  const pendingPromptRef = useRef<string>("generation");
  const queryClient = useQueryClient();

  // NOTA (Task 9/10 contract): createGenerationJob resuelve el GenerationJob
  // completo (id/status ya presentes), no un CreateJobResponse -- por eso
  // onSuccess usa data.id en vez de data.jobId.
  const uploadMutation = useMutation({
    mutationFn: createGenerationJob,
    onSuccess: (data) => {
      setJobId(data.id);
      queue.addTrackedJob({
        id: data.id,
        kind: "generation",
        fileName: pendingPromptRef.current,
        createdAt: Date.now(),
      });
    },
  });

  const jobQuery = useQuery({
    queryKey: ["generationJob", jobId],
    queryFn: () => getGenerationJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => (isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : pollIntervalMs),
  });

  function submit(params: CreateGenerationJobParams): void {
    setJobId(null);
    pendingPromptRef.current = params.prompt.slice(0, 60);
    uploadMutation.mutate(params);
  }

  // Best-effort: a 409 (job already finished) needs no surfaced error since the
  // running poll is the source of truth and reconciles the status on refetch.
  function cancel(): void {
    if (jobId === null) {
      return;
    }
    void cancelGenerationJob(jobId)
      .then(() => queryClient.invalidateQueries({ queryKey: ["generationJob", jobId] }))
      .catch(() => undefined);
  }

  function reset(): void {
    setJobId(null);
    uploadMutation.reset();
  }

  return {
    phase: resolvePhase(uploadMutation.isPending, uploadMutation.data?.status, jobQuery.data),
    job: jobQuery.data,
    errorMessage: resolveErrorMessage(uploadMutation.error, jobQuery.error, jobQuery.data),
    submit,
    cancel,
    reset,
  };
}

export function useGenerationCapabilities() {
  return useQuery<GenerationCapabilities>({
    queryKey: ["generationCapabilities"],
    queryFn: fetchGenerationCapabilities,
  });
}

export interface UseGenerationModelInstallResult {
  phase: ModelInstallPhase;
  progressPct: number | null;
  errorMessage: string | null;
  modelId: string | null;
  install: (repoId: string) => void;
  reset: () => void;
}

// Sibling of useModelInstall (../hooks/useModels.ts): same state machine, poll
// mechanics and query invalidation, wired to the generation-model install
// endpoints instead of the upscaler ones. Shares the pure phase/error helpers
// and the "models" query key so both flows refresh the same installed list.
export function useGenerationModelInstall(
  pollIntervalMs: number = DEFAULT_INSTALL_POLL_INTERVAL_MS,
): UseGenerationModelInstallResult {
  const [installId, setInstallId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const startMutation = useMutation({
    mutationFn: installGenerationModel,
    onSuccess: (data) => setInstallId(data.installId),
  });

  const statusQuery = useQuery({
    queryKey: ["generation-model-install", installId],
    queryFn: () => getGenerationInstallStatus(installId as string),
    enabled: installId !== null,
    refetchInterval: (query) =>
      isTerminalInstallStatus(query.state.data?.status ?? "downloading") ? false : pollIntervalMs,
  });

  const installedModelId = statusQuery.data?.status === "installed" ? statusQuery.data.modelId : null;
  useEffect(() => {
    if (installedModelId) {
      queryClient.invalidateQueries({ queryKey: MODELS_QUERY_KEY });
    }
  }, [installedModelId, queryClient]);

  function install(repoId: string): void {
    setInstallId(null);
    startMutation.mutate(repoId);
  }

  function reset(): void {
    setInstallId(null);
    startMutation.reset();
  }

  return {
    phase: resolveInstallPhase(
      startMutation.isPending,
      isAwaitingFirstStatus(installId, statusQuery.data, statusQuery.isError),
      statusQuery.data?.status as InstallState | undefined,
    ),
    progressPct: statusQuery.data?.progressPct ?? null,
    errorMessage: resolveInstallErrorMessage(startMutation.error, statusQuery.error, statusQuery.data),
    modelId: statusQuery.data?.modelId ?? null,
    install,
    reset,
  };
}
