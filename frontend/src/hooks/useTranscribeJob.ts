import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import type {
  AsrInstallStatus,
  JobStatus,
  ModelResponse,
  ModelSearchResponse,
  TranscribeJob,
} from "../lib/apiTypes";
import { isTerminalInstallStatus } from "../lib/installStatus";
import { isTerminalJobStatus } from "../lib/jobStatus";
import {
  cancelTranscribeJob,
  createTranscribeJob,
  fetchInstalledAsrModels,
  fetchTranscribeDevices,
  getAsrInstallStatus,
  getTranscribeJob,
  installAsrModel,
  searchAsrModels,
  type CreateTranscribeJobParams,
} from "../services/transcribe";

export const DEFAULT_TRANSCRIBE_POLL_INTERVAL_MS = 1500;
export const DEFAULT_ASR_INSTALL_POLL_INTERVAL_MS = 1500;
export const ASR_MODELS_QUERY_KEY = ["asr-models"] as const;

export type TranscribeJobPhase = "idle" | "uploading" | JobStatus;
export type AsrInstallPhase =
  | "idle"
  | "starting"
  | AsrInstallStatus;

export interface UseTranscribeJobResult {
  phase: TranscribeJobPhase;
  job: TranscribeJob | undefined;
  errorMessage: string | null;
  // `null` mientras no se este subiendo, o cuando el total no es computable:
  // dibujar un porcentaje inventado seria mentir sobre lo que falta.
  uploadPercent: number | null;
  submit: (params: CreateTranscribeJobParams) => void;
  cancel: () => void;
  reset: () => void;
}

function resolveJobPhase(
  isUploading: boolean,
  initialStatus: JobStatus | undefined,
  job: TranscribeJob | undefined,
): TranscribeJobPhase {
  if (isUploading) {
    return "uploading";
  }
  if (job) {
    return job.status;
  }
  return initialStatus ?? "idle";
}

function resolveJobError(
  uploadError: unknown,
  pollError: unknown,
  job: TranscribeJob | undefined,
): string | null {
  if (uploadError instanceof Error) {
    return uploadError.message;
  }
  if (pollError instanceof Error) {
    return pollError.message;
  }
  return job?.status === "failed" ? (job.error ?? null) : null;
}

export function useTranscribeJob(
  pollIntervalMs: number = DEFAULT_TRANSCRIBE_POLL_INTERVAL_MS,
): UseTranscribeJobResult {
  const [jobId, setJobId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const createMutation = useMutation({
    mutationFn: (params: Parameters<typeof createTranscribeJob>[0]) =>
      createTranscribeJob(params, { onProgress: setUploadPercent }),
    onSuccess: (response) => setJobId(response.jobId),
  });

  const jobQuery = useQuery({
    queryKey: ["transcribe-job", jobId],
    queryFn: () => getTranscribeJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) =>
      isTerminalJobStatus(query.state.data?.status ?? "queued")
        ? false
        : pollIntervalMs,
  });

  function submit(params: CreateTranscribeJobParams): void {
    setJobId(null);
    createMutation.mutate(params);
  }

  function cancel(): void {
    if (jobId === null) {
      return;
    }
    void cancelTranscribeJob(jobId)
      .then(() =>
        queryClient.invalidateQueries({
          queryKey: ["transcribe-job", jobId],
        }),
      )
      .catch(() => undefined);
  }

  function reset(): void {
    setJobId(null);
    createMutation.reset();
  }

  return {
    uploadPercent,
    phase: resolveJobPhase(
      createMutation.isPending,
      createMutation.data?.status,
      jobQuery.data,
    ),
    job: jobQuery.data,
    errorMessage: resolveJobError(
      createMutation.error,
      jobQuery.error,
      jobQuery.data,
    ),
    submit,
    cancel,
    reset,
  };
}

export function useInstalledAsrModels() {
  return useQuery<ModelResponse[]>({
    queryKey: ASR_MODELS_QUERY_KEY,
    queryFn: fetchInstalledAsrModels,
  });
}

export function useTranscribeDevices() {
  return useQuery({
    queryKey: ["transcribe-devices"],
    queryFn: fetchTranscribeDevices,
  });
}

export function useAsrModelSearchResults(query: string) {
  const trimmed = query.trim();
  return useQuery<ModelSearchResponse>({
    queryKey: ["asr-model-search", trimmed],
    queryFn: () => searchAsrModels(trimmed),
  });
}

export interface UseAsrModelInstallResult {
  phase: AsrInstallPhase;
  progressPct: number | null;
  errorMessage: string | null;
  modelId: string | null;
  install: (repoId: string) => void;
  reset: () => void;
}

export function useAsrModelInstall(
  pollIntervalMs: number = DEFAULT_ASR_INSTALL_POLL_INTERVAL_MS,
): UseAsrModelInstallResult {
  const [installId, setInstallId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const startMutation = useMutation({
    mutationFn: installAsrModel,
    onSuccess: (response) => setInstallId(response.installId),
  });

  const statusQuery = useQuery({
    queryKey: ["asr-model-install", installId],
    queryFn: () => getAsrInstallStatus(installId as string),
    enabled: installId !== null,
    refetchInterval: (query) =>
      isTerminalInstallStatus(query.state.data?.status ?? "downloading")
        ? false
        : pollIntervalMs,
  });

  const installedModelId =
    statusQuery.data?.status === "installed"
      ? statusQuery.data.modelId
      : null;

  useEffect(() => {
    if (installedModelId) {
      void queryClient.invalidateQueries({ queryKey: ASR_MODELS_QUERY_KEY });
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

  let phase: AsrInstallPhase = "idle";
  if (
    startMutation.isPending ||
    (installId !== null &&
      statusQuery.data === undefined &&
      !statusQuery.isError)
  ) {
    phase = "starting";
  } else if (statusQuery.data) {
    phase = statusQuery.data.status;
  } else if (statusQuery.isError) {
    phase = "error";
  }

  const startError =
    startMutation.error instanceof Error ? startMutation.error.message : null;
  const statusError =
    statusQuery.error instanceof Error ? statusQuery.error.message : null;
  const jobError =
    statusQuery.data?.status === "error"
      ? (statusQuery.data.error ?? null)
      : null;

  return {
    phase,
    progressPct: statusQuery.data?.progressPct ?? null,
    errorMessage: startError ?? statusError ?? jobError,
    modelId: statusQuery.data?.modelId ?? null,
    install,
    reset,
  };
}
