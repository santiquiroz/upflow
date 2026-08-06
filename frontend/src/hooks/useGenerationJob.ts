import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "../i18n/LocaleProvider";
import { useEffect, useRef, useState } from "react";
import type {
  ConversionStatusResponse,
  GenerationCapabilities,
  GenerationJob,
  JobStatus,
  ModelSearchResponse,
  Precision,
  PreflightResponse,
  VideoGenerationCapabilities,
} from "../lib/apiTypes";
import { isTerminalInstallStatus } from "../lib/installStatus";
import { isTerminalJobStatus } from "../lib/jobStatus";
import { jobQueueStore, type JobQueueStore } from "../lib/jobQueueStore";
import {
  cancelGenerationJob,
  createGenerationJob,
  fetchGenerationCapabilities,
  fetchVideoGenerationCapabilities,
  cancelConversion,
  fetchActiveConversions,
  getConversionStatus,
  getGenerationInstallStatus,
  getGenerationJob,
  installGenerationModel,
  preflightGenerationModel,
  searchGenerationModels,
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

export function useGenerationJob(
  pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS,
  queue: JobQueueStore = jobQueueStore,
): UseGenerationJobResult {
  const { t } = useTranslation();
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
    errorMessage: resolveErrorMessage(uploadMutation.error, jobQuery.error, jobQuery.data, t),
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

export function useVideoGenerationCapabilities() {
  return useQuery<VideoGenerationCapabilities>({
    queryKey: ["videoGenerationCapabilities"],
    queryFn: fetchVideoGenerationCapabilities,
  });
}

export function useGenerationHfSearchResults(query: string) {
  const trimmed = query.trim();
  return useQuery<ModelSearchResponse>({
    queryKey: ["generation-hf-search", trimmed],
    queryFn: () => searchGenerationModels(trimmed),
  });
}

export function useGenerationModelPreflight(repoId: string, enabled: boolean) {
  return useQuery<PreflightResponse>({
    queryKey: ["generation-model-preflight", repoId],
    queryFn: () => preflightGenerationModel(repoId),
    enabled,
    staleTime: 5 * 60 * 1000,
  });
}

export interface UseGenerationModelInstallResult {
  phase: ModelInstallPhase;
  progressPct: number | null;
  stageLabel: string | null;
  errorMessage: string | null;
  modelId: string | null;
  install: (repoId: string, precision?: Precision, checkpointPath?: string) => void;
  /** `null` cuando no hay una conversión en curso que se pueda cortar. */
  cancelConversion: (() => void) | null;
  reset: () => void;
}

function resolveConversionPhase(
  conversionId: string | null,
  conversionStatus: JobStatus | undefined,
  conversionIsError: boolean,
): ModelInstallPhase | null {
  if (conversionId === null) {
    return null;
  }
  if (conversionIsError || conversionStatus === "failed" || conversionStatus === "cancelled") {
    return "error";
  }
  if (conversionStatus === "completed") {
    return "installed";
  }
  return "converting";
}

function resolveConversionErrorMessage(
  queryError: unknown,
  conversion: ConversionStatusResponse | undefined,
  t: (key: string) => string,
): string | null {
  if (queryError instanceof Error) {
    return queryError.message;
  }
  if (conversion?.status === "failed") {
    return conversion.error ?? t("generate.conversion.failed");
  }
  if (conversion?.status === "cancelled") {
    return conversion.error ?? t("generate.conversion.cancelled");
  }
  return null;
}

// Sibling of useModelInstall (../hooks/useModels.ts): same state machine, poll
// mechanics and query invalidation, wired to the generation-model install
// endpoints instead of the upscaler ones. Shares the pure phase/error helpers
// and the "models" query key so both flows refresh the same installed list.
export function useGenerationModelInstall(
  pollIntervalMs: number = DEFAULT_INSTALL_POLL_INTERVAL_MS,
): UseGenerationModelInstallResult {
  const { t } = useTranslation();
  const [installId, setInstallId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const startMutation = useMutation({
    mutationFn: ({
      repoId,
      precision,
      checkpointPath,
    }: {
      repoId: string;
      precision?: Precision;
      checkpointPath?: string;
    }) =>
      checkpointPath
        ? installGenerationModel(repoId, precision, checkpointPath)
        : installGenerationModel(repoId, precision),
    onSuccess: (data) => setInstallId(data.installId),
  });

  const statusQuery = useQuery({
    queryKey: ["generation-model-install", installId],
    queryFn: () => getGenerationInstallStatus(installId as string),
    enabled: installId !== null,
    refetchInterval: (query) => {
      const status = query.state.data;
      return status?.conversionId != null ||
        isTerminalInstallStatus(status?.status ?? "downloading")
        ? false
        : pollIntervalMs;
    },
  });

  // Las que siguen corriendo en el servidor. Se consultan SIEMPRE, no solo
  // cuando hay una instalacion en curso en esta pantalla: es lo que permite
  // re-enganchar la barra despues de salir de la seccion y volver.
  const activasQuery = useQuery({
    queryKey: ["generation-active-conversions"],
    queryFn: fetchActiveConversions,
    refetchInterval: (query) => ((query.state.data?.length ?? 0) > 0 ? pollIntervalMs : false),
  });

  const conversionId =
    statusQuery.data?.status === "converting"
      ? (statusQuery.data.conversionId ?? null)
      : // Sin instalacion propia, se adopta la que ya estaba corriendo.
        (activasQuery.data?.[0]?.conversionId ?? null);

  const conversionQuery = useQuery({
    queryKey: ["generation-model-conversion", conversionId],
    queryFn: () => getConversionStatus(conversionId as string),
    enabled: conversionId !== null,
    refetchInterval: (query) =>
      isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : pollIntervalMs,
  });

  const installedModelId =
    conversionQuery.data?.status === "completed"
      ? conversionQuery.data.modelId
      : statusQuery.data?.status === "installed"
        ? statusQuery.data.modelId
        : null;
  useEffect(() => {
    if (installedModelId) {
      queryClient.invalidateQueries({ queryKey: MODELS_QUERY_KEY });
    }
  }, [installedModelId, queryClient]);

  function install(repoId: string, precision?: Precision, checkpointPath?: string): void {
    setInstallId(null);
    startMutation.mutate({ repoId, precision, checkpointPath });
  }

  const cancelMutation = useMutation({
    mutationFn: cancelConversion,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["generation-active-conversions"] });
    },
  });

  // Convertir un SDXL tarda cerca de media hora y ocupa la maquina entera. Sin
  // esto, equivocarse de modelo solo se arregla cerrando la app.
  function cancelConversionInFlight(): void {
    if (conversionId !== null) {
      cancelMutation.mutate(conversionId);
    }
  }

  function reset(): void {
    setInstallId(null);
    startMutation.reset();
  }

  const phase =
    resolveConversionPhase(
      conversionId,
      conversionQuery.data?.status,
      conversionQuery.isError,
    ) ??
    resolveInstallPhase(
      startMutation.isPending,
      isAwaitingFirstStatus(installId, statusQuery.data, statusQuery.isError),
      statusQuery.data?.status as InstallState | undefined,
    );
  const stageLabel =
    phase === "converting"
      ? (conversionQuery.data?.stages?.find((stage) => stage.status === "active")?.label ?? null)
      : null;

  return {
    phase,
    progressPct: conversionQuery.data?.progressPct ?? statusQuery.data?.progressPct ?? null,
    stageLabel,
    errorMessage:
      resolveInstallErrorMessage(startMutation.error, statusQuery.error, statusQuery.data, t) ??
      resolveConversionErrorMessage(conversionQuery.error, conversionQuery.data, t),
    modelId: conversionQuery.data?.modelId ?? statusQuery.data?.modelId ?? null,
    install,
    // Solo se puede cortar una conversión; la descarga previa es corta y se
    // ofrece cancelarla sería prometer algo que no cambia nada.
    cancelConversion: conversionId !== null ? cancelConversionInFlight : null,
    reset,
  };
}
