import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { deleteModel, getInstallStatus, getModels, installModel, searchHfModels } from "../lib/api";
import type { InstallStatusResponse, ModelsResponse, ModelSearchResponse } from "../lib/apiTypes";
import { isTerminalInstallStatus } from "../lib/installStatus";

export const DEFAULT_INSTALL_POLL_INTERVAL_MS = 1500;

export const MODELS_QUERY_KEY = ["models"] as const;

export type InstallState = "downloading" | "validating" | "converting" | "installed" | "error";
export type ModelInstallPhase = "idle" | "starting" | InstallState;

export interface UseModelInstallResult {
  phase: ModelInstallPhase;
  progressPct: number | null;
  errorMessage: string | null;
  modelId: string | null;
  install: (repoId: string) => void;
  reset: () => void;
}

// True in the window between installModel resolving (installId set) and the
// first status poll returning: without this the phase would briefly read "idle"
// and the card would re-show "Install", letting a second click fire a duplicate
// install. A status-fetch error clears it so the spinner never sticks forever.
export function isAwaitingFirstStatus(
  installId: string | null,
  statusData: InstallStatusResponse | undefined,
  statusIsError: boolean,
): boolean {
  return installId !== null && statusData === undefined && !statusIsError;
}

export function resolveInstallPhase(
  isStarting: boolean,
  isAwaitingStatus: boolean,
  status: InstallState | undefined,
): ModelInstallPhase {
  if (isStarting || isAwaitingStatus) {
    return "starting";
  }
  return status ?? "idle";
}

export function resolveInstallErrorMessage(
  startError: unknown,
  statusError: unknown,
  statusData: InstallStatusResponse | undefined,
): string | null {
  if (startError instanceof Error) {
    return startError.message;
  }
  if (statusError instanceof Error) {
    return statusError.message;
  }
  if (statusData?.status === "error") {
    return statusData.error ?? "The install failed.";
  }
  return null;
}

export function useModelInstall(pollIntervalMs: number = DEFAULT_INSTALL_POLL_INTERVAL_MS): UseModelInstallResult {
  const [installId, setInstallId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const startMutation = useMutation({
    mutationFn: installModel,
    onSuccess: (data) => setInstallId(data.installId),
  });

  const statusQuery = useQuery({
    queryKey: ["model-install", installId],
    queryFn: () => getInstallStatus(installId as string),
    enabled: installId !== null,
    refetchInterval: (query) => (isTerminalInstallStatus(query.state.data?.status ?? "downloading") ? false : pollIntervalMs),
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

export function useInstalledModels() {
  return useQuery<ModelsResponse>({ queryKey: MODELS_QUERY_KEY, queryFn: getModels });
}

export function useHfSearchResults(query: string) {
  const trimmed = query.trim();
  return useQuery<ModelSearchResponse>({
    queryKey: ["hf-search", trimmed],
    queryFn: () => searchHfModels(trimmed),
    enabled: trimmed.length > 0,
  });
}

export function useDeleteModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: MODELS_QUERY_KEY });
    },
  });
}
