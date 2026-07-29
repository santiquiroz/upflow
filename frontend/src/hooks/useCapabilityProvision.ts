import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import type { ProvisionJob } from "../lib/apiTypes";
import { getProvisionStatus, provisionCapability } from "../services/capabilities";

export const PROVISION_POLL_INTERVAL_MS = 2000;

function isFinished(job: ProvisionJob | undefined): boolean {
  return job?.status === "done" || job?.status === "error";
}

export interface UseCapabilityProvisionResult {
  job: ProvisionJob | undefined;
  capabilityId: string | null;
  errorMessage: string | null;
  isBusy: boolean;
  provision: (capabilityId: string) => void;
  reset: () => void;
}

export function useCapabilityProvision(): UseCapabilityProvisionResult {
  const queryClient = useQueryClient();
  const [capabilityId, setCapabilityId] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const start = useMutation({
    mutationFn: provisionCapability,
    onSuccess: (job) => setJobId(job.jobId),
  });

  const status = useQuery({
    queryKey: ["capability-provision", jobId],
    queryFn: () => getProvisionStatus(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) =>
      isFinished(query.state.data) ? false : PROVISION_POLL_INTERVAL_MS,
  });

  // Al terminar bien, el arbol tiene que volver a resolverse: su status sale de
  // mirar el disco, y el disco recien cambio. Va en un efecto y no en el cuerpo
  // del render, donde invalidar dispararia otro render y podria ciclar.
  const finishedWell = status.data?.status === "done";
  useEffect(() => {
    if (finishedWell) {
      queryClient.invalidateQueries({ queryKey: ["capability-tree"] });
    }
  }, [finishedWell, queryClient]);

  return {
    job: status.data,
    capabilityId,
    errorMessage: resolveErrorMessage(start.error, status.error, status.data),
    isBusy: start.isPending || (jobId !== null && !isFinished(status.data)),
    provision: (id: string) => {
      setCapabilityId(id);
      setJobId(null);
      start.mutate(id);
    },
    reset: () => {
      setCapabilityId(null);
      setJobId(null);
      start.reset();
    },
  };
}

function resolveErrorMessage(
  startError: unknown,
  statusError: unknown,
  job: ProvisionJob | undefined,
): string | null {
  if (startError instanceof Error) {
    return startError.message;
  }
  if (statusError instanceof Error) {
    return statusError.message;
  }
  // El job puede terminar en error sin que ninguna request falle: el script de
  // descarga se murio, no la API.
  return job?.status === "error" ? (job.error ?? "La descarga fallo.") : null;
}
