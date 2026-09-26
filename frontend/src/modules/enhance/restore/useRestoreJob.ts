import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  DEFAULT_POLL_INTERVAL_MS,
  resolveErrorMessage,
  resolvePhase,
  type ImageJobPhase,
} from "../../../hooks/useImageJob";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { cancelJob, getJob } from "../../../lib/api";
import type { JobResponse } from "../../../lib/apiTypes";
import { jobQueueStore, type JobQueueStore } from "../../../lib/jobQueueStore";
import { isTerminalJobStatus } from "../../../lib/jobStatus";
import { createRestoreJob, type CreateRestoreJobParams } from "../../../services/restore";

export interface RestoreJobRequest {
  params: CreateRestoreJobParams;
  fileName: string;
}

export interface UseRestoreJobResult {
  phase: ImageJobPhase;
  job: JobResponse | undefined;
  errorMessage: string | null;
  submit: (request: RestoreJobRequest) => void;
  cancel: () => void;
  reset: () => void;
}

export interface RestoreJobDeps {
  createJob: (params: CreateRestoreJobParams) => Promise<JobResponse>;
  pollIntervalMs: number;
  queue: JobQueueStore;
}

const DEFAULT_DEPS: RestoreJobDeps = {
  createJob: (params) => createRestoreJob(params),
  pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
  queue: jobQueueStore,
};

export function useRestoreJob(deps: RestoreJobDeps = DEFAULT_DEPS): UseRestoreJobResult {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [jobId, setJobId] = useState<string | null>(null);

  const createMutation = useMutation({
    mutationFn: ({ params }: RestoreJobRequest) => deps.createJob(params),
    onSuccess: (created, { fileName }) => {
      setJobId(created.jobId);
      // Familia image: la cola global y la pagina de tareas lo muestran sin cambios.
      deps.queue.addTrackedJob({ id: created.jobId, kind: "image", fileName, createdAt: Date.now() });
    },
  });

  const jobQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => getJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) =>
      isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : deps.pollIntervalMs,
  });

  function submit(request: RestoreJobRequest): void {
    setJobId(null);
    createMutation.mutate(request);
  }

  function cancel(): void {
    if (jobId === null) {
      return;
    }
    void cancelJob(jobId)
      .then(() => queryClient.invalidateQueries({ queryKey: ["job", jobId] }))
      .catch(() => undefined);
  }

  function reset(): void {
    setJobId(null);
    createMutation.reset();
  }

  return {
    // Con token no se sube nada: mientras el backend crea el job ya esta en cola.
    phase: resolvePhase(false, createMutation.isPending ? "queued" : createMutation.data?.status, jobQuery.data),
    job: jobQuery.data,
    errorMessage: resolveErrorMessage(createMutation.error, jobQuery.error, jobQuery.data, t),
    submit,
    cancel,
    reset,
  };
}
