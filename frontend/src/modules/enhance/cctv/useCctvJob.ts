import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { DEFAULT_POLL_INTERVAL_MS, type VideoJobPhase } from "../../../hooks/useVideoJob";
import { cancelVideoJob, getVideoJob } from "../../../lib/api";
import type { VideoJobResponse } from "../../../lib/apiTypes";
import { jobQueueStore, type JobQueueStore } from "../../../lib/jobQueueStore";
import { isTerminalJobStatus } from "../../../lib/jobStatus";
import { createCctvJob, type CctvJobRequest } from "../../../services/cctv";
import { errorInfoOf, type CctvErrorInfo } from "./cctvText";

export interface CctvJobState {
  phase: VideoJobPhase;
  job: VideoJobResponse | undefined;
  error: CctvErrorInfo | null;
  submit: (request: CctvJobRequest) => void;
  cancel: () => void;
}

export function cctvJobPhase(isCreating: boolean, job: VideoJobResponse | undefined): VideoJobPhase {
  if (isCreating) {
    return "queued";
  }
  return job?.status ?? "idle";
}

export function useCctvJob(
  pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS,
  queue: JobQueueStore = jobQueueStore,
): CctvJobState {
  const [jobId, setJobId] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const creation = useMutation({
    mutationFn: createCctvJob,
    onSuccess: (created) => {
      setJobId(created.jobId);
      queue.addTrackedJob({
        id: created.jobId,
        kind: "video",
        fileName: created.originalFilename,
        createdAt: Date.now(),
      });
    },
  });
  const jobQuery = useQuery({
    queryKey: ["videoJob", jobId],
    queryFn: () => getVideoJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => (isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : pollIntervalMs),
  });
  const job = jobQuery.data ?? creation.data;

  function submit(request: CctvJobRequest): void {
    setJobId(null);
    creation.mutate(request);
  }

  // Igual que en useVideoJob: un 409 (ya termino) no es un error, el sondeo reconcilia.
  function cancel(): void {
    if (jobId === null) {
      return;
    }
    void cancelVideoJob(jobId)
      .then(() => queryClient.invalidateQueries({ queryKey: ["videoJob", jobId] }))
      .catch(() => undefined);
  }

  return {
    phase: cctvJobPhase(creation.isPending, job),
    job,
    error: errorInfoOf(creation.error) ?? errorInfoOf(jobQuery.error),
    submit,
    cancel,
  };
}
