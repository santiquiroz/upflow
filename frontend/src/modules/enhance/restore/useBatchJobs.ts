import { useQueries } from "@tanstack/react-query";
import { DEFAULT_POLL_INTERVAL_MS } from "../../../hooks/useImageJob";
import { getJob } from "../../../lib/api";
import type { JobResponse } from "../../../lib/apiTypes";
import type { RecomposeFaceChoice, RecomposeResponse } from "../../../lib/restoreApiTypes";
import { isTerminalJobStatus } from "../../../lib/jobStatus";
import { recomposeFaces } from "../../../services/restore";
import type { BatchEntry } from "./useRestoreBatch";

export interface BatchResultDeps {
  fetchJob: (jobId: string) => Promise<JobResponse>;
  pollIntervalMs: number;
  recompose: (jobId: string, faces: Record<number, RecomposeFaceChoice>) => Promise<RecomposeResponse>;
}

export const DEFAULT_BATCH_RESULT_DEPS: BatchResultDeps = {
  fetchJob: getJob,
  pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
  recompose: recomposeFaces,
};

export function useBatchJobs(entries: readonly BatchEntry[], deps: BatchResultDeps): (JobResponse | undefined)[] {
  const results = useQueries({
    queries: entries.map((entry) => ({
      queryKey: ["job", entry.jobId],
      queryFn: () => deps.fetchJob(entry.jobId),
      refetchInterval: (query: { state: { data?: JobResponse } }) =>
        isTerminalJobStatus(query.state.data?.status ?? "queued") ? false : deps.pollIntervalMs,
    })),
  });
  return results.map((result) => result.data);
}
