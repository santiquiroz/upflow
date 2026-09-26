import { useState } from "react";
import type { JobResponse } from "../../../lib/apiTypes";
import { jobQueueStore, type JobQueueStore } from "../../../lib/jobQueueStore";
import { uploadSequentially } from "../../../lib/sequentialUploads";
import { createRestoreJob, type CreateRestoreJobParams } from "../../../services/restore";

export interface RestoreBatchDeps {
  createJob: (params: CreateRestoreJobParams) => Promise<JobResponse>;
  queue: JobQueueStore;
}

export interface BatchEntry {
  jobId: string;
  fileName: string;
}

export interface UseRestoreBatchResult {
  submitMany: (paramsList: CreateRestoreJobParams[]) => void;
  pending: number;
  sent: number;
  failed: number;
  entries: BatchEntry[];
}

const DEFAULT_DEPS: RestoreBatchDeps = {
  createJob: (params) => createRestoreJob(params),
  queue: jobQueueStore,
};

function fileNameOf(params: CreateRestoreJobParams): string {
  return "file" in params.source ? params.source.file.name : params.source.token;
}

// A diferencia de submitMany de Image, ninguno reemplaza el resultado en pantalla:
// todos van a la cola global y la pagina de tareas los sigue.
export function useRestoreBatch(deps: RestoreBatchDeps = DEFAULT_DEPS): UseRestoreBatchResult {
  const [pending, setPending] = useState(0);
  const [sent, setSent] = useState(0);
  const [failed, setFailed] = useState(0);
  const [entries, setEntries] = useState<BatchEntry[]>([]);

  async function sendOne(params: CreateRestoreJobParams): Promise<void> {
    const job = await deps.createJob(params);
    const fileName = fileNameOf(params);
    deps.queue.addTrackedJob({ id: job.jobId, kind: "image", fileName, createdAt: Date.now() });
    setEntries((current) => [...current, { jobId: job.jobId, fileName }]);
    setSent((count) => count + 1);
  }

  async function submitMany(paramsList: CreateRestoreJobParams[]): Promise<void> {
    setPending(paramsList.length);
    setSent(0);
    setFailed(0);
    const result = await uploadSequentially(paramsList, sendOne, setPending);
    setFailed(result.failed);
  }

  return {
    submitMany: (paramsList) => void submitMany(paramsList),
    pending,
    sent,
    failed,
    entries,
  };
}
