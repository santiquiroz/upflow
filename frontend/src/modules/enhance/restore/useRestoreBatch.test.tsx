import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { JobResponse } from "../../../lib/apiTypes";
import { createJobQueueStore } from "../../../lib/jobQueueStore";
import type { CreateRestoreJobParams } from "../../../services/restore";
import { useRestoreBatch, type RestoreBatchDeps } from "./useRestoreBatch";

function params(name: string): CreateRestoreJobParams {
  return {
    source: { file: new File(["x"], name, { type: "image/jpeg" }) },
    steps: ["repair"],
    options: {},
    scale: 1,
    modelId: null,
    device: null,
    outputFormat: "png",
  };
}

function created(jobId: string): JobResponse {
  return { jobId, status: "queued" } as JobResponse;
}

function renderBatch(createJob: RestoreBatchDeps["createJob"]) {
  const deps: RestoreBatchDeps = { createJob, queue: createJobQueueStore() };
  return { deps, ...renderHook(() => useRestoreBatch(deps)) };
}

describe("useRestoreBatch", () => {
  it("sends the photos one by one and adds each job to the global queue", async () => {
    const createJob = vi.fn().mockResolvedValueOnce(created("job-a")).mockResolvedValueOnce(created("job-b"));
    const { result, deps } = renderBatch(createJob);

    act(() => result.current.submitMany([params("a.jpg"), params("b.jpg")]));

    await waitFor(() => expect(result.current.sent).toBe(2));
    expect(createJob).toHaveBeenCalledTimes(2);
    // La cola global muestra primero lo mas nuevo.
    expect(deps.queue.getSnapshot().map((job) => [job.id, job.kind, job.fileName])).toEqual([
      ["job-b", "image", "b.jpg"],
      ["job-a", "image", "a.jpg"],
    ]);
    expect(result.current.pending).toBe(0);
    expect(result.current.failed).toBe(0);
    expect(result.current.entries).toEqual([
      { jobId: "job-a", fileName: "a.jpg" },
      { jobId: "job-b", fileName: "b.jpg" },
    ]);
  });

  it("keeps the photos of earlier rounds so each result can still be reviewed", async () => {
    const createJob = vi.fn().mockResolvedValueOnce(created("job-a")).mockResolvedValueOnce(created("job-b"));
    const { result } = renderBatch(createJob);

    act(() => result.current.submitMany([params("a.jpg")]));
    await waitFor(() => expect(result.current.sent).toBe(1));
    act(() => result.current.submitMany([params("b.jpg")]));

    await waitFor(() => expect(result.current.entries).toHaveLength(2));
    expect(result.current.sent).toBe(1);
  });

  it("keeps going when a photo is refused and counts it", async () => {
    const createJob = vi.fn().mockRejectedValueOnce(new Error("Queue full")).mockResolvedValueOnce(created("job-b"));
    const { result } = renderBatch(createJob);

    act(() => result.current.submitMany([params("a.jpg"), params("b.jpg")]));

    await waitFor(() => expect(result.current.sent).toBe(1));
    expect(result.current.failed).toBe(1);
    expect(result.current.entries.map((entry) => entry.fileName)).toEqual(["b.jpg"]);
  });

  it("reports what is still waiting to be sent", async () => {
    let release: (job: JobResponse) => void = () => undefined;
    const createJob = vi.fn(() => new Promise<JobResponse>((resolve) => (release = resolve)));
    const { result } = renderBatch(createJob);

    act(() => result.current.submitMany([params("a.jpg"), params("b.jpg")]));

    expect(result.current.pending).toBe(2);
    await act(async () => release(created("job-a")));
    await waitFor(() => expect(result.current.pending).toBe(1));
  });
});
