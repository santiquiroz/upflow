import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../../lib/api";
import type { JobResponse } from "../../../lib/apiTypes";
import { createJobQueueStore } from "../../../lib/jobQueueStore";
import type { CreateRestoreJobParams } from "../../../services/restore";
import { useRestoreJob, type RestoreJobDeps } from "./useRestoreJob";

vi.mock("../../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../lib/api")>();
  return { ...actual, getJob: vi.fn(), cancelJob: vi.fn() };
});

const PARAMS: CreateRestoreJobParams = {
  source: { token: "tok-1" },
  steps: ["repair"],
  options: { preset: "gentle" },
  scale: 1,
  modelId: null,
  device: null,
  outputFormat: "png",
};

function job(status: JobResponse["status"]): JobResponse {
  return { jobId: "job-1", status } as JobResponse;
}

function renderJob(overrides: Partial<RestoreJobDeps> = {}) {
  const deps: RestoreJobDeps = {
    createJob: vi.fn().mockResolvedValue(job("queued")),
    pollIntervalMs: 10,
    queue: createJobQueueStore(),
    ...overrides,
  };
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return { deps, ...renderHook(() => useRestoreJob(deps), { wrapper: Wrapper }) };
}

afterEach(() => {
  vi.mocked(api.getJob).mockReset();
  vi.mocked(api.cancelJob).mockReset();
});

describe("useRestoreJob", () => {
  it("starts idle", () => {
    const { result } = renderJob();

    expect(result.current.phase).toBe("idle");
    expect(result.current.errorMessage).toBeNull();
  });

  it("creates the job, tracks it in the global queue and follows its status", async () => {
    vi.mocked(api.getJob).mockResolvedValue(job("running"));
    const { result, deps } = renderJob();

    act(() => result.current.submit({ params: PARAMS, fileName: "grandma.jpg" }));

    await waitFor(() => expect(result.current.phase).toBe("running"));
    expect(deps.createJob).toHaveBeenCalledWith(PARAMS);
    expect(deps.queue.getSnapshot()).toEqual([expect.objectContaining({ id: "job-1", kind: "image", fileName: "grandma.jpg" })]);
  });

  it("shows the server's reason when the job can't be created", async () => {
    const { result } = renderJob({ createJob: vi.fn().mockRejectedValue(new Error("Missing restore-core pack")) });

    act(() => result.current.submit({ params: PARAMS, fileName: "grandma.jpg" }));

    await waitFor(() => expect(result.current.errorMessage).toBe("Missing restore-core pack"));
    expect(result.current.phase).toBe("idle");
  });

  it("cancels the running job", async () => {
    vi.mocked(api.getJob).mockResolvedValue(job("running"));
    vi.mocked(api.cancelJob).mockResolvedValue(job("cancelled"));
    const { result } = renderJob();
    act(() => result.current.submit({ params: PARAMS, fileName: "grandma.jpg" }));
    await waitFor(() => expect(result.current.phase).toBe("running"));

    act(() => result.current.cancel());

    expect(api.cancelJob).toHaveBeenCalledWith("job-1");
  });
});
