import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import * as audioService from "../services/audio";
import * as generationService from "../services/generation";
import { useAllJobsView } from "./useAllJobs";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, listJobs: vi.fn(), listVideoJobs: vi.fn() };
});
vi.mock("../services/audio", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/audio")>();
  return { ...actual, listAudioJobs: vi.fn() };
});
vi.mock("../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/generation")>();
  return { ...actual, listGenerationJobs: vi.fn() };
});

function wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

afterEach(() => {
  vi.mocked(api.listJobs).mockReset();
  vi.mocked(api.listVideoJobs).mockReset();
  vi.mocked(audioService.listAudioJobs).mockReset();
  vi.mocked(generationService.listGenerationJobs).mockReset();
});

describe("useAllJobsView", () => {
  it("returns an empty list and does not fetch when disabled", () => {
    const { result } = renderHook(() => useAllJobsView(false), { wrapper });
    expect(result.current).toEqual([]);
    expect(api.listJobs).not.toHaveBeenCalled();
  });

  it("merges entries from all 4 kinds with owner ids, newest first", async () => {
    vi.mocked(api.listJobs).mockResolvedValue({
      jobs: [{ jobId: "i1", status: "queued", originalFilename: "a.png", createdAt: "2026-01-01T00:00:01Z", ownerId: "u1", error: null, downloadUrl: null }],
    } as never);
    vi.mocked(api.listVideoJobs).mockResolvedValue({ jobs: [] } as never);
    vi.mocked(audioService.listAudioJobs).mockResolvedValue({
      jobs: [{ id: "a1", status: "completed", originalFilename: "b.wav", createdAt: "2026-01-01T00:00:02Z", ownerId: "u2", error: null, downloadUrl: "/x" }],
    } as never);
    vi.mocked(generationService.listGenerationJobs).mockResolvedValue({ jobs: [] } as never);

    const { result } = renderHook(() => useAllJobsView(true), { wrapper });

    await waitFor(() => expect(result.current).toHaveLength(2));
    expect(result.current[0].id).toBe("a1"); // newer createdAt sorts first
    expect(result.current[0].ownerId).toBe("u2");
    expect(result.current[1].ownerId).toBe("u1");
  });
});
