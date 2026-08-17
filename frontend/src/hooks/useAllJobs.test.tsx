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
  return { ...actual, listJobs: vi.fn(), listVideoJobs: vi.fn(), apiGet: vi.fn() };
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

function emptyEverything() {
  vi.mocked(api.listJobs).mockResolvedValue({ jobs: [] } as never);
  vi.mocked(api.listVideoJobs).mockResolvedValue({ jobs: [] } as never);
  vi.mocked(audioService.listAudioJobs).mockResolvedValue({ jobs: [] } as never);
  vi.mocked(generationService.listGenerationJobs).mockResolvedValue({ jobs: [] } as never);
  vi.mocked(api.apiGet).mockResolvedValue({ jobs: [] } as never);
}

afterEach(() => {
  vi.mocked(api.listJobs).mockReset();
  vi.mocked(api.listVideoJobs).mockReset();
  vi.mocked(api.apiGet).mockReset();
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
    emptyEverything();
    vi.mocked(api.listJobs).mockResolvedValue({
      jobs: [{ jobId: "i1", status: "queued", originalFilename: "a.png", createdAt: "2026-01-01T00:00:01Z", ownerId: "u1", error: null, downloadUrl: null }],
    } as never);
    vi.mocked(audioService.listAudioJobs).mockResolvedValue({
      jobs: [{ id: "a1", status: "completed", originalFilename: "b.wav", createdAt: "2026-01-01T00:00:02Z", ownerId: "u2", error: null, downloadUrl: "/x" }],
    } as never);

    const { result } = renderHook(() => useAllJobsView(true), { wrapper });

    await waitFor(() => expect(result.current).toHaveLength(2));
    expect(result.current[0].id).toBe("a1"); // newer createdAt sorts first
    expect(result.current[0].ownerId).toBe("u2");
    expect(result.current[1].ownerId).toBe("u1");
  });

  // Mientras la vista cubrio solo cuatro familias, un admin veia media app: los
  // trabajos de transcripcion, descarga y 3D de otros usuarios no existian para
  // el aunque el servidor los listara.
  it("tambien trae transcripcion, descarga y 3D, con el ?all=true del admin", async () => {
    emptyEverything();
    vi.mocked(api.apiGet).mockImplementation(((path: string) => {
      if (path.startsWith("/transcribe/jobs")) {
        return Promise.resolve({
          jobs: [{ id: "t1", status: "running", originalFilename: "clase.mp4", createdAt: "2026-01-01T00:00:03Z", ownerId: "u3" }],
        });
      }
      if (path.startsWith("/download/jobs")) {
        return Promise.resolve({
          jobs: [{ id: "d1", status: "completed", mediaTitle: "Un video", createdAt: "2026-01-01T00:00:04Z", ownerId: "u4" }],
        });
      }
      return Promise.resolve({
        jobs: [{ id: "s1", status: "queued", prompt: "un soporte", createdAt: "2026-01-01T00:00:05Z", ownerId: "u5" }],
      });
    }) as never);

    const { result } = renderHook(() => useAllJobsView(true), { wrapper });

    await waitFor(() => expect(result.current).toHaveLength(3));
    expect(result.current.map((entry) => entry.kind)).toEqual(["shape3d", "download", "transcribe"]);
    // Cada familia nombra su trabajo con lo que el usuario reconoce.
    expect(result.current.map((entry) => entry.fileName)).toEqual(["un soporte", "Un video", "clase.mp4"]);
    expect(result.current.map((entry) => entry.ownerId)).toEqual(["u5", "u4", "u3"]);

    const paths = vi.mocked(api.apiGet).mock.calls.map((call) => call[0]);
    expect(paths).toEqual(
      expect.arrayContaining([
        "/transcribe/jobs?all=true",
        "/download/jobs?all=true",
        "/print/generate?all=true",
      ]),
    );
  });

  it("un trabajo sin id no rompe la vista", async () => {
    emptyEverything();
    vi.mocked(api.apiGet).mockResolvedValue({
      jobs: [{ status: "queued", createdAt: "2026-01-01T00:00:01Z" }],
    } as never);

    const { result } = renderHook(() => useAllJobsView(true), { wrapper });

    await waitFor(() => expect(vi.mocked(api.apiGet)).toHaveBeenCalled());
    expect(result.current).toEqual([]);
  });
});
