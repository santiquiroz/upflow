import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import type { CreateInstallResponse, InstallStatusResponse, ModelsResponse } from "../lib/apiTypes";
import { useDeleteModel, useHfSearchResults, useModelInstall } from "./useModels";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return {
    ...actual,
    installModel: vi.fn(),
    getInstallStatus: vi.fn(),
    searchHfModels: vi.fn(),
    deleteModel: vi.fn(),
    getModels: vi.fn(),
  };
});

const POLL_INTERVAL_MS = 10;

function createQueryClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
}

function createWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
}

afterEach(() => {
  vi.mocked(api.installModel).mockReset();
  vi.mocked(api.getInstallStatus).mockReset();
  vi.mocked(api.searchHfModels).mockReset();
  vi.mocked(api.deleteModel).mockReset();
  vi.mocked(api.getModels).mockReset();
});

describe("useModelInstall", () => {
  it("starts idle with no install in flight", () => {
    const { result } = renderHook(() => useModelInstall(POLL_INTERVAL_MS), {
      wrapper: createWrapper(createQueryClient()),
    });

    expect(result.current.phase).toBe("idle");
    expect(result.current.progressPct).toBeNull();
    expect(result.current.errorMessage).toBeNull();
  });

  it("transitions through downloading, validating, converting and stops polling at installed", async () => {
    const createResponse: CreateInstallResponse = {
      installId: "install-1",
      statusUrl: "/api/v1/models/install/install-1",
    };
    vi.mocked(api.installModel).mockResolvedValue(createResponse);
    const baseStatus: InstallStatusResponse = {
      installId: "install-1",
      repoId: "example/anime-2x",
      status: "downloading",
      progressPct: 10,
      modelId: null,
      error: null,
    };
    vi.mocked(api.getInstallStatus)
      .mockResolvedValueOnce({ ...baseStatus, status: "downloading", progressPct: 30 })
      .mockResolvedValueOnce({ ...baseStatus, status: "validating", progressPct: null })
      .mockResolvedValueOnce({ ...baseStatus, status: "converting", progressPct: null })
      .mockResolvedValue({ ...baseStatus, status: "installed", progressPct: null, modelId: "example--anime-2x" });

    const { result } = renderHook(() => useModelInstall(POLL_INTERVAL_MS), {
      wrapper: createWrapper(createQueryClient()),
    });

    act(() => {
      result.current.install("example/anime-2x");
    });

    await waitFor(() => expect(result.current.phase).toBe("installed"));
    expect(result.current.modelId).toBe("example--anime-2x");
    expect(vi.mocked(api.getInstallStatus).mock.calls.length).toBeGreaterThanOrEqual(4);

    const callsAtCompletion = vi.mocked(api.getInstallStatus).mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS * 5));
    expect(vi.mocked(api.getInstallStatus).mock.calls.length).toBe(callsAtCompletion);
  });

  it("stops polling and surfaces the message once the install reaches error", async () => {
    vi.mocked(api.installModel).mockResolvedValue({
      installId: "install-2",
      statusUrl: "/api/v1/models/install/install-2",
    });
    vi.mocked(api.getInstallStatus).mockResolvedValue({
      installId: "install-2",
      repoId: "example/bad-repo",
      status: "error",
      progressPct: null,
      modelId: null,
      error: "Repository does not contain a supported model file",
    });

    const { result } = renderHook(() => useModelInstall(POLL_INTERVAL_MS), {
      wrapper: createWrapper(createQueryClient()),
    });

    act(() => {
      result.current.install("example/bad-repo");
    });

    await waitFor(() => expect(result.current.phase).toBe("error"));
    expect(result.current.errorMessage).toBe("Repository does not contain a supported model file");

    const callsAtCompletion = vi.mocked(api.getInstallStatus).mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS * 5));
    expect(vi.mocked(api.getInstallStatus).mock.calls.length).toBe(callsAtCompletion);
  });

  it("surfaces a rejection from the install request itself without ever polling", async () => {
    vi.mocked(api.installModel).mockRejectedValue(new Error("repo_id must look like 'owner/name'"));

    const { result } = renderHook(() => useModelInstall(POLL_INTERVAL_MS), {
      wrapper: createWrapper(createQueryClient()),
    });

    act(() => {
      result.current.install("not-a-valid-repo-id");
    });

    await waitFor(() => expect(result.current.errorMessage).toBe("repo_id must look like 'owner/name'"));
    expect(api.getInstallStatus).not.toHaveBeenCalled();
  });

  it("stays in 'starting' between the install request resolving and the first status poll", async () => {
    vi.mocked(api.installModel).mockResolvedValue({
      installId: "install-gap",
      statusUrl: "/api/v1/models/install/install-gap",
    });
    // Never resolves: holds the status query in its very first, pending fetch so
    // we can observe the window that used to leak "idle" and re-show Install.
    vi.mocked(api.getInstallStatus).mockReturnValue(new Promise<never>(() => {}));

    const { result } = renderHook(() => useModelInstall(POLL_INTERVAL_MS), {
      wrapper: createWrapper(createQueryClient()),
    });

    act(() => {
      result.current.install("example/anime-2x");
    });

    // Once getInstallStatus has been called the mutation has already resolved
    // (installId is set) yet no status has arrived, so phase must stay "starting".
    await waitFor(() => expect(api.getInstallStatus).toHaveBeenCalled());
    expect(result.current.phase).toBe("starting");
  });

  it("resets back to idle", async () => {
    vi.mocked(api.installModel).mockResolvedValue({
      installId: "install-3",
      statusUrl: "/api/v1/models/install/install-3",
    });
    vi.mocked(api.getInstallStatus).mockResolvedValue({
      installId: "install-3",
      repoId: "example/anime-2x",
      status: "downloading",
      progressPct: 5,
      modelId: null,
      error: null,
    });

    const { result } = renderHook(() => useModelInstall(POLL_INTERVAL_MS), {
      wrapper: createWrapper(createQueryClient()),
    });

    act(() => {
      result.current.install("example/anime-2x");
    });

    await waitFor(() => expect(result.current.phase).toBe("downloading"));

    act(() => {
      result.current.reset();
    });

    expect(result.current.phase).toBe("idle");
    expect(result.current.progressPct).toBeNull();
  });

  it("invalidates the installed models list once the install completes", async () => {
    vi.mocked(api.installModel).mockResolvedValue({
      installId: "install-4",
      statusUrl: "/api/v1/models/install/install-4",
    });
    vi.mocked(api.getInstallStatus).mockResolvedValue({
      installId: "install-4",
      repoId: "example/anime-2x",
      status: "installed",
      progressPct: null,
      modelId: "example--anime-2x",
      error: null,
    });
    vi.mocked(api.getModels).mockResolvedValue({ models: [] } satisfies ModelsResponse);

    const queryClient = createQueryClient();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useModelInstall(POLL_INTERVAL_MS), {
      wrapper: createWrapper(queryClient),
    });

    act(() => {
      result.current.install("example/anime-2x");
    });

    await waitFor(() => expect(result.current.phase).toBe("installed"));
    expect(invalidateSpy).toHaveBeenCalledWith(expect.objectContaining({ queryKey: ["models"] }));
  });
});

describe("useHfSearchResults", () => {
  it("does not query when the search term is blank", () => {
    renderHook(() => useHfSearchResults("   "), { wrapper: createWrapper(createQueryClient()) });

    expect(api.searchHfModels).not.toHaveBeenCalled();
  });

  it("queries Hugging Face once a non-blank search term is provided", async () => {
    vi.mocked(api.searchHfModels).mockResolvedValue({ results: [] });

    const { result } = renderHook(() => useHfSearchResults("anime"), {
      wrapper: createWrapper(createQueryClient()),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(api.searchHfModels).toHaveBeenCalledWith("anime");
  });
});

describe("useDeleteModel", () => {
  it("invalidates the installed models list after a successful delete", async () => {
    vi.mocked(api.deleteModel).mockResolvedValue(undefined);
    const queryClient = createQueryClient();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");

    const { result } = renderHook(() => useDeleteModel(), { wrapper: createWrapper(queryClient) });

    act(() => {
      result.current.mutate("custom-anime-2x");
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(vi.mocked(api.deleteModel).mock.calls[0]?.[0]).toBe("custom-anime-2x");
    expect(invalidateSpy).toHaveBeenCalledWith(expect.objectContaining({ queryKey: ["models"] }));
  });
});
