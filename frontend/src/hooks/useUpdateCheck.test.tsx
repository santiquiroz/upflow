import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { UpdateCheck } from "../lib/apiTypes";
import * as updateService from "../services/update";
import { useUpdateCheck } from "./useUpdateCheck";

vi.mock("../services/update");

function createWrapper() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
}

afterEach(() => {
  vi.mocked(updateService.fetchUpdateCheck).mockReset();
});

describe("useUpdateCheck", () => {
  it("maps the update-check response returned by the service", async () => {
    const status: UpdateCheck = {
      currentVersion: "0.1.0",
      latestVersion: "0.2.0",
      updateAvailable: true,
      releaseUrl: "https://github.com/santiquiroz/upflow/releases/tag/v0.2.0",
      publishedAt: "2026-07-16T10:00:00Z",
      checkedAt: "2026-07-16T10:00:00Z",
      error: null,
    };
    vi.mocked(updateService.fetchUpdateCheck).mockResolvedValue(status);

    const { result } = renderHook(() => useUpdateCheck(), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(status);
    expect(updateService.fetchUpdateCheck).toHaveBeenCalledTimes(1);
  });
});
