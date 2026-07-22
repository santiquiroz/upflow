import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useCapabilities } from "./useCapabilities";
import * as api from "../lib/api";

function withQueryClient() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe("useCapabilities", () => {
  it("loads levers on mount", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({
      levers: [{ id: "hags", label: "HAGS", status: "ok", detail: "enabled", fixable: false }],
    });

    const { result } = renderHook(() => useCapabilities(), { wrapper: withQueryClient() });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.levers[0].id).toBe("hags");
  });

  it("calls fixLever and tracks which lever is fixing", async () => {
    vi.spyOn(api, "getCapabilities").mockResolvedValue({ levers: [] });
    vi.spyOn(api, "fixLever").mockResolvedValue({
      lever: { id: "hags", label: "HAGS", status: "ok", detail: "fixed", fixable: false },
    });

    const { result } = renderHook(() => useCapabilities(), { wrapper: withQueryClient() });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => {
      result.current.fix("hags");
    });

    await waitFor(() => expect(api.fixLever).toHaveBeenCalledWith("hags"));
  });
});
