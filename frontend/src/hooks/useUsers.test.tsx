import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as usersService from "../services/users";
import { useCreateUser, useUsers } from "./useUsers";

vi.mock("../services/users", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/users")>();
  return { ...actual, listUsers: vi.fn(), createUser: vi.fn() };
});

function wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

afterEach(() => {
  vi.mocked(usersService.listUsers).mockReset();
  vi.mocked(usersService.createUser).mockReset();
});

describe("useUsers", () => {
  it("fetches the users list", async () => {
    vi.mocked(usersService.listUsers).mockResolvedValue({ users: [] });

    const { result } = renderHook(() => useUsers(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.users).toEqual([]);
  });

  it("useCreateUser calls the create service", async () => {
    vi.mocked(usersService.createUser).mockResolvedValue({
      user: { id: "u1", username: "bob", role: "user", disabled: false, mustChangePassword: true, quotaOverrides: {}, createdAt: "now", usedJobsToday: 0, usedGpuSecondsToday: 0 },
      temporaryPassword: "temp123",
    });

    const { result } = renderHook(() => useCreateUser(), { wrapper });
    result.current.mutate({ username: "bob", role: "user" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(usersService.createUser).toHaveBeenCalledWith({ username: "bob", role: "user" });
  });
});
