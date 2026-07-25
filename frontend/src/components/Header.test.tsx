import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AuthProvider } from "../hooks/useAuth";
import * as authService from "../services/auth";
import type { MeResponse } from "../lib/apiTypes";
import { Header } from "./Header";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, getMe: vi.fn(), logout: vi.fn() };
});

function renderHeader() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <AuthProvider>{children}</AuthProvider>
      </QueryClientProvider>
    );
  }
  return render(<Header />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(authService.getMe).mockReset();
  vi.mocked(authService.logout).mockReset();
});

describe("Header", () => {
  it("renders nothing in off mode", async () => {
    vi.mocked(authService.getMe).mockResolvedValue({
      userId: null, username: "local", role: "admin", permissions: [], mustChangePassword: false,
      authMode: "off", quota: { maxConcurrent: 0, maxQueued: 0, maxJobsPerDay: 0, maxGpuSecondsPerDay: 0, usedJobsToday: 0, usedGpuSecondsToday: 0 },
    } as MeResponse);
    const { container } = renderHeader();

    // Header returns null both before and after GET /auth/me resolves in off mode,
    // so the container is always empty — wait on the mocked call instead of DOM state.
    await waitFor(() => expect(authService.getMe).toHaveBeenCalled());
    expect(container.querySelector("button")).toBeNull();
  });

  it("shows the username and logs out on click in multi mode", async () => {
    vi.mocked(authService.getMe).mockResolvedValue({
      userId: "u1", username: "alice", role: "user", permissions: [], mustChangePassword: false,
      authMode: "multi", quota: { maxConcurrent: 1, maxQueued: 5, maxJobsPerDay: 50, maxGpuSecondsPerDay: 3600, usedJobsToday: 0, usedGpuSecondsToday: 0 },
    } as MeResponse);
    vi.mocked(authService.logout).mockResolvedValue({ ok: true });
    renderHeader();

    await waitFor(() => expect(screen.getByText("alice")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /user menu/i }));
    fireEvent.click(screen.getByRole("menuitem", { name: /log out/i }));

    await waitFor(() => expect(authService.logout).toHaveBeenCalled());
  });
});
