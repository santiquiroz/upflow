import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as authService from "../services/auth";
import { ApiError } from "../lib/api";
import type { MeResponse } from "../lib/apiTypes";
import { AuthProvider, useAuth } from "./useAuth";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, getMe: vi.fn() };
});

function Wrapper({ children }: { children: ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>{children}</AuthProvider>
    </QueryClientProvider>
  );
}

function Probe() {
  const { me, isLoading, needsSetup, hasPermission } = useAuth();
  if (isLoading) return <span>loading</span>;
  if (needsSetup) return <span>needs-setup</span>;
  return (
    <span>
      {me?.username ?? "none"}:{hasPermission("users:manage") ? "yes" : "no"}
    </span>
  );
}

afterEach(() => {
  vi.mocked(authService.getMe).mockReset();
});

describe("useAuth", () => {
  it("exposes the resolved user and permission check after GET /auth/me succeeds", async () => {
    const meResponse: MeResponse = {
      userId: "u1", username: "alice", role: "admin", permissions: ["users:manage"],
      mustChangePassword: false, authMode: "multi",
      quota: { maxConcurrent: 0, maxQueued: 0, maxJobsPerDay: 0, maxGpuSecondsPerDay: 0, usedJobsToday: 0, usedGpuSecondsToday: 0 },
    };
    vi.mocked(authService.getMe).mockResolvedValue(meResponse);

    render(<Probe />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByText("alice:yes")).toBeInTheDocument());
  });

  it("exposes needsSetup when GET /auth/me fails with setup_required", async () => {
    vi.mocked(authService.getMe).mockRejectedValue(new ApiError(401, "setup_required"));

    render(<Probe />, { wrapper: Wrapper });

    await waitFor(() => expect(screen.getByText("needs-setup")).toBeInTheDocument());
  });

  it("throws when useAuth is used outside AuthProvider", () => {
    function Bare() {
      useAuth();
      return null;
    }
    expect(() => render(<Bare />)).toThrow("useAuth must be used within an AuthProvider");
  });
});
