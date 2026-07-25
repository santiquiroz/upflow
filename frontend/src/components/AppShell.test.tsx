import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { AppShell } from "./AppShell";
import { AuthProvider } from "../hooks/useAuth";
import * as authService from "../services/auth";

vi.mock("../services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/auth")>();
  return { ...actual, getMe: vi.fn() };
});

vi.mocked(authService.getMe).mockResolvedValue({
  userId: null, username: "local", role: "admin", permissions: [], mustChangePassword: false,
  authMode: "off", quota: { maxConcurrent: 0, maxQueued: 0, maxJobsPerDay: 0, maxGpuSecondsPerDay: 0, usedJobsToday: 0, usedGpuSecondsToday: 0 },
});

function renderAt(path: string) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <AuthProvider>
          <AppShell>
            <div>content</div>
          </AppShell>
        </AuthProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("AppShell", () => {
  it("renders all four nav entries with visible labels", () => {
    renderAt("/");

    expect(screen.getByRole("link", { name: "Enhance" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Models" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Realtime" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Settings" })).toBeInTheDocument();
  });

  it("marks only the entry matching the current route as active", () => {
    renderAt("/models");

    expect(screen.getByRole("link", { name: "Models" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Enhance" })).not.toHaveAttribute("aria-current");
    expect(screen.getByRole("link", { name: "Realtime" })).not.toHaveAttribute("aria-current");
    expect(screen.getByRole("link", { name: "Settings" })).not.toHaveAttribute("aria-current");
  });

  it("highlights Enhance as active on the root route", () => {
    renderAt("/");

    expect(screen.getByRole("link", { name: "Enhance" })).toHaveAttribute("aria-current", "page");
  });

  it("renders the page content passed as children", () => {
    renderAt("/");

    expect(screen.getByText("content")).toBeInTheDocument();
  });

  it("renders the job queue panel with its empty state", () => {
    renderAt("/");

    expect(screen.getByRole("heading", { name: "Job Queue" })).toBeInTheDocument();
    expect(screen.getByText(/no active jobs/i)).toBeInTheDocument();
  });
});
