import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { AuthProvider } from "./hooks/useAuth";
import * as authService from "./services/auth";
import type { MeResponse } from "./lib/apiTypes";
import { ApiError } from "./lib/api";

vi.mock("./services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./services/auth")>();
  return { ...actual, getMe: vi.fn() };
});

function renderApp() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <AuthProvider>
          <App />
        </AuthProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

const OFF_MODE_ME: MeResponse = {
  userId: null, username: "local", role: "admin", permissions: ["jobs:create", "users:manage"],
  mustChangePassword: false, authMode: "off",
  quota: { maxConcurrent: 0, maxQueued: 0, maxJobsPerDay: 0, maxGpuSecondsPerDay: 0, usedJobsToday: 0, usedGpuSecondsToday: 0 },
};

afterEach(() => {
  vi.mocked(authService.getMe).mockReset();
});

describe("App auth gate", () => {
  it("renders the normal app UI unchanged in off mode", async () => {
    vi.mocked(authService.getMe).mockResolvedValue(OFF_MODE_ME);
    renderApp();

    await waitFor(() => expect(screen.getByRole("heading", { name: /enhance/i })).toBeInTheDocument());
  });

  it("renders LoginPage when GET /auth/me returns not_authenticated", async () => {
    vi.mocked(authService.getMe).mockRejectedValue(new ApiError(401, "not_authenticated"));
    renderApp();

    await waitFor(() => expect(screen.getByRole("button", { name: /ingresar/i })).toBeInTheDocument());
  });

  it("renders SetupPage when GET /auth/me returns setup_required", async () => {
    vi.mocked(authService.getMe).mockRejectedValue(new ApiError(401, "setup_required"));
    renderApp();

    await waitFor(() => expect(screen.getByRole("button", { name: /crear/i })).toBeInTheDocument());
  });

  it("shows the forced password change modal when mustChangePassword is true", async () => {
    vi.mocked(authService.getMe).mockResolvedValue({ ...OFF_MODE_ME, authMode: "multi", mustChangePassword: true });
    renderApp();

    await waitFor(() => expect(screen.getByRole("heading", { name: /cambiá tu contraseña/i })).toBeInTheDocument());
  });
});
