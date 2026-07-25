import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as usersService from "../services/users";
import { UsersPage } from "./UsersPage";

vi.mock("../services/users", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/users")>();
  return { ...actual, listUsers: vi.fn(), createUser: vi.fn(), updateUser: vi.fn(), getUserJobs: vi.fn() };
});

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<UsersPage />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(usersService.listUsers).mockReset();
});

describe("UsersPage", () => {
  it("renders the users table with existing users", async () => {
    vi.mocked(usersService.listUsers).mockResolvedValue({
      users: [
        { id: "u1", username: "admin", role: "admin", disabled: false, mustChangePassword: false, quotaOverrides: {}, createdAt: "2026-01-01", usedJobsToday: 2, usedGpuSecondsToday: 30 },
      ],
    });

    renderPage();

    expect(await screen.findByText("admin")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Users", level: 1 })).toBeInTheDocument();
  });
});
