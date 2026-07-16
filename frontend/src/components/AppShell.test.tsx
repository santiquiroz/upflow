import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppShell } from "./AppShell";

function renderAt(path: string) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <AppShell>
          <div>content</div>
        </AppShell>
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
