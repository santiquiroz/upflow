import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
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

  it("highlights Tasks as active on the root route", () => {
    // La raiz paso a ser el selector de tareas; Enhance vive en /enhance.
    renderAt("/");

    expect(screen.getByRole("link", { name: "Tasks" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Enhance" })).not.toHaveAttribute("aria-current");
  });

  it("highlights Enhance as active on its own route", () => {
    renderAt("/enhance");

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

  it("hides the Users nav entry when the user lacks users:manage", async () => {
    vi.mocked(authService.getMe).mockResolvedValue({
      userId: "u1", username: "alice", role: "user", permissions: ["jobs:create"], mustChangePassword: false,
      authMode: "multi", quota: { maxConcurrent: 1, maxQueued: 5, maxJobsPerDay: 50, maxGpuSecondsPerDay: 3600, usedJobsToday: 0, usedGpuSecondsToday: 0 },
    });
    renderAt("/");

    await waitFor(() => expect(screen.queryByRole("link", { name: /users/i })).not.toBeInTheDocument());
  });

  it("shows the Users nav entry when the user has users:manage", async () => {
    vi.mocked(authService.getMe).mockResolvedValue({
      userId: "u1", username: "admin", role: "admin", permissions: ["users:manage"], mustChangePassword: false,
      authMode: "multi", quota: { maxConcurrent: 0, maxQueued: 0, maxJobsPerDay: 0, maxGpuSecondsPerDay: 0, usedJobsToday: 0, usedGpuSecondsToday: 0 },
    });
    renderAt("/");

    await waitFor(() => expect(screen.getByRole("link", { name: /users/i })).toBeInTheDocument());
  });
});

describe("la pagina no se sale de la pantalla", () => {
  // Reportado el 2026-08-06: "si scrolleo mucho la pagina se corre de los
  // limites". El armado es una grilla de tres columnas dentro de un contenedor
  // de la altura EXACTA de la pantalla. En una grilla, un hijo no se encoge por
  // debajo de su contenido: si la navegacion o la cola de trabajos crecen,
  // estiran la fila mas alla de la pantalla y todo se sale.
  //
  // Se arregla con `min-h-0` (permite encoger) + `overflow-y-auto` (scrollea
  // adentro en vez de empujar). Las tres columnas lo necesitan, no solo el
  // centro.
  it.each([
    ["nav.mainLabel", "navegacion"],
    ["nav.queueLabel", "cola de trabajos"],
  ])("la columna de %s scrollea adentro en vez de estirar la fila", async (labelKey) => {
    renderAt("/");
    const { en } = await import("../i18n/en");

    const columna = await screen.findByLabelText(en[labelKey as keyof typeof en] as string);

    expect(columna.className).toContain("min-h-0");
    expect(columna.className).toContain("overflow-y-auto");
  });

  it("la columna central tambien puede encogerse", async () => {
    renderAt("/");

    const centro = document.querySelector("main");

    expect(centro?.className).toContain("min-h-0");
    expect(centro?.className).toContain("overflow-y-auto");
  });
});
