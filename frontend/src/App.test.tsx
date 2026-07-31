import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { AuthProvider } from "./hooks/useAuth";
import * as authService from "./services/auth";
import type { MeResponse } from "./lib/apiTypes";
import { ApiError } from "./lib/api";
import { en } from "./i18n/en";
import * as transcribeService from "./services/transcribe";

vi.mock("./services/auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./services/auth")>();
  return { ...actual, getMe: vi.fn() };
});

vi.mock("./services/transcribe", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("./services/transcribe")>();
  return {
    ...actual,
    fetchInstalledAsrModels: vi.fn(),
    fetchTranscribeDevices: vi.fn(),
    searchAsrModels: vi.fn(),
  };
});

function renderApp(initialEntry = "/") {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
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
  vi.mocked(transcribeService.fetchInstalledAsrModels).mockReset();
  vi.mocked(transcribeService.fetchTranscribeDevices).mockReset();
  vi.mocked(transcribeService.searchAsrModels).mockReset();
});

describe("App auth gate", () => {
  it("renders the normal app UI unchanged in off mode", async () => {
    vi.mocked(authService.getMe).mockResolvedValue(OFF_MODE_ME);
    renderApp();

    // La raiz paso a ser el selector de tareas (Fase 3.3 del gestor unificado).
    await waitFor(() =>
      expect(screen.getByRole("heading", { name: /what do you want to do/i })).toBeInTheDocument(),
    );
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

  it("renders the transcription page at /transcribe", async () => {
    vi.mocked(authService.getMe).mockResolvedValue(OFF_MODE_ME);
    vi.mocked(transcribeService.fetchInstalledAsrModels).mockResolvedValue([]);
    vi.mocked(transcribeService.fetchTranscribeDevices).mockResolvedValue({
      devices: [],
      defaultDeviceId: "cpu",
    });
    vi.mocked(transcribeService.searchAsrModels).mockResolvedValue({
      results: [],
    });

    renderApp("/transcribe");

    expect(
      await screen.findByRole("heading", {
        name: en["transcribe.page.title"],
      }),
    ).toBeInTheDocument();
  });
});

vi.mock("./services/download", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./services/download")>();
  return { ...actual, probeMedia: vi.fn(), createDownloadJob: vi.fn(), getDownloadJob: vi.fn() };
});

describe("App — las pestañas de trabajo no pierden su estado", () => {
  it("el formulario de Download sobrevive a pasar por otra pestaña", async () => {
    // La queja real: "estoy escalando algo, paso a descargar un audio y se pierde
    // todo el formulario". Las páginas de trabajo quedan montadas y ocultas.
    vi.mocked(authService.getMe).mockResolvedValue(OFF_MODE_ME);
    renderApp("/download");

    const input = await screen.findByLabelText(/Dirección del video/i);
    const { fireEvent } = await import("@testing-library/react");
    fireEvent.change(input, { target: { value: "https://youtube.com/watch?v=persistente" } });

    fireEvent.click(screen.getByRole("link", { name: /Settings/i }));
    await waitFor(() =>
      expect(screen.queryByLabelText(/Dirección del video/i)).not.toBeVisible(),
    );

    fireEvent.click(screen.getByRole("link", { name: /Download/i }));

    const restored = await screen.findByLabelText(/Dirección del video/i);
    expect(restored).toHaveValue("https://youtube.com/watch?v=persistente");
  });

  it("una pestaña de trabajo nunca visitada no se monta", async () => {
    // Montar todo al arrancar dispararía las consultas de todas las páginas.
    vi.mocked(authService.getMe).mockResolvedValue(OFF_MODE_ME);
    renderApp("/");

    await waitFor(() =>
      expect(screen.getByRole("heading", { name: /what do you want to do/i })).toBeInTheDocument(),
    );
    expect(screen.queryByLabelText(/Dirección del video/i)).not.toBeInTheDocument();
  });
});
