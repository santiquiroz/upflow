import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RealtimeCapabilities } from "../../lib/apiTypes";
import * as realtimeService from "../../services/realtime";
import { RealtimePage } from "./RealtimePage";

// El modulo dejo de ser una pagina de roadmap: ahora configura y abre el overlay
// de verdad. Magpie corre como proceso aparte (es GPL-3.0) y no necesita drivers.

vi.mock("../../services/realtime", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/realtime")>();
  return { ...actual, fetchRealtimeCapabilities: vi.fn(), startRealtime: vi.fn() };
});

const INSTALADO: RealtimeCapabilities = {
  available: true,
  reason: null,
  presets: [
    { id: "anime4k", label: "Anime4K", description: "Para anime y dibujo." },
    { id: "fsr", label: "FSR", description: "Para imagen real y juegos." },
  ],
};

const NO_INSTALADO: RealtimeCapabilities = {
  available: false,
  reason: "El overlay de tiempo real no está instalado.",
  presets: [],
};

function renderPage(capabilities: RealtimeCapabilities = INSTALADO) {
  vi.mocked(realtimeService.fetchRealtimeCapabilities).mockResolvedValue(capabilities);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<RealtimePage />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(realtimeService.fetchRealtimeCapabilities).mockReset();
  vi.mocked(realtimeService.startRealtime).mockReset();
});

describe("RealtimePage", () => {
  it("renders without crashing", () => {
    renderPage();
    expect(screen.getByRole("heading", { level: 1, name: /tiempo real/i })).toBeInTheDocument();
  });

  it("says plainly that no driver is needed", async () => {
    // El overlay es una ventana normal, no engancha el swapchain del juego. Por
    // eso no hay nada que firmar ni que instalar a nivel sistema.
    renderPage();
    expect(await screen.findByText(/no hace falta instalar ningún driver/i)).toBeInTheDocument();
  });

  it("offers the presets returned by the backend", async () => {
    renderPage();
    expect(await screen.findByRole("radio", { name: /Anime4K/ })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: /FSR/ })).toBeInTheDocument();
  });

  it("starts the overlay with the chosen preset and frame cap", async () => {
    const startSpy = vi.mocked(realtimeService.startRealtime);
    startSpy.mockResolvedValue({ pid: 777, preset: "fsr" });
    renderPage();
    await screen.findByRole("radio", { name: /FSR/ });

    fireEvent.click(screen.getByRole("radio", { name: /FSR/ }));
    fireEvent.change(screen.getByLabelText(/límite de cuadros/i), { target: { value: "120" } });
    fireEvent.click(screen.getByRole("button", { name: /abrir overlay/i }));

    await waitFor(() => expect(startSpy).toHaveBeenCalledWith("fsr", 120));
    expect(await screen.findByRole("status")).toHaveTextContent(/overlay abierto/i);
  });

  it("sends no cap when the user leaves it unlimited", async () => {
    const startSpy = vi.mocked(realtimeService.startRealtime);
    startSpy.mockResolvedValue({ pid: 1, preset: "anime4k" });
    renderPage();
    await screen.findByRole("radio", { name: /Anime4K/ });

    fireEvent.click(screen.getByRole("button", { name: /abrir overlay/i }));

    await waitFor(() => expect(startSpy).toHaveBeenCalledWith("anime4k", null));
  });

  it("never invents a keyboard shortcut", async () => {
    // Magpie guarda el atajo como un codigo empaquetado y su propia ventana lo
    // muestra. Nombrar uno concreto aca seria inventarlo.
    vi.mocked(realtimeService.startRealtime).mockResolvedValue({ pid: 1, preset: "anime4k" });
    renderPage();
    await screen.findByRole("radio", { name: /Anime4K/ });
    fireEvent.click(screen.getByRole("button", { name: /abrir overlay/i }));

    const aviso = await screen.findByRole("status");
    expect(aviso).toHaveTextContent(/atajo que muestra la ventana de magpie/i);
    expect(aviso.textContent).not.toMatch(/win\s*\+|ctrl\s*\+|alt\s*\+/i);
  });

  it("shows the real error when the overlay fails to open", async () => {
    vi.mocked(realtimeService.startRealtime).mockRejectedValue(new Error("Magpie no arrancó"));
    renderPage();
    await screen.findByRole("radio", { name: /Anime4K/ });

    fireEvent.click(screen.getByRole("button", { name: /abrir overlay/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Magpie no arrancó");
  });

  it("explains how to install it when it is missing, instead of offering a dead button", async () => {
    renderPage(NO_INSTALADO);

    expect(await screen.findByText(/no está instalado/i)).toBeInTheDocument();
    expect(screen.getByText(/download-magpie/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /abrir overlay/i })).not.toBeInTheDocument();
  });

  it("is still honest about what frame generation cannot do", async () => {
    renderPage();
    expect(await screen.findByText(/Lossless Scaling/i)).toBeInTheDocument();
    expect(screen.getByText(/AFMF/i)).toBeInTheDocument();
  });
});
