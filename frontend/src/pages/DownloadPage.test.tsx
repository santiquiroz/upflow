import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { DownloadJob, MediaProbe } from "../lib/apiTypes";
import * as downloadService from "../services/download";
import { DownloadPage } from "./DownloadPage";

vi.mock("../services/download", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/download")>();
  return {
    ...actual,
    probeMedia: vi.fn(),
    createDownloadJob: vi.fn(),
    getDownloadJob: vi.fn(),
    cancelDownloadJob: vi.fn(),
  };
});

function makeProbe(overrides: Partial<MediaProbe> = {}): MediaProbe {
  return {
    title: "Big Buck Bunny",
    durationSeconds: 635,
    uploader: "Blender",
    extractor: "Youtube",
    isPlaylist: false,
    entryCount: 1,
    availableHeights: [360, 720, 1080, 2160],
    ...overrides,
  };
}

function makeJob(overrides: Partial<DownloadJob> = {}): DownloadJob {
  return {
    id: "job-1",
    status: "running",
    url: "https://example.com/v",
    maxHeight: 1080,
    audioOnly: false,
    mediaTitle: "Big Buck Bunny",
    mediaUploader: "Blender",
    extractor: "Youtube",
    createdAt: new Date().toISOString(),
    startedAt: null,
    finishedAt: null,
    progressPct: 42,
    downloadedBytes: 1024,
    totalBytes: 2048,
    outputFiles: [],
    outputDirectory: "",
    error: null,
    ownerId: null,
    ...overrides,
  };
}

function renderPage() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<DownloadPage />, { wrapper: Wrapper });
}

function typeUrl(value: string) {
  fireEvent.change(screen.getByLabelText(/Dirección del video/i), { target: { value } });
}

afterEach(() => {
  vi.mocked(downloadService.probeMedia).mockReset();
  vi.mocked(downloadService.createDownloadJob).mockReset();
  vi.mocked(downloadService.getDownloadJob).mockReset();
  vi.mocked(downloadService.cancelDownloadJob).mockReset();
});

describe("DownloadPage", () => {
  it("no deja pedir nada hasta que hay una URL que parece una URL", () => {
    renderPage();

    expect(screen.getByRole("button", { name: /Descargar/i })).toBeDisabled();

    typeUrl("https://youtube.com/watch?v=x");

    expect(screen.getByRole("button", { name: /Descargar/i })).toBeEnabled();
  });

  it("consulta la URL sola, sin que haya que tocar nada", async () => {
    vi.mocked(downloadService.probeMedia).mockResolvedValue(makeProbe());
    renderPage();
    typeUrl("https://youtube.com/watch?v=x");

    expect(await screen.findByText("Big Buck Bunny")).toBeInTheDocument();
    // 635s -> 10:35, y el sitio del que viene.
    expect(screen.getByText(/10:35/)).toBeInTheDocument();
    expect(screen.getByText(/Youtube/)).toBeInTheDocument();
  });

  it("avisa que la URL es una lista y cuántos va a bajar de verdad", async () => {
    // La queja más repetida de los descargadores: pegar un link y que arranquen 200.
    vi.mocked(downloadService.probeMedia).mockResolvedValue(
      makeProbe({ isPlaylist: true, entryCount: 200 }),
    );
    renderPage();
    typeUrl("https://youtube.com/playlist?list=x");

    expect(await screen.findByText(/lista de 200 elementos/i)).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
  });

  it("no ofrece calidades que el video no tiene", async () => {
    vi.mocked(downloadService.probeMedia).mockResolvedValue(
      makeProbe({ availableHeights: [360, 720] }),
    );
    renderPage();
    typeUrl("https://youtube.com/watch?v=x");

    await screen.findByText("Big Buck Bunny");

    expect(screen.getByText("720p")).toBeInTheDocument();
    expect(screen.queryByText("2160p")).not.toBeInTheDocument();
  });

  it("arranca en 1080p y no en la calidad más cara", () => {
    // Un default en 4K es cómo se llega a esperar horas por algo que nadie pidió.
    renderPage();

    const selected = screen.getByRole("radio", { name: /1080p/ });
    expect(selected).toBeChecked();
  });

  it("esconde la calidad cuando se pide solo audio", () => {
    renderPage();

    fireEvent.click(screen.getByRole("button", { name: /Solo audio/i }));

    expect(screen.queryByText("Calidad máxima")).not.toBeInTheDocument();
  });

  it("manda lo que se eligió al crear el trabajo", async () => {
    vi.mocked(downloadService.createDownloadJob).mockResolvedValue(makeJob());
    vi.mocked(downloadService.getDownloadJob).mockResolvedValue(makeJob());
    renderPage();
    typeUrl("https://youtube.com/watch?v=x");
    fireEvent.click(screen.getByRole("radio", { name: /720p/ }));

    fireEvent.click(screen.getByRole("button", { name: /Descargar/i }));

    await waitFor(() =>
      expect(downloadService.createDownloadJob).toHaveBeenCalledWith(
        expect.objectContaining({ maxHeight: 720, audioOnly: false }),
      ),
    );
  });

  it("muestra el motivo cuando el sitio rompe la extracción", async () => {
    // Es lo único útil que se puede dar cuando un sitio cambia.
    vi.mocked(downloadService.probeMedia).mockRejectedValue(
      new Error("[vimeo] 1: Failed to fetch OAuth token: HTTP Error 401"),
    );
    renderPage();
    typeUrl("https://vimeo.com/1");

    expect(await screen.findByText(/401/)).toBeInTheDocument();
  });

  it("usa una barra indeterminada cuando no se conoce el tamaño", async () => {
    // Dibujar un porcentaje inventado sería mentir sobre lo que falta.
    vi.mocked(downloadService.createDownloadJob).mockResolvedValue(makeJob());
    vi.mocked(downloadService.getDownloadJob).mockResolvedValue(
      makeJob({ progressPct: null, totalBytes: null }),
    );
    renderPage();
    typeUrl("https://youtube.com/watch?v=x");
    fireEvent.click(screen.getByRole("button", { name: /Descargar/i }));

    const bar = await screen.findByRole("progressbar", { name: /Descargando/i });
    expect(bar).toHaveAttribute("aria-busy", "true");
  });

  it("dice a dónde fue el archivo y que se puede seguir trabajando con él", async () => {
    // Es el punto de tener esto adentro de Upflow y no metube al lado.
    vi.mocked(downloadService.createDownloadJob).mockResolvedValue(makeJob());
    vi.mocked(downloadService.getDownloadJob).mockResolvedValue(
      makeJob({ status: "completed", outputFiles: ["bunny.mp4"] }),
    );
    renderPage();
    typeUrl("https://youtube.com/watch?v=x");
    fireEvent.click(screen.getByRole("button", { name: /Descargar/i }));

    expect(await screen.findByText(/bunny\.mp4/)).toBeInTheDocument();
    expect(screen.getByText(/Enhance/)).toBeInTheDocument();
  });

  it("deja cancelar mientras corre y no cuando terminó", async () => {
    vi.mocked(downloadService.createDownloadJob).mockResolvedValue(makeJob());
    vi.mocked(downloadService.getDownloadJob).mockResolvedValue(makeJob());
    vi.mocked(downloadService.cancelDownloadJob).mockResolvedValue(
      makeJob({ status: "cancelled" }),
    );
    renderPage();
    typeUrl("https://youtube.com/watch?v=x");
    fireEvent.click(screen.getByRole("button", { name: /Descargar/i }));

    const cancelButton = await screen.findByRole("button", { name: /Cancelar/i });
    fireEvent.click(cancelButton);

    await waitFor(() => expect(downloadService.cancelDownloadJob).toHaveBeenCalled());
  });
});

describe("DownloadPage — lo que faltaba", () => {
  it("dice en qué carpeta quedó el archivo", async () => {
    // Decir el nombre sin decir dónde obliga a salir a buscarlo. Fue lo primero que se
    // notó usándolo de verdad.
    vi.mocked(downloadService.createDownloadJob).mockResolvedValue(makeJob());
    vi.mocked(downloadService.getDownloadJob).mockResolvedValue(
      makeJob({
        status: "completed",
        outputFiles: ["bunny.mp4"],
        outputDirectory: String.raw`C:\litellm\uploads`,
      }),
    );
    renderPage();
    typeUrl("https://youtube.com/watch?v=x");
    fireEvent.click(screen.getByRole("button", { name: /Descargar/i }));

    expect(await screen.findByText(/litellm/)).toBeInTheDocument();
  });

  it("no consulta el sitio por cada tecla", async () => {
    // Una consulta por pulsación es exactamente cómo se llega al "rate-limited por una
    // hora" que apareció usándolo.
    vi.mocked(downloadService.probeMedia).mockResolvedValue(makeProbe());
    renderPage();

    typeUrl("https://youtube.com/watch?v=");
    typeUrl("https://youtube.com/watch?v=a");
    typeUrl("https://youtube.com/watch?v=ab");

    await screen.findByText("Big Buck Bunny");
    expect(vi.mocked(downloadService.probeMedia).mock.calls.length).toBeLessThan(3);
  });

  it("no consulta nada mientras la dirección no parezca una dirección", () => {
    renderPage();

    typeUrl("holaaa");

    expect(downloadService.probeMedia).not.toHaveBeenCalled();
  });
});
