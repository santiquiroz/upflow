import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ModelSearchResponse } from "../../lib/apiTypes";
import * as generationService from "../../services/generation";
import { GenerationHfSearch } from "./GenerationHfSearch";

vi.mock("../../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/generation")>();
  return {
    ...actual,
    searchGenerationModels: vi.fn(),
    preflightGenerationModel: vi.fn(),
    installGenerationModel: vi.fn(),
    getGenerationInstallStatus: vi.fn(),
    getConversionStatus: vi.fn(),
    fetchActiveConversions: vi.fn(),
    cancelConversion: vi.fn(),
  };
});

const SEARCH_RESULT: ModelSearchResponse = {
  results: [
    {
      id: "amd/sdxl-onnx",
      author: "amd",
      pipelineTag: "text-to-image",
      downloads: 10,
      likes: 2,
      tags: [],
      compat: "ready_onnx",
      compatReasonKey: "compat.readyOnnx",
      compatReasonParams: {},
      availablePrecisions: [],
    },
  ],
};

function renderSearch() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<GenerationHfSearch debounceMs={0} />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(generationService.searchGenerationModels).mockReset();
  vi.mocked(generationService.preflightGenerationModel).mockReset();
  vi.mocked(generationService.installGenerationModel).mockReset();
  vi.mocked(generationService.getGenerationInstallStatus).mockReset();
  vi.mocked(generationService.getConversionStatus).mockReset();
});

describe("GenerationHfSearch", () => {
  it("renders browse results for an empty query instead of the search banner", async () => {
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(SEARCH_RESULT);

    renderSearch();

    await waitFor(() => expect(generationService.searchGenerationModels).toHaveBeenCalledWith(""));
    expect(await screen.findByText("amd/sdxl-onnx")).toBeInTheDocument();
    expect(
      screen.queryByText(
        "Search Hugging Face for a Stable Diffusion (text-to-image) pipeline to install.",
      ),
    ).not.toBeInTheDocument();
  });

  it("searches with the generation endpoint and displays results", async () => {
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(SEARCH_RESULT);

    renderSearch();
    fireEvent.change(screen.getByRole("searchbox", { name: /search hugging face/i }), {
      target: { value: "sdxl" },
    });

    await waitFor(() => expect(generationService.searchGenerationModels).toHaveBeenCalledWith("sdxl"));
    expect(await screen.findByText("amd/sdxl-onnx")).toBeInTheDocument();
  });

  it("installs with the generation endpoint when Install is clicked", async () => {
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(SEARCH_RESULT);
    vi.mocked(generationService.installGenerationModel).mockResolvedValue({
      installId: "abc",
      statusUrl: "/x",
    });
    vi.mocked(generationService.getGenerationInstallStatus).mockImplementation(
      () => new Promise<never>(() => {}),
    );

    renderSearch();
    fireEvent.change(screen.getByRole("searchbox", { name: /search hugging face/i }), {
      target: { value: "sdxl" },
    });
    fireEvent.click(await screen.findByRole("button", { name: /install/i }));

    await waitFor(() =>
      expect(vi.mocked(generationService.installGenerationModel).mock.calls[0]?.[0]).toBe("amd/sdxl-onnx"),
    );
  });

  it("shows the active conversion stage for a search result install", async () => {
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(SEARCH_RESULT);
    vi.mocked(generationService.installGenerationModel).mockResolvedValue({
      installId: "abc",
      statusUrl: "/x",
    });
    vi.mocked(generationService.getGenerationInstallStatus).mockResolvedValue({
      installId: "abc",
      repoId: "amd/sdxl-onnx",
      status: "converting",
      progressPct: null,
      modelId: null,
      error: null,
      conversionId: "convert-1",
    });
    vi.mocked(generationService.getConversionStatus).mockResolvedValue({
      conversionId: "convert-1",
      repoId: "amd/sdxl-onnx",
      status: "running",
      progressPct: 65,
      stage: "optimizing",
      stages: [
        {
          key: "optimizing",
          label: "Optimizing graph",
          weight: 20,
          status: "active",
        },
      ],
      modelId: null,
      error: null,
    });

    renderSearch();
    fireEvent.change(screen.getByRole("searchbox", { name: /search hugging face/i }), {
      target: { value: "sdxl" },
    });
    fireEvent.click(await screen.findByRole("button", { name: /install/i }));

    expect(await screen.findByText("Converting — Optimizing graph")).toBeInTheDocument();
  });
});

const MIXED_RESULTS: ModelSearchResponse = {
  results: [
    {
      id: "amd/listo-onnx",
      author: "amd",
      pipelineTag: "text-to-image",
      downloads: 10,
      likes: 2,
      tags: [],
      compat: "ready_onnx",
      compatReasonKey: "compat.readyOnnx",
      compatReasonParams: {},
      availablePrecisions: [],
    },
    {
      id: "john/necesita-conversion",
      author: "john",
      pipelineTag: "text-to-image",
      downloads: 5,
      likes: 1,
      tags: [],
      compat: "needs_conversion",
      compatReasonKey: "compat.needsConversion",
      compatReasonParams: {},
      availablePrecisions: ["fp16"],
    },
  ],
};

describe("GenerationHfSearch — filtro por compatibilidad", () => {
  it("'Listos' esconde los que requieren conversión", async () => {
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(MIXED_RESULTS);

    renderSearch();
    await screen.findByText("amd/listo-onnx");

    fireEvent.click(screen.getByRole("radio", { name: /Ready to use/i }));

    expect(screen.getByText("amd/listo-onnx")).toBeInTheDocument();
    expect(screen.queryByText("john/necesita-conversion")).not.toBeInTheDocument();
  });

  it("'Con conversión' muestra solo los que la requieren", async () => {
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(MIXED_RESULTS);

    renderSearch();
    await screen.findByText("amd/listo-onnx");

    fireEvent.click(screen.getByRole("radio", { name: /Needs conversion/i }));

    expect(screen.getByText("john/necesita-conversion")).toBeInTheDocument();
    expect(screen.queryByText("amd/listo-onnx")).not.toBeInTheDocument();
  });

  it("'Todos' es el default y no esconde nada", async () => {
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(MIXED_RESULTS);

    renderSearch();

    expect(await screen.findByText("amd/listo-onnx")).toBeInTheDocument();
    expect(screen.getByText("john/necesita-conversion")).toBeInTheDocument();
    expect((screen.getByRole("radio", { name: /All/i }) as HTMLInputElement).checked).toBe(true);
  });
});


describe("volver a la seccion no pierde la conversion", () => {
  // Reportado el 2026-08-06: "al intentar instalar y convertir un modelo y salir
  // a otra ventana se pierde el progreso". La conversion nunca se perdia —sigue
  // en el servidor— pero el id vivia solo en la pantalla y se iba con ella.
  // Convertir un SDXL tarda cerca de media hora: creer que se perdio y
  // arrancarlo de nuevo es media hora tirada.
  it("re-engancha la barra de una conversion que ya venia corriendo", async () => {
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(SEARCH_RESULT);
    vi.mocked(generationService.fetchActiveConversions).mockResolvedValue([
      {
        conversionId: "c1",
        repoId: "amd/sdxl-onnx",
        status: "running",
        progressPct: 42,
        stage: "exporting:unet",
        stages: null,
        modelId: null,
        error: null,
      },
    ]);
    vi.mocked(generationService.getConversionStatus).mockResolvedValue({
      conversionId: "c1",
      repoId: "amd/sdxl-onnx",
      status: "running",
      progressPct: 42,
      stage: "exporting:unet",
      stages: null,
      modelId: null,
      error: null,
    });

    renderSearch();

    // Sin instalar nada en esta pantalla, el progreso de la que ya corria
    // tiene que aparecer igual.
    await waitFor(() => {
      expect(generationService.getConversionStatus).toHaveBeenCalledWith("c1");
    });
  });

  it("sin conversiones en curso no consulta ninguna", async () => {
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(SEARCH_RESULT);
    vi.mocked(generationService.fetchActiveConversions).mockResolvedValue([]);

    renderSearch();

    await waitFor(() => {
      expect(generationService.fetchActiveConversions).toHaveBeenCalled();
    });
    expect(generationService.getConversionStatus).not.toHaveBeenCalled();
  });
});


describe("cortar una conversion", () => {
  // Convertir un SDXL tarda cerca de media hora y ocupa la maquina entera. Sin
  // esto, equivocarse de modelo solo se arregla cerrando la app.
  it("ofrece cortarla y manda el corte al servidor", async () => {
    const CORRIENDO = {
      conversionId: "c1",
      repoId: "amd/sdxl-onnx",
      status: "running" as const,
      progressPct: 42,
      stage: "exporting:unet",
      stages: null,
      modelId: null,
      error: null,
    };
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(SEARCH_RESULT);
    vi.mocked(generationService.fetchActiveConversions).mockResolvedValue([CORRIENDO]);
    vi.mocked(generationService.getConversionStatus).mockResolvedValue(CORRIENDO);
    vi.mocked(generationService.cancelConversion).mockResolvedValue({
      ...CORRIENDO,
      status: "cancelled" as const,
    });

    renderSearch();

    const boton = await screen.findByRole("button", { name: /cancel|cancelar/i });
    fireEvent.click(boton);

    await waitFor(() => {
      // React Query suma un segundo argumento de contexto; lo que importa es el id.
      expect(vi.mocked(generationService.cancelConversion).mock.calls[0][0]).toBe("c1");
    });
  });

  it("sin conversion en curso no ofrece cortar nada", async () => {
    vi.mocked(generationService.searchGenerationModels).mockResolvedValue(SEARCH_RESULT);
    vi.mocked(generationService.fetchActiveConversions).mockResolvedValue([]);

    renderSearch();

    await screen.findByText("amd/sdxl-onnx");
    expect(screen.queryByRole("button", { name: /cancel|cancelar/i })).toBeNull();
  });
});
