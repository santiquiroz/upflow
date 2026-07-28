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
      compatReason: "Pipeline ONNX listo",
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
