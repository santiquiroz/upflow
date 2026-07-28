import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { HfModelSearchResultResponse, PreflightResponse } from "../../lib/apiTypes";
import * as generationService from "../../services/generation";
import { GenerationModelCard } from "./GenerationModelCard";

vi.mock("../../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/generation")>();
  return {
    ...actual,
    preflightGenerationModel: vi.fn(),
    installGenerationModel: vi.fn(),
    getGenerationInstallStatus: vi.fn(),
    getConversionStatus: vi.fn(),
  };
});

const GB = 1024 ** 3;

const RESULT: HfModelSearchResultResponse = {
  id: "owner/name",
  author: "owner",
  pipelineTag: "text-to-image",
  downloads: 10,
  likes: 2,
  tags: [],
  compat: "needs_conversion",
  compatReason: "Sin ONNX propio para unet",
  availablePrecisions: ["fp16", "fp32"],
};

const TIGHT_PREFLIGHT: PreflightResponse = {
  repoId: "owner/name",
  compat: "needs_conversion",
  compatReason: "Sin ONNX propio para unet",
  degraded: false,
  referenceWidth: 512,
  referenceHeight: 512,
  precisions: [
    { precision: "fp16", downloadBytes: 3 * GB, estimatedPeakBytes: 9 * GB },
    { precision: "fp32", downloadBytes: 6 * GB, estimatedPeakBytes: 18 * GB },
  ],
  devices: [{ id: "dml:0", name: "RX 6600", kind: "gpu", freeVramBytes: 7 * GB }],
  disk: { targetPath: "D:\\temp", freeBytes: 1 * GB },
};

function renderCard(result: HfModelSearchResultResponse = RESULT) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<GenerationModelCard result={result} />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(generationService.preflightGenerationModel).mockReset();
  vi.mocked(generationService.installGenerationModel).mockReset();
});

describe("GenerationModelCard", () => {
  it("shows the compat badge without expanding and without calling preflight", () => {
    renderCard();
    expect(screen.getByText(/requiere conversión/i)).toBeInTheDocument();
    expect(generationService.preflightGenerationModel).not.toHaveBeenCalled();
  });

  it("keeps Install enabled even when every warning fires", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue(TIGHT_PREFLIGHT);

    renderCard();
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));

    await waitFor(() => expect(screen.getByText(/no entra/i)).toBeInTheDocument());
    expect(screen.getByText(/libres en D:\\temp/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^install$/i })).toBeEnabled();
  });

  it("offers only the precisions the repo publishes", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue({
      ...TIGHT_PREFLIGHT,
      precisions: [{ precision: "fp32", downloadBytes: 6 * GB, estimatedPeakBytes: 8 * GB }],
    });

    renderCard({ ...RESULT, availablePrecisions: ["fp32"] });
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));

    await waitFor(() => expect(screen.getByRole("radio", { name: /fp32/i })).toBeInTheDocument());
    expect(screen.queryByRole("radio", { name: /fp16/i })).not.toBeInTheDocument();
  });

  it("fires preflight once even when toggled repeatedly", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue(TIGHT_PREFLIGHT);

    renderCard();
    const toggle = screen.getByRole("button", { name: /detalles/i });
    fireEvent.click(toggle);
    await waitFor(() => expect(screen.getByText(/no entra/i)).toBeInTheDocument());
    fireEvent.click(toggle);
    fireEvent.click(toggle);

    await waitFor(() =>
      expect(generationService.preflightGenerationModel).toHaveBeenCalledTimes(1),
    );
  });

  it("installs with the selected precision", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue(TIGHT_PREFLIGHT);
    vi.mocked(generationService.installGenerationModel).mockResolvedValue({
      installId: "1",
      statusUrl: "/x",
    });

    renderCard();
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));
    await waitFor(() => expect(screen.getByRole("radio", { name: /fp32/i })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("radio", { name: /fp32/i }));
    fireEvent.click(screen.getByRole("button", { name: /^install$/i }));

    await waitFor(() =>
      expect(generationService.installGenerationModel).toHaveBeenCalledWith("owner/name", "fp32"),
    );
  });

  it("still allows install when preflight came back degraded", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue({
      ...TIGHT_PREFLIGHT,
      degraded: true,
      compat: null,
      precisions: [],
      disk: null,
    });

    renderCard();
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));

    await waitFor(() => expect(screen.getByText(/no se pudo evaluar/i)).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /^install$/i })).toBeEnabled();
  });
});
