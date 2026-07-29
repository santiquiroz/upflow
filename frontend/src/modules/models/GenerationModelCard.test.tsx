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
  checkpoints: [],
  freeRamBytes: null,
};

const SINGLE_FILE_RESULT: HfModelSearchResultResponse = {
  ...RESULT,
  compat: "single_file",
  compatReason: "Checkpoints safetensors en la raíz",
  availablePrecisions: ["fp16"],
};

const SINGLE_FILE_PREFLIGHT: PreflightResponse = {
  ...TIGHT_PREFLIGHT,
  compat: "single_file",
  compatReason: "Checkpoints safetensors en la raíz",
  checkpoints: [
    {
      path: "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
      sizeBytes: 6.5 * GB,
      architecture: "xl_base",
      installable: true,
      reason: "Checkpoint SDXL completo.",
    },
    {
      path: "sdxl_vae.safetensors",
      sizeBytes: 335 * 1024 ** 2,
      architecture: null,
      installable: false,
      reason: "Es un VAE, no un pipeline completo.",
    },
  ],
  freeRamBytes: 12 * GB,
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

  it("shows single-file checkpoints and explains disabled candidates", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue(
      SINGLE_FILE_PREFLIGHT,
    );

    renderCard(SINGLE_FILE_RESULT);
    expect(screen.getByText(/single-file/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));

    const installable = await screen.findByRole("radio", {
      name: /ponyDiffusionV6XL_v6StartWithThisOne.*6\.5 GB/i,
    });
    const vae = screen.getByRole("radio", { name: /sdxl_vae.*335\.0 MB/i });
    expect(installable).toBeChecked();
    expect(installable).toBeEnabled();
    expect(vae).toBeDisabled();
    expect(screen.getByText(/es un VAE, no un pipeline completo/i)).toBeInTheDocument();
  });

  it("installs the checkpoint selected by the user", async () => {
    const preflight: PreflightResponse = {
      ...SINGLE_FILE_PREFLIGHT,
      checkpoints: [
        SINGLE_FILE_PREFLIGHT.checkpoints[0],
        {
          path: "ponyDiffusionV6XL_v6Alt.safetensors",
          sizeBytes: 6 * GB,
          architecture: "xl_base",
          installable: true,
          reason: "Checkpoint SDXL completo.",
        },
      ],
    };
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue(preflight);
    vi.mocked(generationService.installGenerationModel).mockResolvedValue({
      installId: "1",
      statusUrl: "/x",
    });

    renderCard(SINGLE_FILE_RESULT);
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));
    const alternate = await screen.findByRole("radio", { name: /v6Alt/i });
    fireEvent.click(alternate);
    fireEvent.click(screen.getByRole("button", { name: /^install$/i }));

    await waitFor(() =>
      expect(generationService.installGenerationModel).toHaveBeenCalledWith(
        "owner/name",
        "fp16",
        "ponyDiffusionV6XL_v6Alt.safetensors",
      ),
    );
  });

  it("keeps Install enabled with no selection when no checkpoint is installable", async () => {
    vi.mocked(generationService.preflightGenerationModel).mockResolvedValue({
      ...SINGLE_FILE_PREFLIGHT,
      checkpoints: SINGLE_FILE_PREFLIGHT.checkpoints.map((checkpoint, index) => ({
        ...checkpoint,
        installable: index === 0 ? null : false,
        reason: index === 0 ? "No se pudo leer el header." : checkpoint.reason,
      })),
    });

    renderCard(SINGLE_FILE_RESULT);
    fireEvent.click(screen.getByRole("button", { name: /detalles/i }));

    const checkpointRadios = (await screen.findAllByRole("radio")).filter((radio) =>
      radio.getAttribute("name")?.includes("checkpoint"),
    );
    expect(checkpointRadios).not.toHaveLength(0);
    expect(checkpointRadios.every((radio) => !(radio as HTMLInputElement).checked)).toBe(true);
    expect(screen.getByText(/no se pudo evaluar.*no se pudo leer el header/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^install$/i })).toBeEnabled();
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
