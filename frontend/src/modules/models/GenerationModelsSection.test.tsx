import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type { InstallStatusResponse, ModelResponse, ModelsResponse } from "../../lib/apiTypes";
import * as generationService from "../../services/generation";
import { GenerationModelsSection } from "./GenerationModelsSection";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getModels: vi.fn(), deleteModel: vi.fn() };
});

vi.mock("../../services/generation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../services/generation")>();
  return { ...actual, installGenerationModel: vi.fn(), getGenerationInstallStatus: vi.fn() };
});

const POLL_INTERVAL_MS = 10;
const REPO_ID_PLACEHOLDER = "amd/stable-diffusion-1.5_io16_amdgpu";

const DIFFUSION_MODEL: ModelResponse = {
  id: "amd--stable-diffusion-1-5",
  name: "Stable Diffusion 1.5 (AMD)",
  kind: "diffusion-onnx",
  source: "https://huggingface.co/amd/stable-diffusion-1.5_io16_amdgpu",
  scale: null,
  arch: null,
  sizeBytes: 2_147_483_648,
  status: "installed",
  error: null,
};

const UPSCALE_MODEL: ModelResponse = {
  id: "custom-anime-2x",
  name: "Custom Anime 2x",
  kind: "onnx",
  source: "https://huggingface.co/example/model",
  scale: 2,
  arch: "compact",
  sizeBytes: 5_242_880,
  status: "installed",
  error: null,
};

function renderSection(models: ModelResponse[] = [DIFFUSION_MODEL]) {
  vi.mocked(api.getModels).mockResolvedValue({ models } satisfies ModelsResponse);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<GenerationModelsSection pollIntervalMs={POLL_INTERVAL_MS} />, { wrapper: Wrapper });
}

function typeRepoId(repoId: string) {
  fireEvent.change(screen.getByPlaceholderText(REPO_ID_PLACEHOLDER), { target: { value: repoId } });
}

afterEach(() => {
  vi.mocked(api.getModels).mockReset();
  vi.mocked(api.deleteModel).mockReset();
  vi.mocked(generationService.installGenerationModel).mockReset();
  vi.mocked(generationService.getGenerationInstallStatus).mockReset();
});

describe("GenerationModelsSection", () => {
  it("lists only diffusion-onnx models, excluding upscaler onnx models", async () => {
    renderSection([DIFFUSION_MODEL, UPSCALE_MODEL]);

    expect(await screen.findByText("Stable Diffusion 1.5 (AMD)")).toBeInTheDocument();
    expect(screen.queryByText("Custom Anime 2x")).not.toBeInTheDocument();
  });

  it("installs a model by repo id, shows polling progress, and refreshes the list once installed", async () => {
    vi.mocked(generationService.installGenerationModel).mockResolvedValue({ installId: "install-1", statusUrl: "/x" });
    const base: InstallStatusResponse = {
      installId: "install-1",
      repoId: "amd/sd15",
      status: "downloading",
      progressPct: 30,
      modelId: null,
      error: null,
    };
    vi.mocked(generationService.getGenerationInstallStatus)
      .mockResolvedValueOnce(base)
      .mockResolvedValue({ ...base, status: "installed", progressPct: null, modelId: "amd--sd15" });

    renderSection([]);
    await screen.findByText(/no generation models installed yet/i);

    typeRepoId("amd/sd15");
    fireEvent.click(screen.getByRole("button", { name: /install/i }));

    await waitFor(() =>
      expect(vi.mocked(generationService.installGenerationModel).mock.calls[0]?.[0]).toBe("amd/sd15"),
    );
    await waitFor(() => expect(screen.getByText("30%")).toBeInTheDocument());

    vi.mocked(api.getModels).mockResolvedValue({ models: [DIFFUSION_MODEL] } satisfies ModelsResponse);
    await waitFor(() => expect(screen.getByText("Stable Diffusion 1.5 (AMD)")).toBeInTheDocument());
  });

  it("shows the backend error message when the install fails", async () => {
    vi.mocked(generationService.installGenerationModel).mockResolvedValue({ installId: "install-2", statusUrl: "/x" });
    vi.mocked(generationService.getGenerationInstallStatus).mockResolvedValue({
      installId: "install-2",
      repoId: "amd/sd15",
      status: "error",
      progressPct: null,
      modelId: null,
      error: "CUDA is required for this model but was not detected",
    });

    renderSection([]);
    typeRepoId("amd/sd15");
    fireEvent.click(screen.getByRole("button", { name: /install/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/cuda is required/i);
  });

  it("requires a destructive confirmation before deleting a diffusion model", async () => {
    renderSection([DIFFUSION_MODEL]);

    fireEvent.click(await screen.findByRole("button", { name: /delete stable diffusion 1\.5/i }));

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(/delete stable diffusion 1\.5 \(amd\)\?/i)).toBeInTheDocument();
    expect(api.deleteModel).not.toHaveBeenCalled();
  });

  it("cancels without deleting when the cancel action is chosen", async () => {
    renderSection([DIFFUSION_MODEL]);

    fireEvent.click(await screen.findByRole("button", { name: /delete stable diffusion 1\.5/i }));
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(api.deleteModel).not.toHaveBeenCalled();
  });

  it("deletes the diffusion model once the destructive action is confirmed", async () => {
    vi.mocked(api.deleteModel).mockResolvedValue(undefined);
    renderSection([DIFFUSION_MODEL]);

    fireEvent.click(await screen.findByRole("button", { name: /delete stable diffusion 1\.5/i }));
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(vi.mocked(api.deleteModel).mock.calls[0]?.[0]).toBe(DIFFUSION_MODEL.id));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});
