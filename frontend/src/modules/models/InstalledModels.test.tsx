import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type { ModelResponse, ModelsResponse } from "../../lib/apiTypes";
import { InstalledModels } from "./InstalledModels";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, getModels: vi.fn(), deleteModel: vi.fn() };
});

const BUILTIN_MODEL: ModelResponse = {
  id: "realesrgan-x4plus",
  name: "RealESRGAN x4plus",
  kind: "builtin-ncnn",
  source: "builtin",
  scale: 4,
  arch: "esrgan",
  sizeBytes: 0,
  status: "installed",
  error: null,
};

const ONNX_MODEL: ModelResponse = {
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

function renderList(models: ModelResponse[]) {
  vi.mocked(api.getModels).mockResolvedValue({ models } satisfies ModelsResponse);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<InstalledModels />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(api.getModels).mockReset();
  vi.mocked(api.deleteModel).mockReset();
});

describe("InstalledModels", () => {
  it("marks a builtin model as non-deletable and renders no delete button for it", async () => {
    renderList([BUILTIN_MODEL, ONNX_MODEL]);

    await screen.findByText("RealESRGAN x4plus");
    expect(screen.getByTitle(/built-in models cannot be removed/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /delete realesrgan x4plus/i })).not.toBeInTheDocument();
  });

  it("renders a delete button for a non-builtin onnx model", async () => {
    renderList([BUILTIN_MODEL, ONNX_MODEL]);

    expect(await screen.findByRole("button", { name: /delete custom anime 2x/i })).toBeInTheDocument();
  });

  it("shows tabular size and scale for an installed model", async () => {
    renderList([ONNX_MODEL]);

    const meta = await screen.findByText(/2x.*5\.0 MB/);
    expect(meta).toHaveClass("font-mono-tabular");
  });

  it("shows the ONNX empty state when only builtin models are installed", async () => {
    renderList([BUILTIN_MODEL]);

    expect(await screen.findByText(/no custom onnx models installed yet/i)).toBeInTheDocument();
  });

  it("requires a destructive confirmation before deleting an onnx model", async () => {
    renderList([BUILTIN_MODEL, ONNX_MODEL]);

    fireEvent.click(await screen.findByRole("button", { name: /delete custom anime 2x/i }));

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(/delete custom anime 2x\?/i)).toBeInTheDocument();
    expect(api.deleteModel).not.toHaveBeenCalled();
  });

  it("cancels without deleting when the cancel action is chosen", async () => {
    renderList([BUILTIN_MODEL, ONNX_MODEL]);

    fireEvent.click(await screen.findByRole("button", { name: /delete custom anime 2x/i }));
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(api.deleteModel).not.toHaveBeenCalled();
  });

  it("deletes the model once the destructive action is confirmed", async () => {
    vi.mocked(api.deleteModel).mockResolvedValue(undefined);
    renderList([BUILTIN_MODEL, ONNX_MODEL]);

    fireEvent.click(await screen.findByRole("button", { name: /delete custom anime 2x/i }));
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(vi.mocked(api.deleteModel).mock.calls[0]?.[0]).toBe("custom-anime-2x"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  });
});
