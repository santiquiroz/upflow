import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import type { ModelResponse, ModelsResponse } from "../lib/apiTypes";
import { ModelPicker } from "./ModelPicker";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, getModels: vi.fn() };
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
  sizeBytes: 1_048_576,
  status: "installed",
  error: null,
};

const CONVERTING_ONNX_MODEL: ModelResponse = {
  ...ONNX_MODEL,
  id: "converting-model",
  name: "Converting Model",
  status: "converting",
};

function renderPicker(models: ModelResponse[], onChange = vi.fn()) {
  vi.mocked(api.getModels).mockResolvedValue({ models } satisfies ModelsResponse);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return { onChange, ...render(<ModelPicker value={null} onChange={onChange} />, { wrapper: Wrapper }) };
}

afterEach(() => {
  vi.mocked(api.getModels).mockReset();
});

describe("ModelPicker", () => {
  it("groups builtin and onnx models under separate labeled sections", async () => {
    renderPicker([BUILTIN_MODEL, ONNX_MODEL]);

    const builtinGroup = await screen.findByRole("group", { name: "Builtin" });
    const onnxGroup = screen.getByRole("group", { name: "ONNX" });

    expect(within(builtinGroup).getByText("RealESRGAN x4plus")).toBeInTheDocument();
    expect(within(builtinGroup).queryByText("Custom Anime 2x")).not.toBeInTheDocument();
    expect(within(onnxGroup).getByText("Custom Anime 2x")).toBeInTheDocument();
    expect(within(onnxGroup).queryByText("RealESRGAN x4plus")).not.toBeInTheDocument();
  });

  it("omits a group entirely when it has no models", async () => {
    renderPicker([BUILTIN_MODEL]);

    await screen.findByRole("group", { name: "Builtin" });
    expect(screen.queryByRole("group", { name: "ONNX" })).not.toBeInTheDocument();
  });

  it("shows scale and architecture as tabular numbers", async () => {
    renderPicker([BUILTIN_MODEL]);

    const meta = await screen.findByText(/4x/);
    expect(meta).toHaveClass("font-mono-tabular");
  });

  it("disables selection for a model that is not installed yet", async () => {
    renderPicker([CONVERTING_ONNX_MODEL]);

    const radio = await screen.findByRole("radio", { name: /Converting Model/ });
    expect(radio).toBeDisabled();
  });

  it("calls onChange with the selected model when an installed model is picked", async () => {
    const { onChange } = renderPicker([BUILTIN_MODEL, ONNX_MODEL]);

    const radio = await screen.findByRole("radio", { name: /Custom Anime 2x/ });
    fireEvent.click(radio);

    expect(onChange).toHaveBeenCalledWith(ONNX_MODEL);
  });

  it("shows an error message when the models request fails", async () => {
    vi.mocked(api.getModels).mockRejectedValue(new Error("network down"));
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    function Wrapper({ children }: { children: ReactNode }) {
      return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
    }
    render(<ModelPicker value={null} onChange={vi.fn()} />, { wrapper: Wrapper });

    expect(await screen.findByText(/Could not load models/i)).toBeInTheDocument();
  });
});
