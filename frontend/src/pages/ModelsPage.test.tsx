import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../lib/api";
import type { DevicesResponse, ModelsResponse } from "../lib/apiTypes";
import { ModelsPage } from "./ModelsPage";

vi.mock("../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../lib/api")>();
  return { ...actual, getModels: vi.fn(), getDevices: vi.fn(), searchHfModels: vi.fn() };
});

function renderPage() {
  vi.mocked(api.getModels).mockResolvedValue({ models: [] } satisfies ModelsResponse);
  vi.mocked(api.getDevices).mockResolvedValue({
    devices: [{ id: "cpu", kind: "cpu", name: "CPU", backend: "cpu" }],
    defaultDeviceId: "cpu",
  } satisfies DevicesResponse);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<ModelsPage />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(api.getModels).mockReset();
  vi.mocked(api.getDevices).mockReset();
  vi.mocked(api.searchHfModels).mockReset();
});

describe("ModelsPage", () => {
  it("renders the search, installed models and default device sections", async () => {
    renderPage();

    expect(screen.getByRole("heading", { name: "Models", level: 1 })).toBeInTheDocument();
    expect(screen.getByRole("searchbox", { name: /search hugging face/i })).toBeInTheDocument();
    expect(await screen.findByText(/no custom onnx models installed yet/i)).toBeInTheDocument();
    expect(await screen.findByText("CPU")).toBeInTheDocument();
  });
});
