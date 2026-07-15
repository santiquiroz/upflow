import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../lib/api";
import type { ModelSearchResponse } from "../../lib/apiTypes";
import { HfSearch } from "./HfSearch";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return { ...actual, searchHfModels: vi.fn(), installModel: vi.fn(), getInstallStatus: vi.fn() };
});

const DEBOUNCE_MS = 20;

function renderSearch() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<HfSearch debounceMs={DEBOUNCE_MS} />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(api.searchHfModels).mockReset();
});

describe("HfSearch", () => {
  it("shows an empty state before any search is entered", () => {
    renderSearch();

    expect(screen.getByText(/search hugging face for an onnx/i)).toBeInTheDocument();
    expect(api.searchHfModels).not.toHaveBeenCalled();
  });

  it("debounces typing and only queries once after the delay settles", async () => {
    vi.mocked(api.searchHfModels).mockResolvedValue({ results: [] });

    renderSearch();
    const input = screen.getByRole("searchbox", { name: /search hugging face/i });

    fireEvent.change(input, { target: { value: "a" } });
    fireEvent.change(input, { target: { value: "an" } });
    fireEvent.change(input, { target: { value: "anime" } });

    await waitFor(() => expect(api.searchHfModels).toHaveBeenCalledTimes(1));
    expect(api.searchHfModels).toHaveBeenCalledWith("anime");
  });

  it("shows a no-results state when the search returns nothing", async () => {
    vi.mocked(api.searchHfModels).mockResolvedValue({ results: [] });

    renderSearch();
    fireEvent.change(screen.getByRole("searchbox", { name: /search hugging face/i }), {
      target: { value: "doesnotexist" },
    });

    expect(await screen.findByText(/no models found for "doesnotexist"/i)).toBeInTheDocument();
  });

  it("renders a result card per hit once results arrive", async () => {
    const payload: ModelSearchResponse = {
      results: [
        { id: "example/anime-2x", author: "example", pipelineTag: "image-to-image", downloads: 120, likes: 5, tags: ["onnx"] },
        { id: "other/model-4x", author: null, pipelineTag: null, downloads: 3, likes: 0, tags: [] },
      ],
    };
    vi.mocked(api.searchHfModels).mockResolvedValue(payload);

    renderSearch();
    fireEvent.change(screen.getByRole("searchbox", { name: /search hugging face/i }), {
      target: { value: "anime" },
    });

    expect(await screen.findByText("example/anime-2x")).toBeInTheDocument();
    expect(screen.getByText("other/model-4x")).toBeInTheDocument();
  });

  it("shows a clear error state when the search request fails", async () => {
    vi.mocked(api.searchHfModels).mockRejectedValue(new Error("Hugging Face search failed"));

    renderSearch();
    fireEvent.change(screen.getByRole("searchbox", { name: /search hugging face/i }), {
      target: { value: "anime" },
    });

    expect(await screen.findByRole("alert")).toHaveTextContent(/search failed/i);
  });
});
