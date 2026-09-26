import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as api from "../../../lib/api";
import type { EngineInfoResponse } from "../../../lib/apiTypes";
import * as restoreService from "../../../services/restore";
import { PhotoRestorePanel, versionedUrl } from "./PhotoRestorePanel";
import { makeAnalysis } from "./restoreTestFixtures";

vi.mock("../../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../lib/api")>();
  return { ...actual, getEngineInfo: vi.fn() };
});

vi.mock("../../../services/restore", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../services/restore")>();
  return { ...actual, analyzePhoto: vi.fn(), setPhotoGeometry: vi.fn(), uploadDamageMask: vi.fn() };
});

const ENGINE_INFO = { maxUploadMb: 1 } as EngineInfoResponse;

function renderPanel() {
  vi.mocked(api.getEngineInfo).mockResolvedValue(ENGINE_INFO);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  render(<PhotoRestorePanel />, { wrapper: Wrapper });
  return queryClient;
}

function dropPhoto(file = new File(["x"], "grandma.jpg", { type: "image/jpeg" })) {
  const input = document.getElementById("restore-file-input") as HTMLInputElement;
  fireEvent.change(input, { target: { files: [file] } });
  return file;
}

afterEach(() => {
  vi.mocked(api.getEngineInfo).mockReset();
  vi.mocked(restoreService.analyzePhoto).mockReset();
  vi.mocked(restoreService.setPhotoGeometry).mockReset();
});

describe("PhotoRestorePanel", () => {
  it("accepts one photo at a time, TIFF scans included", () => {
    renderPanel();

    const input = document.getElementById("restore-file-input") as HTMLInputElement;
    expect(input.multiple).toBe(false);
    expect(input.accept).toContain(".tif");
    expect(input.accept).toContain(".tiff");
    expect(screen.getByText("Drop a photo here or click to browse")).toBeInTheDocument();
  });

  it("always says the photo stays on this computer", () => {
    renderPanel();

    expect(screen.getByText("Runs on your computer. Your photos are not uploaded anywhere.")).toBeInTheDocument();
  });

  it("analyzes the dropped photo and shows the working copy with the framing tools", async () => {
    vi.mocked(restoreService.analyzePhoto).mockResolvedValue(makeAnalysis());
    renderPanel();

    const photo = dropPhoto();

    expect(screen.getByRole("status")).toHaveTextContent("Analyzing the photo…");
    const preview = await screen.findByRole("img", { name: "Working copy of grandma.jpg" });
    expect(restoreService.analyzePhoto).toHaveBeenCalledWith(photo, expect.anything());
    expect(preview).toHaveAttribute("src", "/api/v1/restore/analysis/tok-1/preview.jpg?v=1");
    expect(screen.getByText("1200 × 800 px · 8-bit")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Rotate" })).toBeInTheDocument();
  });

  it("sends the new geometry and refreshes the preview", async () => {
    vi.mocked(restoreService.analyzePhoto).mockResolvedValue(makeAnalysis());
    vi.mocked(restoreService.setPhotoGeometry).mockResolvedValue(
      makeAnalysis({ width: 800, height: 1200, geometry: { rotate90: 1, crop: null, angle: 0 } }),
    );
    renderPanel();
    dropPhoto();
    await screen.findByRole("img", { name: "Working copy of grandma.jpg" });

    fireEvent.click(screen.getByRole("button", { name: "Rotate" }));

    expect(await screen.findByText("800 × 1200 px · 8-bit")).toBeInTheDocument();
    expect(restoreService.setPhotoGeometry).toHaveBeenCalledWith("tok-1", { rotate90: 1, crop: null, angle: 0 });
    expect(screen.getByRole("img", { name: "Working copy of grandma.jpg" })).toHaveAttribute(
      "src",
      "/api/v1/restore/analysis/tok-1/preview.jpg?v=2",
    );
  });

  it("shows the server error when the photo cannot be analyzed", async () => {
    vi.mocked(restoreService.analyzePhoto).mockRejectedValue(new Error("Unsupported image format"));
    renderPanel();

    dropPhoto();

    expect(await screen.findByRole("alert")).toHaveTextContent("Unsupported image format");
    expect(screen.queryByRole("button", { name: "Rotate" })).not.toBeInTheDocument();
  });

  it("rejects a photo over the upload limit before uploading it", async () => {
    const queryClient = renderPanel();
    await waitFor(() => expect(queryClient.getQueryData(["engine"])).toEqual(ENGINE_INFO));

    dropPhoto(new File([new Uint8Array(2 * 1024 * 1024)], "huge.tif", { type: "image/tiff" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("2 MB");
    expect(restoreService.analyzePhoto).not.toHaveBeenCalled();
  });
});

describe("versionedUrl", () => {
  it("adds the revision as a query parameter", () => {
    expect(versionedUrl("/a/preview.jpg", 3)).toBe("/a/preview.jpg?v=3");
    expect(versionedUrl("/a/preview.jpg?x=1", 3)).toBe("/a/preview.jpg?x=1&v=3");
  });
});
