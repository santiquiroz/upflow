import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "../../../lib/api";
import type { EngineInfoResponse, JobResponse } from "../../../lib/apiTypes";
import * as restoreService from "../../../services/restore";
import { PhotoRestorePanel } from "./PhotoRestorePanel";
import {
  makeAnalysis,
  makeCapabilities,
  makeCompletedRestoreJob,
  makeFace,
  makeRestoreMetadata,
} from "./restoreTestFixtures";

vi.mock("../../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../lib/api")>();
  return { ...actual, getEngineInfo: vi.fn(), getJob: vi.fn() };
});

vi.mock("../../../services/restore", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../services/restore")>();
  return {
    ...actual,
    analyzePhoto: vi.fn(),
    setPhotoGeometry: vi.fn(),
    uploadDamageMask: vi.fn(),
    getRestoreCapabilities: vi.fn(),
    createRestoreJob: vi.fn(),
  };
});

const ENGINE_INFO = { maxUploadMb: 1 } as EngineInfoResponse;

function renderPanel() {
  vi.mocked(api.getEngineInfo).mockResolvedValue(ENGINE_INFO);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <MemoryRouter>
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </MemoryRouter>
    );
  }
  render(<PhotoRestorePanel />, { wrapper: Wrapper });
  return queryClient;
}

function dropPhoto(file = new File(["x"], "grandma.jpg", { type: "image/jpeg" })) {
  const input = document.getElementById("restore-file-input") as HTMLInputElement;
  fireEvent.change(input, { target: { files: [file] } });
  return file;
}

beforeEach(() => {
  vi.mocked(restoreService.getRestoreCapabilities).mockResolvedValue(makeCapabilities());
});

afterEach(() => {
  vi.mocked(api.getEngineInfo).mockReset();
  vi.mocked(restoreService.analyzePhoto).mockReset();
  vi.mocked(restoreService.setPhotoGeometry).mockReset();
  vi.mocked(restoreService.getRestoreCapabilities).mockReset();
  vi.mocked(restoreService.createRestoreJob).mockReset();
  vi.mocked(api.getJob).mockReset();
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

describe("PhotoRestorePanel: summary and Restore", () => {
  const QUEUED = { jobId: "job-9", status: "queued" } as JobResponse;

  it("shows the summary after the analysis and restores the session's photo with the chosen fixes", async () => {
    vi.mocked(restoreService.analyzePhoto).mockResolvedValue(makeAnalysis());
    vi.mocked(restoreService.createRestoreJob).mockResolvedValue(QUEUED);
    vi.mocked(api.getJob).mockResolvedValue({ ...QUEUED, status: "running" } as JobResponse);
    renderPanel();
    dropPhoto();

    const restore = await screen.findByRole("button", { name: "Restore" });
    expect(screen.getByText("1 fix selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Customize" })).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(restore);

    await waitFor(() => expect(restoreService.createRestoreJob).toHaveBeenCalledTimes(1));
    expect(vi.mocked(restoreService.createRestoreJob).mock.calls[0][0]).toEqual({
      source: { token: "tok-1" },
      steps: ["repair"],
      options: { preset: "gentle", repair: { engine: "fast", sensitivity: 0.5, grow_px: 0 } },
      scale: 1,
      modelId: null,
      device: null,
      outputFormat: "png",
    });
    await waitFor(() => expect(api.getJob).toHaveBeenCalledWith("job-9"));
    expect(screen.getByRole("button", { name: "Restore" })).toBeDisabled();
  });

  it("shows the faces to restore and sends only the ones left checked", async () => {
    const portrait = { steps: ["faces"], options: { faces: { model: "gfpgan-v1.4", blend: 0.6 } } };
    const faces = [
      makeFace({ index: 0, eyePx: 48, sharpness: 0.02, enabled: true, blend: 0.6 }),
      makeFace({ index: 1, eyePx: 36, sharpness: 0.02, enabled: true, blend: 0.6, thumbnailUrl: null }),
    ];
    vi.mocked(restoreService.analyzePhoto).mockResolvedValue(
      makeAnalysis({
        proposedPreset: "portrait",
        proposedSteps: portrait.steps,
        proposedOptions: portrait.options,
        presetSelections: { portrait },
        faces,
      }),
    );
    vi.mocked(restoreService.createRestoreJob).mockResolvedValue(QUEUED);
    vi.mocked(api.getJob).mockResolvedValue({ ...QUEUED, status: "running" } as JobResponse);
    renderPanel();
    dropPhoto();

    fireEvent.click(await screen.findByRole("checkbox", { name: "Restore face 2" }));
    fireEvent.click(screen.getByRole("button", { name: "Restore" }));

    await waitFor(() => expect(restoreService.createRestoreJob).toHaveBeenCalledTimes(1));
    expect(vi.mocked(restoreService.createRestoreJob).mock.calls[0][0].options).toEqual({
      preset: "portrait",
      faces: { model: "gfpgan-v1.4", blend: 0.6, selected: [0], per_face: { "0": 0.6 } },
    });
  });

  it("has no face grid when face restoration is off", async () => {
    vi.mocked(restoreService.analyzePhoto).mockResolvedValue(makeAnalysis());
    renderPanel();
    dropPhoto();

    await screen.findByRole("button", { name: "Restore" });
    expect(screen.queryByRole("heading", { name: "Faces" })).not.toBeInTheDocument();
  });

  it("shows the comparison and downloads once the restoration finishes", async () => {
    const completed = makeCompletedRestoreJob(makeRestoreMetadata());
    vi.mocked(restoreService.analyzePhoto).mockResolvedValue(makeAnalysis());
    vi.mocked(restoreService.createRestoreJob).mockResolvedValue({ ...completed, status: "queued" });
    vi.mocked(api.getJob).mockResolvedValue(completed);
    renderPanel();
    dropPhoto();

    fireEvent.click(await screen.findByRole("button", { name: "Restore" }));

    expect(await screen.findByRole("heading", { name: "Result" })).toBeInTheDocument();
    expect(screen.getByAltText("Before: grandma.jpg")).toHaveAttribute(
      "src",
      "/api/v1/restore/analysis/tok-1/preview.jpg?v=1",
    );
    expect(screen.getByRole("link", { name: "Download restored" })).toHaveAttribute("download", "grandma_restored.png");
    expect(screen.getAllByText("Runs on your computer. Your photos are not uploaded anywhere.")).toHaveLength(1);
  });

  it("previews a chosen area with the same settings as its own job", async () => {
    vi.mocked(restoreService.analyzePhoto).mockResolvedValue(makeAnalysis());
    vi.mocked(restoreService.createRestoreJob).mockResolvedValue(QUEUED);
    vi.mocked(api.getJob).mockResolvedValue({ ...QUEUED, status: "running" } as JobResponse);
    renderPanel();
    dropPhoto();

    fireEvent.click(await screen.findByRole("button", { name: "Choose an area" }));
    fireEvent.click(screen.getByRole("button", { name: "Preview this area" }));

    await waitFor(() => expect(restoreService.createRestoreJob).toHaveBeenCalledTimes(1));
    expect(vi.mocked(restoreService.createRestoreJob).mock.calls[0][0]).toEqual(
      expect.objectContaining({
        source: { token: "tok-1" },
        steps: ["repair"],
        options: {
          preset: "gentle",
          repair: { engine: "fast", sensitivity: 0.5, grow_px: 0 },
          preview_crop: [344, 144, 512, 512],
        },
      }),
    );
    expect(screen.queryByRole("heading", { name: "Result" })).not.toBeInTheDocument();
  });

  it("applies the settings of the finished photo to more photos", async () => {
    const completed = makeCompletedRestoreJob(makeRestoreMetadata());
    vi.mocked(restoreService.analyzePhoto).mockResolvedValue(makeAnalysis());
    vi.mocked(restoreService.createRestoreJob).mockResolvedValue({ ...completed, status: "queued" });
    vi.mocked(api.getJob).mockResolvedValue(completed);
    renderPanel();
    dropPhoto();
    fireEvent.click(await screen.findByRole("button", { name: "Restore" }));
    await screen.findByRole("heading", { name: "Result" });

    const more = new File(["y"], "grandpa.jpg", { type: "image/jpeg" });
    fireEvent.change(screen.getByLabelText("Apply these settings to more photos"), { target: { files: [more] } });

    await waitFor(() => expect(restoreService.createRestoreJob).toHaveBeenCalledTimes(2));
    expect(vi.mocked(restoreService.createRestoreJob).mock.calls[1][0]).toEqual({
      source: { file: more },
      steps: ["repair"],
      options: { preset: "gentle", repair: { engine: "fast", sensitivity: 0.5, grow_px: 0 } },
      scale: 1,
      modelId: null,
      device: null,
      outputFormat: "png",
    });
    expect(await screen.findByText("1 photo added to the job queue.")).toBeInTheDocument();
  });

  it("says when the restore options can't be loaded", async () => {
    vi.mocked(restoreService.getRestoreCapabilities).mockRejectedValue(new Error("boom"));
    vi.mocked(restoreService.analyzePhoto).mockResolvedValue(makeAnalysis());
    renderPanel();
    dropPhoto();

    expect(await screen.findByText("Couldn't load the restore options.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Restore" })).not.toBeInTheDocument();
  });
});
