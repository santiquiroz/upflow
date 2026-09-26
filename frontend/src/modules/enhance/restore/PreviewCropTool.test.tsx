import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "../../../lib/api";
import type { JobResponse } from "../../../lib/apiTypes";
import { createJobQueueStore } from "../../../lib/jobQueueStore";
import type { CreateRestoreJobParams } from "../../../services/restore";
import { PreviewCropTool, type PreviewCropToolProps } from "./PreviewCropTool";
import type { PreviewBeforeDeps } from "./previewBefore";
import { makeCompletedRestoreJob, makeRestoreMetadata } from "./restoreTestFixtures";

vi.mock("../../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../lib/api")>();
  return { ...actual, getJob: vi.fn(), cancelJob: vi.fn() };
});

const PARAMS: CreateRestoreJobParams = {
  source: { token: "tok-1" },
  steps: ["repair", "denoise"],
  options: { preset: "gentle", repair: { sensitivity: 0.5 } },
  scale: 1,
  modelId: null,
  device: null,
  outputFormat: "png",
};
const PREVIEW_URL = "/api/v1/restore/analysis/tok-1/preview.jpg?v=0";
const PREVIEW_JOB = makeCompletedRestoreJob(
  makeRestoreMetadata({ artifacts: ["preview"], previewCrop: [344, 144, 512, 512] }),
);

function setup(overrides: Partial<PreviewCropToolProps> = {}) {
  const createJob = vi.fn().mockResolvedValue({ jobId: "job-1", status: "queued" } as JobResponse);
  const queue = createJobQueueStore();
  const beforeDeps: PreviewBeforeDeps = { cut: vi.fn().mockResolvedValue("blob:before"), release: vi.fn() };
  const props: PreviewCropToolProps = {
    previewUrl: PREVIEW_URL,
    alt: "Working copy of grandma.jpg",
    originalName: "grandma.jpg",
    workingSize: { width: 1200, height: 800 },
    canRun: true,
    settingsKey: "a",
    buildRequest: vi.fn().mockResolvedValue({ params: PARAMS, fileName: "grandma.jpg" }),
    jobDeps: { createJob, pollIntervalMs: 10, queue },
    beforeDeps,
    ...overrides,
  };
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const wrap = (element: ReactElement) => <QueryClientProvider client={queryClient}>{element}</QueryClientProvider>;
  const view = render(wrap(<PreviewCropTool {...props} />));
  return { props, createJob, queue, beforeDeps, rerender: (next: PreviewCropToolProps) => view.rerender(wrap(<PreviewCropTool {...next} />)) };
}

function openTool() {
  fireEvent.click(screen.getByRole("button", { name: "Choose an area" }));
}

function drag(from: [number, number], to: [number, number]) {
  const overlay = screen.getByTestId("restore-preview-area-overlay");
  vi.spyOn(overlay, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    width: 600,
    height: 400,
    right: 600,
    bottom: 400,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  });
  fireEvent.pointerDown(overlay, { clientX: from[0], clientY: from[1], pointerId: 1 });
  fireEvent.pointerMove(overlay, { clientX: to[0], clientY: to[1], pointerId: 1 });
  fireEvent.pointerUp(overlay, { pointerId: 1 });
}

function runPreview() {
  fireEvent.click(screen.getByRole("button", { name: "Preview this area" }));
}

// jsdom no trae PointerEvent: sin esto los eventos llegan sin clientX/clientY.
class TestPointerEvent extends MouseEvent {
  readonly pointerId: number;

  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 0;
  }
}

beforeEach(() => {
  vi.stubGlobal("PointerEvent", TestPointerEvent);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.mocked(api.getJob).mockReset();
});

describe("PreviewCropTool", () => {
  it("stays closed until the user asks for it", () => {
    setup();
    expect(screen.getByRole("heading", { name: "Preview this area" })).toBeInTheDocument();
    expect(screen.queryByTestId("restore-preview-area-overlay")).not.toBeInTheDocument();
  });

  it("starts on the largest area at the center of the photo", () => {
    setup();
    openTool();
    expect(screen.getByText("512 × 512 px")).toBeInTheDocument();
  });

  it("lets the user drag a smaller area", () => {
    setup();
    openTool();
    drag([100, 100], [200, 150]);
    expect(screen.getByText("200 × 100 px")).toBeInTheDocument();
  });

  it("runs the same chain on the chosen area only, as its own queued job", async () => {
    vi.mocked(api.getJob).mockResolvedValue({ jobId: "job-1", status: "running" } as JobResponse);
    const { createJob, queue } = setup();
    openTool();
    drag([100, 100], [200, 150]);

    runPreview();

    await waitFor(() => expect(createJob).toHaveBeenCalled());
    expect(createJob).toHaveBeenCalledWith({ ...PARAMS, options: { ...PARAMS.options, preview_crop: [200, 200, 200, 100] } });
    expect(queue.getSnapshot()[0]).toEqual(expect.objectContaining({ fileName: "grandma.jpg (preview area)" }));
  });

  it("sends nothing when the settings could not be prepared", async () => {
    const buildRequest = vi.fn().mockResolvedValue(null);
    const { createJob } = setup({ buildRequest });
    openTool();

    runPreview();

    await waitFor(() => expect(buildRequest).toHaveBeenCalled());
    expect(createJob).not.toHaveBeenCalled();
  });

  it("can't run while restoring is not possible", () => {
    setup({ canRun: false });
    openTool();
    expect(screen.getByRole("button", { name: "Preview this area" })).toBeDisabled();
  });

  it("compares the area before and after once the preview finishes", async () => {
    vi.mocked(api.getJob).mockResolvedValue(PREVIEW_JOB);
    const { beforeDeps } = setup();
    openTool();
    runPreview();

    expect(await screen.findByAltText("Preview area after: grandma.jpg")).toHaveAttribute(
      "src",
      "/api/v1/jobs/job-1/artifacts/preview",
    );
    expect(await screen.findByAltText("Preview area before: grandma.jpg")).toHaveAttribute("src", "blob:before");
    expect(beforeDeps.cut).toHaveBeenCalledWith(PREVIEW_URL, [344, 144, 512, 512], { width: 1200, height: 800 });
  });

  it("says the preview is out of date after the settings change", async () => {
    vi.mocked(api.getJob).mockResolvedValue(PREVIEW_JOB);
    const { props, rerender } = setup();
    openTool();
    runPreview();
    await screen.findByAltText("Preview area after: grandma.jpg");
    expect(screen.queryByText(/The settings changed since this preview/)).not.toBeInTheDocument();

    rerender({ ...props, settingsKey: "b" });

    expect(screen.getByText(/The settings changed since this preview/)).toBeInTheDocument();
  });
});
