import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { JobQueueEntry } from "../hooks/useJobQueue";
import type { JobStage, VideoJobResponse } from "../lib/apiTypes";
import { JobDetailModal } from "./JobDetailModal";

function stage(key: string, label: string, status: JobStage["status"]): JobStage {
  return { key, label, weight: 0.25, status };
}

const BASE_VIDEO_JOB: VideoJobResponse = {
  jobId: "vid-1",
  status: "running",
  originalFilename: "clip.mp4",
  modelName: "realesr-animevideov3-x2",
  scale: 2,
  outputContainer: "mp4",
  videoCodec: "libx264",
  videoPreset: "medium",
  crf: 17,
  keepAudio: true,
  fpsMultiplier: 1,
  targetFps: null,
  audioEnhance: null,
  modelId: "realesr-animevideov3-x2",
  device: "dml:0",
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  error: null,
  metadata: {},
  progressPct: null,
  downloadUrl: null,
};

function buildEntry(overrides: Partial<VideoJobResponse> = {}, entryOverrides: Partial<JobQueueEntry> = {}): JobQueueEntry {
  const job: VideoJobResponse = { ...BASE_VIDEO_JOB, ...overrides };
  return {
    id: job.jobId,
    kind: "video",
    fileName: job.originalFilename,
    createdAt: 1,
    status: job.status,
    downloadUrl: job.downloadUrl,
    errorMessage: null,
    job,
    ...entryOverrides,
  };
}

describe("JobDetailModal", () => {
  it("renders the file name and job summary fields", () => {
    render(<JobDetailModal entry={buildEntry()} onClose={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "clip.mp4" })).toBeInTheDocument();
    expect(screen.getByText("Video")).toBeInTheDocument();
    expect(screen.getByText("realesr-animevideov3-x2")).toBeInTheDocument();
    expect(screen.getByText("dml:0")).toBeInTheDocument();
    expect(screen.getByText("2x")).toBeInTheDocument();
  });

  it("renders numeric details as tabular mono but leaves text details unstyled", () => {
    render(<JobDetailModal entry={buildEntry()} onClose={vi.fn()} />);

    expect(screen.getByText("2x")).toHaveClass("font-mono-tabular");
    expect(screen.getByText("realesr-animevideov3-x2")).not.toHaveClass("font-mono-tabular");
  });

  it("renders a vertical stepper from job.metadata.stages", () => {
    const entry = buildEntry({
      metadata: {
        stages: [
          stage("probing", "Probing video", "done"),
          stage("upscaling_frames", "Upscaling frames", "active"),
          stage("encoding_video", "Encoding video", "pending"),
        ],
      },
    });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText("Probing video")).toBeInTheDocument();
    expect(screen.getByText("Upscaling frames")).toBeInTheDocument();
    expect(screen.getByText("Encoding video")).toBeInTheDocument();
  });

  it("shows a determinate progress bar with the percentage when progressPct is available", () => {
    const entry = buildEntry({ progressPct: 42 });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    const bar = screen.getByRole("progressbar", { name: "Progress" });
    expect(bar).toHaveAttribute("aria-valuenow", "42");
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("shows an indeterminate progress bar when progress is not yet available", () => {
    const entry = buildEntry({ progressPct: null });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    const bar = screen.getByRole("progressbar", { name: "Progress" });
    expect(bar).toHaveAttribute("aria-busy", "true");
    expect(bar).not.toHaveAttribute("aria-valuenow");
  });

  it("shows frames X / Y as tabular numbers when frame counts are present", () => {
    const entry = buildEntry({ progressPct: 20, metadata: { framesDone: 120, framesTotal: 600 } });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText(/frames/)).toHaveTextContent("120 / 600 frames");
    expect(screen.getByText("120")).toHaveClass("font-mono-tabular");
    expect(screen.getByText("600")).toHaveClass("font-mono-tabular");
  });

  it("uses interpFramesTotal as the denominator during interpolation so the ratio stays valid", () => {
    const entry = buildEntry({
      progressPct: 90,
      metadata: { stage: "interpolating_frames", framesDone: 800, framesTotal: 400, interpFramesTotal: 800 },
    });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText(/frames/)).toHaveTextContent("800 / 800 frames");
  });

  it("keeps using framesTotal during a normal upscaling stage", () => {
    const entry = buildEntry({
      progressPct: 40,
      metadata: { stage: "upscaling_frames", framesDone: 200, framesTotal: 400, interpFramesTotal: 800 },
    });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText(/frames/)).toHaveTextContent("200 / 400 frames");
  });

  it("omits the frames readout when framesTotal is unknown (VFR source)", () => {
    const entry = buildEntry({ progressPct: 20, metadata: { framesDone: 120, framesTotal: null } });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.queryByText(/frames/)).not.toBeInTheDocument();
  });

  it("shows the audio enhancement mode when configured", () => {
    const entry = buildEntry({ keepAudio: true, audioEnhance: "deepfilter" });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText("DeepFilterNet")).toBeInTheDocument();
  });

  it("shows Disabled for audio when the job dropped the audio track", () => {
    const entry = buildEntry({ keepAudio: false });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByText("Disabled")).toBeInTheDocument();
  });

  it("shows the failure message and hides the progress section when the job failed", () => {
    const entry = buildEntry({ status: "failed", error: "Model crashed" }, { status: "failed", errorMessage: "Model crashed" });

    render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Model crashed");
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });

  describe("ETA", () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(0);
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it("hides the ETA until a second poll establishes a rate", () => {
      const entry = buildEntry({ progressPct: 20 });

      render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

      expect(screen.queryByText(/ETA/)).not.toBeInTheDocument();
    });

    it("shows the ETA once a steady rate is established across polls", () => {
      const entry = buildEntry({ progressPct: 20 });
      const { rerender } = render(<JobDetailModal entry={entry} onClose={vi.fn()} />);

      vi.setSystemTime(10_000);
      rerender(<JobDetailModal entry={buildEntry({ progressPct: 40 })} onClose={vi.fn()} />);

      expect(screen.getByText(/ETA/)).toBeInTheDocument();
    });

    it("resets its sample buffer when a different job is shown", () => {
      const firstEntry = buildEntry({ progressPct: 20 }, { id: "vid-1" });
      const { rerender } = render(<JobDetailModal entry={firstEntry} onClose={vi.fn()} />);
      vi.setSystemTime(10_000);
      rerender(<JobDetailModal entry={buildEntry({ progressPct: 40 }, { id: "vid-1" })} onClose={vi.fn()} />);
      expect(screen.getByText(/ETA/)).toBeInTheDocument();

      const otherJobEntry = buildEntry({ jobId: "vid-2", progressPct: 5 }, { id: "vid-2" });
      rerender(<JobDetailModal entry={otherJobEntry} onClose={vi.fn()} />);

      expect(screen.queryByText(/ETA/)).not.toBeInTheDocument();
    });
  });

  it("closes when Escape is pressed", () => {
    const onClose = vi.fn();
    render(<JobDetailModal entry={buildEntry()} onClose={onClose} />);

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("closes when the Close button is clicked", () => {
    const onClose = vi.fn();
    render(<JobDetailModal entry={buildEntry()} onClose={onClose} />);

    fireEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
