import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { JobResponse, VideoJobResponse } from "../lib/apiTypes";
import { JobCard } from "./JobCard";

const BASE_JOB: JobResponse = {
  jobId: "job-1",
  status: "queued",
  originalFilename: "photo.png",
  modelName: "realesrgan-x4plus",
  scale: 4,
  outputFormat: "png",
  modelId: "realesrgan-x4plus",
  device: "dml:0",
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  error: null,
  downloadUrl: null,
};

const BASE_VIDEO_JOB: VideoJobResponse = {
  jobId: "vid-1",
  status: "queued",
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
  downloadUrl: null,
};

describe("JobCard", () => {
  it("shows a placeholder when idle", () => {
    render(<JobCard phase="idle" />);

    expect(screen.getByText(/select a file/i)).toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });

  it("shows an indeterminate progress bar and the file name while uploading", () => {
    render(<JobCard phase="uploading" fileName="photo.png" />);

    expect(screen.getByRole("progressbar")).toBeInTheDocument();
    expect(screen.getByText(/uploading/i)).toBeInTheDocument();
    expect(screen.getByText("photo.png")).toBeInTheDocument();
  });

  it("shows a queued state with a progress indicator", () => {
    render(<JobCard phase="queued" job={BASE_JOB} />);

    expect(screen.getByRole("progressbar")).toBeInTheDocument();
    expect(screen.getByText(/queued/i)).toBeInTheDocument();
  });

  it("shows a running state with a progress indicator", () => {
    render(<JobCard phase="running" job={{ ...BASE_JOB, status: "running" }} />);

    expect(screen.getByRole("progressbar")).toBeInTheDocument();
    expect(screen.getByText(/processing/i)).toBeInTheDocument();
  });

  it("shows the current pipeline stage for a running video job", () => {
    const job: VideoJobResponse = {
      ...BASE_VIDEO_JOB,
      status: "running",
      metadata: { stage: "upscaling_frames" },
    };
    render(<JobCard phase="running" job={job} />);

    expect(screen.getByText(/upscaling frames/i)).toBeInTheDocument();
  });

  it("shows only the generic Processing label when a running video job has no stage yet", () => {
    const job: VideoJobResponse = { ...BASE_VIDEO_JOB, status: "running", metadata: {} };
    render(<JobCard phase="running" job={job} />);

    expect(screen.getByText(/processing/i)).toBeInTheDocument();
  });

  it("shows a preview and a download link when completed", () => {
    const job: JobResponse = { ...BASE_JOB, status: "completed", downloadUrl: "/api/v1/jobs/job-1/download" };
    render(<JobCard phase="completed" job={job} />);

    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    const link = screen.getByRole("link", { name: /download/i });
    expect(link).toHaveAttribute("href", "/api/v1/jobs/job-1/download");
    expect(screen.getByRole("img", { name: /photo\.png/i })).toHaveAttribute(
      "src",
      "/api/v1/jobs/job-1/download",
    );
  });

  it("shows an error message when failed", () => {
    const job: JobResponse = { ...BASE_JOB, status: "failed", error: "Model crashed" };
    render(<JobCard phase="failed" job={job} />);

    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("Model crashed");
  });

  it("shows the error when idle but an upload-level errorMessage is set (upload rejected before a job existed)", () => {
    render(<JobCard phase="idle" errorMessage="Job queue is full; try again later" />);

    expect(screen.getByRole("alert")).toHaveTextContent("Job queue is full; try again later");
    expect(screen.queryByText(/select a file/i)).not.toBeInTheDocument();
  });

  it("prefers an explicit errorMessage over the job error when failed", () => {
    const job: JobResponse = { ...BASE_JOB, status: "failed", error: "Model crashed" };
    render(<JobCard phase="failed" job={job} errorMessage="Upload rejected: file too large" />);

    expect(screen.getByRole("alert")).toHaveTextContent("Upload rejected: file too large");
  });

  it("renders scale as a tabular number in the completed state", () => {
    const job: JobResponse = { ...BASE_JOB, status: "completed", downloadUrl: "/download", scale: 4 };
    render(<JobCard phase="completed" job={job} />);

    const scaleValue = screen.getByText("4x");
    expect(scaleValue).toHaveClass("font-mono-tabular");
  });

  it("shows a download link without an image preview for a completed video job", () => {
    const job: VideoJobResponse = {
      ...BASE_VIDEO_JOB,
      status: "completed",
      downloadUrl: "/api/v1/video/jobs/vid-1/download",
    };
    render(<JobCard phase="completed" job={job} />);

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    const link = screen.getByRole("link", { name: /download/i });
    expect(link).toHaveAttribute("href", "/api/v1/video/jobs/vid-1/download");
    expect(screen.getByText("2x")).toHaveClass("font-mono-tabular");
  });

  it("shows the normalized outputFps as a tabular number when present in metadata", () => {
    const job: VideoJobResponse = {
      ...BASE_VIDEO_JOB,
      status: "completed",
      downloadUrl: "/download",
      metadata: { outputFps: "24000/1001" },
    };
    render(<JobCard phase="completed" job={job} />);

    const fpsValue = screen.getByText("23.98");
    expect(fpsValue).toHaveClass("font-mono-tabular");
  });

  it("omits the FPS row when outputFps is absent from metadata", () => {
    const job: VideoJobResponse = { ...BASE_VIDEO_JOB, status: "completed", downloadUrl: "/download", metadata: {} };
    render(<JobCard phase="completed" job={job} />);

    expect(screen.queryByText(/fps/i)).not.toBeInTheDocument();
  });
});
