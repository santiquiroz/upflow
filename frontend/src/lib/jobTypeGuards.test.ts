import { describe, expect, it } from "vitest";
import type { AudioJob, GenerationJob, JobResponse, VideoJobResponse } from "./apiTypes";
import { isGenerationJob } from "./jobTypeGuards";

const IMAGE_JOB: JobResponse = {
  jobId: "job-1",
  status: "queued",
  originalFilename: "photo.png",
  modelName: "realesrgan-x4plus",
  scale: 4,
  outputFormat: "png",
  modelId: "realesrgan-x4plus",
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  error: null,
  metadata: {},
  progressPct: null,
  downloadUrl: null,
};

const VIDEO_JOB: VideoJobResponse = {
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
  audioRestore: null,
  interpEngine: "rife",
  modelId: "realesr-animevideov3-x2",
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  error: null,
  metadata: {},
  progressPct: null,
  downloadUrl: null,
};

const AUDIO_JOB: AudioJob = {
  id: "audio-1",
  status: "queued",
  originalFilename: "clip.wav",
  denoise: "rnnoise",
  restore: null,
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  progressPct: null,
  stages: null,
  error: null,
  downloadUrl: null,
};

const GENERATION_JOB: GenerationJob = {
  id: "gen-1",
  status: "queued",
  prompt: "a red apple",
  negativePrompt: null,
  modelId: "gen--amd--sd15",
  steps: 25,
  guidance: 7.5,
  width: 512,
  height: 512,
  seed: null,
  device: null,
  autoUpscale: false,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  progressPct: null,
  stages: null,
  error: null,
  downloadUrl: null,
};

describe("isGenerationJob", () => {
  it("recognizes a generation job", () => {
    expect(isGenerationJob(GENERATION_JOB)).toBe(true);
  });

  it("rejects an image job", () => {
    expect(isGenerationJob(IMAGE_JOB)).toBe(false);
  });

  it("rejects a video job", () => {
    expect(isGenerationJob(VIDEO_JOB)).toBe(false);
  });

  it("rejects an audio job", () => {
    expect(isGenerationJob(AUDIO_JOB)).toBe(false);
  });
});
