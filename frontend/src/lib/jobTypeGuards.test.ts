import { describe, expect, it } from "vitest";
import type {
  AudioJob,
  DownloadJob,
  GenerationJob,
  JobResponse,
  Shape3dJob,
  TranscribeJob,
  VideoJobResponse,
} from "./apiTypes";
import { isGenerationJob, jobKindOf } from "./jobTypeGuards";

const IMAGE_JOB: JobResponse = {
  ownerId: null,
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
  ownerId: null,
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
  ownerId: null,
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
  ownerId: null,
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

const TRANSCRIBE_JOB: TranscribeJob = {
  ownerId: null,
  id: "tr-1",
  status: "queued",
  originalFilename: "charla.mp4",
  modelId: "whisper-small",
  language: null,
  device: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  progressPct: null,
  text: null,
  error: null,
  downloadUrl: null,
};

const DOWNLOAD_JOB: DownloadJob = {
  ownerId: null,
  thenSeparate: false,
  followupJobIds: [],
  followupError: null,
  id: "dl-1",
  status: "queued",
  url: "https://example.com/x",
  maxHeight: 1080,
  audioOnly: false,
  audioFormat: "mp3",
  audioBitrateKbps: null,
  videoContainer: "mp4",
  mediaTitle: null,
  mediaUploader: null,
  extractor: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  progressPct: null,
  downloadedBytes: 0,
  totalBytes: null,
  outputFiles: [],
  outputDirectory: "",
  error: null,
};

const SHAPE3D_JOB: Shape3dJob = {
  id: "3d-1",
  status: "queued",
  prompt: "una maceta",
  printer: "ender-3",
  source: "mesh",
  code: null,
  retries: 0,
  targetMm: null,
  targetMmSource: null,
  targetMmReference: null,
  createdAt: "2026-01-01T00:00:00Z",
  startedAt: null,
  finishedAt: null,
  canPrint: null,
  sizeMm: null,
  triangleCount: null,
  blockers: [],
  advice: [],
  error: null,
  downloadUrl: null,
};

describe("jobKindOf", () => {
  it.each([
    [IMAGE_JOB, "image"],
    [VIDEO_JOB, "video"],
    [AUDIO_JOB, "audio"],
    [GENERATION_JOB, "generation"],
    [TRANSCRIBE_JOB, "transcribe"],
    [DOWNLOAD_JOB, "download"],
    [SHAPE3D_JOB, "shape3d"],
  ])("recognizes the %# family", (job, expected) => {
    expect(jobKindOf(job)).toBe(expected);
  });

  it("does not mistake a 3D job for a generation job, even though both carry a prompt", () => {
    expect(jobKindOf(SHAPE3D_JOB)).toBe("shape3d");
    expect(jobKindOf(GENERATION_JOB)).toBe("generation");
  });
});

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
