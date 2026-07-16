// Mirrors app/schemas.py response shapes. Field names follow the camelCase
// serialization_alias each Pydantic model uses on the wire, not the Python
// snake_case attribute names.

export type JobStatus = "queued" | "running" | "completed" | "failed";

export type StageStatus = "pending" | "active" | "done";

// Mirrors app/services/progress.py::Stage (asdict()'d into job.metadata.stages).
export interface JobStage {
  key: string;
  label: string;
  weight: number;
  status: StageStatus;
}

// Known job.metadata keys written by app/services/progress.py and
// video_upscaler.py. The index signature keeps this permissive for the
// remaining ad-hoc metadata (duration, hasAudio, outputWidth, ...) callers
// read defensively rather than relying on a fully-typed dictionary.
export interface JobMetadata {
  stage?: string;
  stages?: JobStage[];
  stageStartedAt?: string;
  framesDone?: number;
  framesTotal?: number | null;
  interpFramesTotal?: number | null;
  outputFps?: string;
  [key: string]: unknown;
}

export interface CreateJobResponse {
  jobId: string;
  status: JobStatus;
  statusUrl: string;
  downloadUrl: string | null;
}

export interface JobResponse {
  jobId: string;
  status: JobStatus;
  originalFilename: string;
  modelName: string;
  scale: number;
  outputFormat: string;
  modelId: string | null;
  device: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  error: string | null;
  metadata: JobMetadata;
  progressPct: number | null;
  downloadUrl: string | null;
}

export interface VideoJobResponse {
  jobId: string;
  status: JobStatus;
  originalFilename: string;
  modelName: string;
  scale: number;
  outputContainer: string;
  videoCodec: string;
  videoPreset: string;
  crf: number;
  keepAudio: boolean;
  fpsMultiplier: number;
  targetFps: string | null;
  audioEnhance: string | null;
  modelId: string | null;
  device: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  error: string | null;
  metadata: JobMetadata;
  progressPct: number | null;
  downloadUrl: string | null;
}

export interface SupportedModelResponse {
  key: string;
  label: string;
  category: string;
  description: string;
  scales: number[];
}

export interface VideoProfileResponse {
  key: string;
  label: string;
  category: string;
  description: string;
  modelKey: string;
  scale: number;
  videoCodec: string;
  videoPreset: string;
  crf: number;
  keepAudio: boolean;
}

export interface EngineInfoResponse {
  engine: string;
  configuredBinary: string;
  configuredModelsDir: string;
  available: boolean;
  defaultModel: string;
  allowedScales: number[];
  supportedModels: SupportedModelResponse[];
  videoProfiles: VideoProfileResponse[];
  ffmpegAvailable: boolean;
}

export interface HealthResponse {
  status: "ok";
  engine: string;
  gpuConcurrency: number;
  queueDepth: number;
  videoQueueDepth: number;
}

// "auto" is never returned by GET /devices (real hardware only) -- it's a
// frontend-only sentinel for the synthetic "Auto" DevicePicker entry, mirrored
// server-side by app.services.devices_service.AUTO_DEVICE_ID.
export type DeviceKind = "cpu" | "gpu" | "npu" | "auto";
export type DeviceBackend = "cpu" | "directml" | "winml" | "auto";

export interface DeviceInfoResponse {
  id: string;
  kind: DeviceKind;
  name: string;
  backend: DeviceBackend;
}

export interface DevicesResponse {
  devices: DeviceInfoResponse[];
  defaultDeviceId: string;
}

export interface ModelResponse {
  id: string;
  name: string;
  kind: string;
  source: string;
  scale: number | null;
  arch: string | null;
  sizeBytes: number;
  status: string;
  error: string | null;
}

export interface ModelsResponse {
  models: ModelResponse[];
}

export interface HfModelSearchResultResponse {
  id: string;
  author: string | null;
  pipelineTag: string | null;
  downloads: number;
  likes: number;
  tags: string[];
}

export interface ModelSearchResponse {
  results: HfModelSearchResultResponse[];
}

export interface CreateInstallResponse {
  installId: string;
  statusUrl: string;
}

// Mirrors app/schemas.py::UpdateCheckResponse. Backend always answers 200:
// on any failure (offline, rate-limit, bad JSON) it returns updateAvailable
// false with a non-null error instead of throwing.
export interface UpdateCheck {
  currentVersion: string;
  latestVersion: string | null;
  updateAvailable: boolean;
  releaseUrl: string | null;
  publishedAt: string | null;
  checkedAt: string;
  error: string | null;
}

export interface InstallStatusResponse {
  installId: string;
  repoId: string;
  status: string;
  progressPct: number | null;
  modelId: string | null;
  error: string | null;
}
