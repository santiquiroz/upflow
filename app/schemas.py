from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import JobStatus
from app.services.capability_probe import LeverStatus


class CreateJobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    status: JobStatus
    status_url: str = Field(serialization_alias="statusUrl")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class JobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    model_name: str = Field(serialization_alias="modelName")
    scale: int
    output_format: str = Field(serialization_alias="outputFormat")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    device: str | None = None
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class VideoJobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    model_name: str = Field(serialization_alias="modelName")
    scale: int
    output_container: str = Field(serialization_alias="outputContainer")
    video_codec: str = Field(serialization_alias="videoCodec")
    video_preset: str = Field(serialization_alias="videoPreset")
    crf: int
    keep_audio: bool = Field(serialization_alias="keepAudio")
    fps_multiplier: int = Field(serialization_alias="fpsMultiplier")
    target_fps: str | None = Field(default=None, serialization_alias="targetFps")
    audio_enhance: str | None = Field(default=None, serialization_alias="audioEnhance")
    audio_restore: str | None = Field(default=None, serialization_alias="audioRestore")
    audio_track_indices: list[int] | None = Field(default=None, serialization_alias="audioTrackIndices")
    keep_subtitles: bool = Field(default=False, serialization_alias="keepSubtitles")
    audio_output_format: str = Field(default="auto", serialization_alias="audioOutputFormat")
    interp_engine: str = Field(default="rife", serialization_alias="interpEngine")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    device: str | None = None
    backend: str | None = None
    video_encoder: str = Field(default="auto", serialization_alias="videoEncoder")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class AudioJobResponse(BaseModel):
    id: str
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    denoise: str | None = None
    restore: str | None = None
    device: str | None = None
    output_format: str = Field(default="flac", serialization_alias="outputFormat")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    stages: list[dict[str, Any]] | None = None
    error: str | None = None
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class AudioCapabilitiesResponse(BaseModel):
    denoise_modes: list[str] = Field(serialization_alias="denoiseModes")
    restore_available: bool = Field(serialization_alias="restoreAvailable")
    restore_modes: list[str] = Field(default_factory=list, serialization_alias="restoreModes")


class VideoCapabilitiesResponse(BaseModel):
    interp_engines: list[str] = Field(default_factory=list, serialization_alias="interpEngines")


class LeverResponse(BaseModel):
    id: str
    label: str
    status: LeverStatus
    detail: str
    fixable: bool


class CapabilitiesResponse(BaseModel):
    levers: list[LeverResponse]


class FixLeverResponse(BaseModel):
    lever: LeverResponse


class AudioTrackResponse(BaseModel):
    index: int
    codec: str
    channels: int
    is_default: bool = Field(serialization_alias="isDefault")
    language: str | None = None


class SubtitleTrackResponse(BaseModel):
    index: int
    codec: str
    language: str | None = None


class AnalyzeVideoResponse(BaseModel):
    upload_token: str = Field(serialization_alias="uploadToken")
    audio_tracks: list[AudioTrackResponse] = Field(serialization_alias="audioTracks")
    subtitle_tracks: list[SubtitleTrackResponse] = Field(serialization_alias="subtitleTracks")


class SupportedModelResponse(BaseModel):
    key: str
    label: str
    category: str
    description: str
    scales: list[int]


class VideoProfileResponse(BaseModel):
    key: str
    label: str
    category: str
    description: str
    model_key: str = Field(serialization_alias="modelKey")
    scale: int
    video_codec: str = Field(serialization_alias="videoCodec")
    video_preset: str = Field(serialization_alias="videoPreset")
    crf: int
    keep_audio: bool = Field(serialization_alias="keepAudio")


class EngineInfoResponse(BaseModel):
    engine: str
    configured_binary: str = Field(serialization_alias="configuredBinary")
    configured_models_dir: str = Field(serialization_alias="configuredModelsDir")
    available: bool
    default_model: str = Field(serialization_alias="defaultModel")
    allowed_scales: list[int] = Field(serialization_alias="allowedScales")
    supported_models: list[SupportedModelResponse] = Field(serialization_alias="supportedModels")
    video_profiles: list[VideoProfileResponse] = Field(serialization_alias="videoProfiles")
    ffmpeg_available: bool = Field(serialization_alias="ffmpegAvailable")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    engine: str
    gpu_concurrency: int = Field(serialization_alias="gpuConcurrency")
    queue_depth: int = Field(serialization_alias="queueDepth")
    video_queue_depth: int = Field(serialization_alias="videoQueueDepth")


class DeviceInfoResponse(BaseModel):
    id: str
    kind: Literal["cpu", "gpu", "npu"]
    name: str
    backend: Literal["cpu", "directml", "winml"]


class DevicesResponse(BaseModel):
    devices: list[DeviceInfoResponse]
    default_device_id: str = Field(serialization_alias="defaultDeviceId")


class ModelResponse(BaseModel):
    id: str
    name: str
    kind: str
    source: str
    scale: int | None = None
    arch: str | None = None
    size_bytes: int = Field(serialization_alias="sizeBytes")
    status: str
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelResponse]


class HfModelSearchResultResponse(BaseModel):
    id: str
    author: str | None = None
    pipeline_tag: str | None = Field(default=None, serialization_alias="pipelineTag")
    downloads: int
    likes: int
    tags: list[str]


class ModelSearchResponse(BaseModel):
    results: list[HfModelSearchResultResponse]


class InstallModelRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    repo_id: str = Field(alias="repoId")


class CreateInstallResponse(BaseModel):
    install_id: str = Field(serialization_alias="installId")
    status_url: str = Field(serialization_alias="statusUrl")


class UpdateCheckResponse(BaseModel):
    current_version: str = Field(serialization_alias="currentVersion")
    latest_version: str | None = Field(default=None, serialization_alias="latestVersion")
    update_available: bool = Field(serialization_alias="updateAvailable")
    release_url: str | None = Field(default=None, serialization_alias="releaseUrl")
    published_at: str | None = Field(default=None, serialization_alias="publishedAt")
    checked_at: datetime = Field(serialization_alias="checkedAt")
    error: str | None = None


class InstallStatusResponse(BaseModel):
    install_id: str = Field(serialization_alias="installId")
    repo_id: str = Field(serialization_alias="repoId")
    status: str
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    error: str | None = None


class CreateGenerationJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str | None = Field(default=None, alias="negativePrompt", max_length=2000)
    model_id: str = Field(alias="modelId")
    steps: int = Field(default=25, ge=1, le=100)
    guidance: float = Field(default=7.5, ge=0, le=30)
    width: int = Field(default=512, ge=64, le=1024, multiple_of=64)
    height: int = Field(default=512, ge=64, le=1024, multiple_of=64)
    seed: int | None = Field(default=None, ge=0)
    device: str | None = None
    auto_upscale: bool = Field(default=False, alias="autoUpscale")
    upscale_model_name: str | None = Field(default=None, alias="upscaleModelName")
    upscale_scale: int | None = Field(default=None, alias="upscaleScale", ge=2, le=4)
    upscale_model_id: str | None = Field(default=None, alias="upscaleModelId")


class GenerationJobResponse(BaseModel):
    id: str
    status: JobStatus
    prompt: str
    negative_prompt: str | None = Field(default=None, serialization_alias="negativePrompt")
    model_id: str = Field(serialization_alias="modelId")
    steps: int
    guidance: float
    width: int
    height: int
    seed: int | None = None
    device: str | None = None
    auto_upscale: bool = Field(default=False, serialization_alias="autoUpscale")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    stages: list[dict[str, Any]] | None = None
    error: str | None = None
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class GenerationModelSummary(BaseModel):
    id: str
    name: str


class GenerationCapabilitiesResponse(BaseModel):
    available: bool
    reason: str | None = None
    models: list[GenerationModelSummary] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    cpu_only: bool = Field(default=False, serialization_alias="cpuOnly")


class CpuFallbackReportResponse(BaseModel):
    model_id: str = Field(serialization_alias="modelId")
    device_id: str = Field(serialization_alias="deviceId")
    hot_ops: list[str] = Field(serialization_alias="hotOps")
    clean: bool


class OnnxDiagnosticEntryResponse(BaseModel):
    model_id: str = Field(serialization_alias="modelId")
    device_id: str = Field(serialization_alias="deviceId")
    report: CpuFallbackReportResponse | None = None


class OnnxDiagnosticsResponse(BaseModel):
    entries: list[OnnxDiagnosticEntryResponse]


class ScanOnnxDiagnosticResponse(BaseModel):
    report: CpuFallbackReportResponse
