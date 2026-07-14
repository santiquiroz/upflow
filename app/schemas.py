from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models import JobStatus


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
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    error: str | None = None
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
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


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
