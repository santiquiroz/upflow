from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


@dataclass(slots=True)
class UpscaleJob:
    source_path: Path
    original_filename: str
    model_name: str
    scale: int
    output_format: str
    model_id: str | None = None
    device: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class UpdateStatus:
    current_version: str
    latest_version: str | None
    update_available: bool
    release_url: str | None
    published_at: str | None
    checked_at: datetime
    error: str | None


@dataclass(slots=True)
class VideoUpscaleJob:
    source_path: Path
    original_filename: str
    model_name: str
    scale: int
    output_container: str
    video_codec: str
    video_preset: str
    crf: int
    keep_audio: bool
    fps_multiplier: int = 1
    target_fps: str | None = None
    audio_enhance: str | None = None
    audio_restore: str | None = None
    model_id: str | None = None
    device: str | None = None
    # Upscale runtime override (SP11): None|auto -> Auto rule; ncnn|onnx force one.
    backend: str | None = None
    # Video encoder (SP12): "software" (libx264/libx265, default) | "auto" (pick a
    # hardware encoder AMF/NVENC/QSV by the job's GPU, fall back to software).
    video_encoder: str = "software"
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AudioJob:
    source_path: Path
    original_filename: str
    denoise: str | None = None
    restore: str | None = None
    device: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
