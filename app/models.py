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


TERMINAL_JOB_STATUSES = (JobStatus.completed, JobStatus.failed, JobStatus.cancelled)


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
    # Which audio streams to keep in the output (Fase A Task 2). None means
    # "keep_audio decides alone" (existing behavior); a list selects specific
    # ffprobe stream indices, consumed by the pipeline in Task 3.
    audio_track_indices: list[int] | None = None
    # Copy subtitle streams into the output (Fase A Task 2). Forces an mkv
    # container upgrade when the requested container can't carry subtitles
    # losslessly -- see VideoJobManager._resolve_output_container.
    keep_subtitles: bool = False
    # Frame-interpolation engine (Task 4.2): "rife" (default, always) or
    # "gmfss" (opt-in, much higher quality, 10x or more slower -- even higher
    # on short clips due to model load overhead). Only consulted when
    # interpolation is actually requested (fps_multiplier>1 or target_fps set).
    interp_engine: str = "rife"
    model_id: str | None = None
    device: str | None = None
    # Upscale runtime override (SP11): None|auto -> Auto rule; ncnn|onnx force one.
    backend: str | None = None
    # Video encoder (SP12): "auto" (default) picks a hardware encoder AMF/NVENC/QSV
    # by the job's GPU and falls back to software; "software" forces libx264/libx265.
    # Default is "auto" because software x265 slow at 4x costs ~112 min/episode vs
    # ~16 min on the GPU -- the software default was the dominant wall-time cost.
    video_encoder: str = "auto"
    # ffprobe output captured during job validation, reused by the pipeline so the
    # same file isn't probed twice. In-memory only: the API response is built field
    # by field, so this never serializes (it holds the absolute source path).
    probe: dict[str, Any] | None = None
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
