from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from app.api.routes import create_job, create_video_job, sanitize_filename
from app.config import Settings
from app.models import UpscaleJob, VideoUpscaleJob
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

FORBIDDEN_CHARS = ':<>"|?*'


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


def make_png_bytes(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


class FakeEngine(UpscaleEngine):
    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        return job.source_path


class FakeUpscaler:
    async def run(self, job: VideoUpscaleJob) -> Path:
        return job.source_path


class FakeMediaTools:
    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video"}]}


# ---------------------------------------------------------------------------
# 3.5 — ADS / reserved-name filenames (Windows)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("char", list(FORBIDDEN_CHARS))
def test_sanitize_filename_strips_each_forbidden_char(char: str) -> None:
    result = sanitize_filename(f"weird{char}name.png", default="upload.png")

    assert char not in result


def test_sanitize_filename_strips_all_forbidden_chars_at_once() -> None:
    result = sanitize_filename('my:file<name>"weird|name?*.png', default="upload.png")

    assert not any(char in result for char in FORBIDDEN_CHARS)


@pytest.mark.parametrize("reserved", ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"])
def test_sanitize_filename_escapes_reserved_device_name(reserved: str) -> None:
    result = sanitize_filename(f"{reserved}.png", default="upload.png")

    assert Path(result).stem.upper() != reserved


@pytest.mark.parametrize("reserved", ["con", "Con", "nul"])
def test_sanitize_filename_escapes_reserved_device_name_case_insensitively(reserved: str) -> None:
    result = sanitize_filename(f"{reserved}.txt", default="upload.png")

    assert Path(result).stem.upper() not in {"CON", "NUL"}


def test_sanitize_filename_escapes_reserved_device_name_without_extension() -> None:
    result = sanitize_filename("NUL", default="upload.png")

    assert result.upper() != "NUL"


def test_sanitize_filename_falls_back_to_default_when_stripped_to_empty() -> None:
    result = sanitize_filename('::<<>>""||??**', default="upload.png")

    assert result == "upload.png"


def test_sanitize_filename_keeps_a_normal_name_unchanged() -> None:
    result = sanitize_filename("holiday-photo (2024).png", default="upload.png")

    assert result == "holiday-photo (2024).png"


def test_sanitize_filename_strips_directory_components() -> None:
    result = sanitize_filename("../../etc/passwd", default="upload.png")

    assert "/" not in result
    assert ".." not in result


async def test_image_upload_with_forbidden_chars_lands_on_disk_sanitized(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    jobs = JobManager(settings, FakeEngine(), asyncio.Semaphore(settings.gpu_concurrency))

    response = await create_job(
        request=None,
        file=make_upload('weird:name<>"file.png', make_png_bytes()),
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
        jobs=jobs,
        storage=storage,
        settings=settings,
    )

    job = jobs.get_job(response.job_id)
    assert job is not None
    assert not any(char in job.source_path.name for char in FORBIDDEN_CHARS)
    assert job.source_path.exists()


async def test_image_upload_keeps_original_filename_as_display_metadata(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    jobs = JobManager(settings, FakeEngine(), asyncio.Semaphore(settings.gpu_concurrency))

    response = await create_job(
        request=None,
        file=make_upload("holiday-photo.png", make_png_bytes()),
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
        jobs=jobs,
        storage=storage,
        settings=settings,
    )

    job = jobs.get_job(response.job_id)
    assert job is not None
    assert job.original_filename == "holiday-photo.png"


async def test_video_upload_with_forbidden_chars_lands_on_disk_sanitized(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(
        settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(settings.gpu_concurrency)
    )

    response = await create_video_job(
        request=None,
        file=make_upload('clip:name<>"weird.mp4', b"fake-video-bytes"),
        profile_key="anime-balanced-2x",
        model_name=None,
        scale=None,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
        fps_multiplier=None,
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
    )

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert not any(char in job.source_path.name for char in FORBIDDEN_CHARS)
    assert job.source_path.exists()
