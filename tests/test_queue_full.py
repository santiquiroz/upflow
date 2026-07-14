from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from app.api.routes import create_job, create_video_job
from app.config import Settings
from app.exceptions import QueueFullError
from app.models import UpscaleJob, VideoUpscaleJob
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager


def make_settings(tmp_path: Path, max_queue_size: int = 1) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), MAX_QUEUE_SIZE=max_queue_size)


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
# 3.8 — unbounded queue → maxsize + QueueFullError
# ---------------------------------------------------------------------------


async def test_job_manager_queue_respects_configured_maxsize(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_queue_size=3)
    jobs = JobManager(settings, FakeEngine(), asyncio.Semaphore(settings.gpu_concurrency))

    assert jobs.queue.maxsize == 3


async def test_job_manager_raises_queue_full_error_when_queue_is_full(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_queue_size=1)
    jobs = JobManager(settings, FakeEngine(), asyncio.Semaphore(settings.gpu_concurrency))

    source_a = tmp_path / "a.png"
    source_a.write_bytes(make_png_bytes())
    source_b = tmp_path / "b.png"
    source_b.write_bytes(make_png_bytes())

    await jobs.create_job(
        source_path=source_a,
        original_filename="a.png",
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
    )

    with pytest.raises(QueueFullError):
        await jobs.create_job(
            source_path=source_b,
            original_filename="b.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
        )

    assert len(jobs.jobs) == 1, "the rejected job must not be registered as an orphan"


async def test_video_job_manager_raises_queue_full_error_when_queue_is_full(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_queue_size=1)
    video_jobs = VideoJobManager(
        settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(settings.gpu_concurrency)
    )

    source_a = tmp_path / "a.mp4"
    source_a.write_bytes(b"fake-video-bytes-a")
    source_b = tmp_path / "b.mp4"
    source_b.write_bytes(b"fake-video-bytes-b")

    await video_jobs.create_job(
        source_path=source_a,
        original_filename="a.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )

    with pytest.raises(QueueFullError):
        await video_jobs.create_job(
            source_path=source_b,
            original_filename="b.mp4",
            model_name="realesr-animevideov3-x2",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=False,
        )

    assert len(video_jobs.jobs) == 1, "the rejected job must not be registered as an orphan"


async def test_create_job_route_returns_429_when_queue_full(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_queue_size=1)
    storage = StorageService(settings)
    jobs = JobManager(settings, FakeEngine(), asyncio.Semaphore(settings.gpu_concurrency))

    await create_job(
        request=None,
        file=make_upload("a.png", make_png_bytes()),
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
        jobs=jobs,
        storage=storage,
        settings=settings,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_job(
            request=None,
            file=make_upload("b.png", make_png_bytes()),
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
            jobs=jobs,
            storage=storage,
            settings=settings,
        )

    assert exc_info.value.status_code == 429
    assert len(list(settings.uploads_path.iterdir())) == 1, "rejected upload must be cleaned up"


async def test_create_video_job_route_returns_429_when_queue_full(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, max_queue_size=1)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(
        settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(settings.gpu_concurrency)
    )

    await create_video_job(
        request=None,
        file=make_upload("a.mp4", b"fake-video-bytes-a"),
        profile_key="anime-balanced-2x",
        model_name=None,
        scale=None,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
        fps_multiplier=None,
        target_fps=None,
        audio_enhance=None,
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_video_job(
            request=None,
            file=make_upload("b.mp4", b"fake-video-bytes-b"),
            profile_key="anime-balanced-2x",
            model_name=None,
            scale=None,
            output_container=None,
            video_codec=None,
            video_preset=None,
            crf=None,
            keep_audio=None,
            fps_multiplier=None,
            target_fps=None,
            audio_enhance=None,
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
        )

    assert exc_info.value.status_code == 429
    assert len(list(settings.uploads_path.iterdir())) == 1, "rejected upload must be cleaned up"
