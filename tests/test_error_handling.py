from __future__ import annotations

import asyncio
import io
import logging
import struct
import subprocess
import zlib
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import create_job, create_video_job
from app.config import Settings
from app.models import UpscaleJob, VideoUpscaleJob
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

# PIL raises DecompressionBombWarning above Image.MAX_IMAGE_PIXELS (89_478_485) and
# DecompressionBombError above 2x that (178_956_970). Dimensions below claim more than
# 2x pixels so the error branch fires directly with no warning ever emitted.
BOMB_WIDTH = 20000
BOMB_HEIGHT = 20000


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


def make_bomb_png_bytes(width: int = BOMB_WIDTH, height: int = BOMB_HEIGHT) -> bytes:
    """Builds a tiny PNG whose IHDR header claims huge dimensions.

    No real pixel data is needed: Pillow's decompression-bomb check runs against the
    header-declared size before any pixel decoding happens.
    """

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


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


class FakeMediaToolsRaisingCalledProcessError:
    async def ffprobe_json(self, source_path: Path) -> dict:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffprobe", str(source_path)],
            output="",
            stderr="Invalid data found when processing input",
        )


class FakeDevicesService:
    def list_devices(self) -> list[dict]:
        return [{"id": "dml:0", "kind": "gpu", "name": "Fake GPU", "backend": "directml"}]

    def resolve_default(self, devices: list[dict]) -> dict:
        return devices[0]


class ExplodingJobManager:
    async def create_job(self, **kwargs: object) -> UpscaleJob:
        raise RuntimeError("disk full at C:\\internal\\secret-path\\uploads")


class ExplodingVideoJobManager:
    async def create_job(self, **kwargs: object) -> VideoUpscaleJob:
        raise RuntimeError("disk full at C:\\internal\\secret-path\\uploads")


async def test_bomb_sized_image_upload_returns_400_and_no_leftover_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    jobs = JobManager(settings, FakeEngine(), asyncio.Semaphore(settings.gpu_concurrency))

    with pytest.raises(HTTPException) as exc_info:
        await create_job(
            request=None,
            file=make_upload("bomb.png", make_bomb_png_bytes()),
            model_name="realesrgan-x4plus",
            model_id=None,
            device=None,
            scale=4,
            output_format="png",
            jobs=jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        )

    assert exc_info.value.status_code == 400
    assert list(settings.uploads_path.iterdir()) == []


async def test_malformed_video_upload_returns_400_and_no_leftover_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(
        settings,
        FakeUpscaler(),
        FakeMediaToolsRaisingCalledProcessError(),
        asyncio.Semaphore(settings.gpu_concurrency),
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_video_job(
            request=None,
            file=make_upload("clip.mp4", b"not-really-a-video"),
            profile_key="anime-balanced-2x",
            model_name=None,
            scale=None,
            output_container=None,
            video_codec=None,
            video_preset=None,
            crf=None,
            keep_audio=None,
            model_id=None,
            device=None,
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        )

    assert exc_info.value.status_code == 400
    assert list(settings.uploads_path.iterdir()) == []


async def test_unexpected_image_job_error_returns_clean_500_and_no_leftover_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as exc_info:
            await create_job(
                request=None,
                file=make_upload("photo.png", b"irrelevant-bytes"),
                model_name="realesrgan-x4plus",
                model_id=None,
                device=None,
                scale=4,
                output_format="png",
                jobs=ExplodingJobManager(),
                storage=storage,
                settings=settings,
                devices=FakeDevicesService(),
            )

    assert exc_info.value.status_code == 500
    assert "secret-path" not in str(exc_info.value.detail)
    assert "disk full" not in str(exc_info.value.detail)
    assert list(settings.uploads_path.iterdir()) == []
    assert any(record.levelno >= logging.ERROR for record in caplog.records), (
        "the unexpected exception must be logged server-side even though the client sees a generic message"
    )


async def test_unexpected_video_job_error_returns_clean_500_and_no_leftover_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(HTTPException) as exc_info:
            await create_video_job(
                request=None,
                file=make_upload("clip.mp4", b"irrelevant-bytes"),
                profile_key="anime-balanced-2x",
                model_name=None,
                scale=None,
                output_container=None,
                video_codec=None,
                video_preset=None,
                crf=None,
                keep_audio=None,
                model_id=None,
                device=None,
                video_jobs=ExplodingVideoJobManager(),
                storage=storage,
                settings=settings,
                devices=FakeDevicesService(),
            )

    assert exc_info.value.status_code == 500
    assert "secret-path" not in str(exc_info.value.detail)
    assert "disk full" not in str(exc_info.value.detail)
    assert list(settings.uploads_path.iterdir()) == []
    assert any(record.levelno >= logging.ERROR for record in caplog.records), (
        "the unexpected exception must be logged server-side even though the client sees a generic message"
    )


async def test_validate_input_image_translates_decompression_bomb_error_to_value_error(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    jobs = JobManager(settings, FakeEngine(), asyncio.Semaphore(settings.gpu_concurrency))
    source_path = tmp_path / "bomb.png"
    source_path.write_bytes(make_bomb_png_bytes())

    with pytest.raises(ValueError):
        jobs._validate_input_image(source_path)


async def test_validate_video_translates_calledprocesserror_to_value_error(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    video_jobs = VideoJobManager(
        settings,
        FakeUpscaler(),
        FakeMediaToolsRaisingCalledProcessError(),
        asyncio.Semaphore(settings.gpu_concurrency),
    )
    source_path = tmp_path / "clip.mp4"
    source_path.write_bytes(b"not-really-a-video")

    with pytest.raises(ValueError):
        await video_jobs._validate_video(source_path)
