from __future__ import annotations

import asyncio
import io
from pathlib import Path

from PIL import Image
from starlette.datastructures import UploadFile

from app.api.routes import create_job, create_video_job
from app.config import Settings
from app.models import UpscaleJob, VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager


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


class FakeDevicesService:
    def list_devices(self) -> list[dict]:
        return [{"id": "dml:0", "kind": "gpu", "name": "Fake GPU", "backend": "directml"}]

    def resolve_default(self, devices: list[dict]) -> dict:
        return devices[0]


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


def make_png_bytes(color: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


async def test_concurrent_image_uploads_with_same_name_get_distinct_paths(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    jobs = JobManager(settings, FakeEngine(), DeviceSemaphores(settings))

    content_a = make_png_bytes("red")
    content_b = make_png_bytes("blue")

    response_a, response_b = await asyncio.gather(
        create_job(
            request=None,
            file=make_upload("photo.png", content_a),
            model_name="realesrgan-x4plus",
            model_id=None,
            device=None,
            scale=4,
            output_format="png",
            jobs=jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        ),
        create_job(
            request=None,
            file=make_upload("photo.png", content_b),
            model_name="realesrgan-x4plus",
            model_id=None,
            device=None,
            scale=4,
            output_format="png",
            jobs=jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        ),
    )

    job_a = jobs.get_job(response_a.job_id)
    job_b = jobs.get_job(response_b.job_id)

    assert job_a is not None
    assert job_b is not None
    assert job_a.source_path != job_b.source_path
    assert job_a.source_path.read_bytes() == content_a
    assert job_b.source_path.read_bytes() == content_b


async def test_concurrent_video_uploads_with_same_name_get_distinct_paths(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(
        settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings)
    )

    content_a = b"fake-video-bytes-a"
    content_b = b"fake-video-bytes-b"

    response_a, response_b = await asyncio.gather(
        create_video_job(
            request=None,
            file=make_upload("clip.mp4", content_a),
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
            audio_restore=None,
            model_id=None,
            device=None,
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        ),
        create_video_job(
            request=None,
            file=make_upload("clip.mp4", content_b),
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
            audio_restore=None,
            model_id=None,
            device=None,
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        ),
    )

    job_a = video_jobs.get_job(response_a.job_id)
    job_b = video_jobs.get_job(response_b.job_id)

    assert job_a is not None
    assert job_b is not None
    assert job_a.source_path != job_b.source_path
    assert job_a.source_path.read_bytes() == content_a
    assert job_b.source_path.read_bytes() == content_b
