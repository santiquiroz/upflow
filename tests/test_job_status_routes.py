from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.routes import (
    create_video_job,
    download_job,
    download_video_job,
    engine_info,
    get_job,
    get_video_job,
)
from app.config import Settings
from app.models import JobStatus, UpscaleJob, VideoUpscaleJob
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
    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        return job.source_path


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video"}]}


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


def make_job_manager(settings: Settings) -> JobManager:
    return JobManager(settings, FakeEngine(), asyncio.Semaphore(1))


def make_video_job_manager(settings: Settings) -> VideoJobManager:
    return VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))


def make_job(tmp_path: Path, **overrides: object) -> UpscaleJob:
    fields = dict(
        source_path=tmp_path / "source.png",
        original_filename="source.png",
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
    )
    fields.update(overrides)
    return UpscaleJob(**fields)


def make_video_job(tmp_path: Path, **overrides: object) -> VideoUpscaleJob:
    fields = dict(
        source_path=tmp_path / "source.mp4",
        original_filename="source.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )
    fields.update(overrides)
    return VideoUpscaleJob(**fields)


async def test_engine_info_returns_engine_and_catalog_details(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    request = types.SimpleNamespace(
        app=types.SimpleNamespace(
            state=types.SimpleNamespace(engine=FakeEngine(), media_tools=FakeMediaTools())
        )
    )

    response = await engine_info(request=request, settings=settings)

    assert response.engine == settings.engine
    assert response.available is True
    assert len(response.supported_models) == len(settings.model_catalog)
    assert len(response.video_profiles) == len(settings.video_profile_catalog)
    assert response.ffmpeg_available is True


async def test_get_job_returns_404_when_job_is_unknown(tmp_path: Path) -> None:
    jobs = make_job_manager(make_settings(tmp_path))

    with pytest.raises(HTTPException) as exc_info:
        await get_job(job_id="missing", jobs=jobs)

    assert exc_info.value.status_code == 404


async def test_get_job_returns_response_and_download_url_tracks_status(tmp_path: Path) -> None:
    jobs = make_job_manager(make_settings(tmp_path))
    job = make_job(tmp_path)
    jobs.jobs[job.id] = job

    queued_response = await get_job(job_id=job.id, jobs=jobs)
    job.status = JobStatus.completed
    completed_response = await get_job(job_id=job.id, jobs=jobs)

    assert queued_response.job_id == job.id
    assert queued_response.status == JobStatus.queued
    assert queued_response.original_filename == "source.png"
    assert queued_response.download_url is None
    assert completed_response.download_url == f"/api/v1/jobs/{job.id}/download"


async def test_get_video_job_returns_404_when_job_is_unknown(tmp_path: Path) -> None:
    video_jobs = make_video_job_manager(make_settings(tmp_path))

    with pytest.raises(HTTPException) as exc_info:
        await get_video_job(job_id="missing", video_jobs=video_jobs)

    assert exc_info.value.status_code == 404


async def test_get_video_job_returns_response_and_download_url_tracks_status(tmp_path: Path) -> None:
    video_jobs = make_video_job_manager(make_settings(tmp_path))
    job = make_video_job(tmp_path)
    video_jobs.jobs[job.id] = job

    queued_response = await get_video_job(job_id=job.id, video_jobs=video_jobs)
    job.status = JobStatus.completed
    completed_response = await get_video_job(job_id=job.id, video_jobs=video_jobs)

    assert queued_response.job_id == job.id
    assert queued_response.status == JobStatus.queued
    assert queued_response.original_filename == "source.mp4"
    assert queued_response.download_url is None
    assert completed_response.download_url == f"/api/v1/video/jobs/{job.id}/download"


async def test_create_video_job_returns_400_when_profile_is_unknown(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    video_jobs = make_video_job_manager(settings)
    storage = StorageService(settings)

    with pytest.raises(HTTPException) as exc_info:
        await create_video_job(
            request=None,
            file=None,
            profile_key="missing-profile",
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

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Unknown profile: missing-profile"


async def test_download_job_returns_404_when_job_is_unknown(tmp_path: Path) -> None:
    jobs = make_job_manager(make_settings(tmp_path))

    with pytest.raises(HTTPException) as exc_info:
        await download_job(job_id="missing", jobs=jobs)

    assert exc_info.value.status_code == 404


async def test_download_job_returns_409_when_job_is_not_completed(tmp_path: Path) -> None:
    jobs = make_job_manager(make_settings(tmp_path))
    job = make_job(tmp_path)
    jobs.jobs[job.id] = job

    with pytest.raises(HTTPException) as exc_info:
        await download_job(job_id=job.id, jobs=jobs)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Job is not completed yet"


async def test_download_job_returns_file_response_for_completed_job(tmp_path: Path) -> None:
    jobs = make_job_manager(make_settings(tmp_path))
    output_path = tmp_path / "out.png"
    output_path.write_bytes(b"png")
    job = make_job(tmp_path, status=JobStatus.completed, output_path=output_path)
    jobs.jobs[job.id] = job

    response = await download_job(job_id=job.id, jobs=jobs)

    assert str(response.path) == str(output_path)
    assert response.media_type == "application/octet-stream"


async def test_download_video_job_returns_404_when_job_is_unknown(tmp_path: Path) -> None:
    video_jobs = make_video_job_manager(make_settings(tmp_path))

    with pytest.raises(HTTPException) as exc_info:
        await download_video_job(job_id="missing", video_jobs=video_jobs)

    assert exc_info.value.status_code == 404


async def test_download_video_job_returns_409_when_job_is_not_completed(tmp_path: Path) -> None:
    video_jobs = make_video_job_manager(make_settings(tmp_path))
    job = make_video_job(tmp_path)
    video_jobs.jobs[job.id] = job

    with pytest.raises(HTTPException) as exc_info:
        await download_video_job(job_id=job.id, video_jobs=video_jobs)

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Video job is not completed yet"


async def test_download_video_job_returns_file_response_for_completed_job(tmp_path: Path) -> None:
    video_jobs = make_video_job_manager(make_settings(tmp_path))
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"mp4")
    job = make_video_job(tmp_path, status=JobStatus.completed, output_path=output_path)
    video_jobs.jobs[job.id] = job

    response = await download_video_job(job_id=job.id, video_jobs=video_jobs)

    assert str(response.path) == str(output_path)
    assert response.media_type == "application/octet-stream"
