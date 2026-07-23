from __future__ import annotations
import threading

import asyncio
import io
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import app
from app.models import GenerationJob, JobStatus, UpscaleJob, VideoUpscaleJob, utc_now
from app.services import retention_sweeper as retention_sweeper_module
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.generation_job_manager import GenerationJobManager
from app.services.job_manager import JobManager
from app.services.model_registry import ModelRegistry
from app.services.retention_sweeper import RetentionSweeper
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager
from app.services.video_upscaler import VideoUpscaler


def make_settings(tmp_path: Path, output_ttl_hours: int = 24) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), OUTPUT_TTL_HOURS=output_ttl_hours)


def make_png_bytes(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeVideoEngine:
    def available(self) -> bool:
        return True


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [{"codec_type": "video", "width": 4, "height": 4, "avg_frame_rate": "30/1"}],
            "format": {"duration": "1.0"},
        }


class FakePipelineVideoUpscaler(VideoUpscaler):
    """Fakes _run_process so no real ffmpeg/engine binary is invoked; drops dummy frame/output files instead."""

    async def _run_process(self, command: list[str]) -> None:
        if "-fps_mode" in command:
            self._write_dummy_frame(command)
        elif "-vn" in command:
            self._write_dummy_audio(command)
        elif command[0] == str(self.settings.engine_binary_path):
            self._write_dummy_upscaled_frame(command)
        elif "-framerate" in command:
            self._write_dummy_output(command)

    @staticmethod
    def _write_dummy_frame(command: list[str]) -> None:
        frames_in_dir = Path(command[-1]).parent
        frames_in_dir.mkdir(parents=True, exist_ok=True)
        (frames_in_dir / "00000001.png").write_bytes(b"fake-frame-in")

    @staticmethod
    def _write_dummy_audio(command: list[str]) -> None:
        audio_path = Path(command[-1])
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake-audio")

    @staticmethod
    def _write_dummy_upscaled_frame(command: list[str]) -> None:
        frames_out_dir = Path(command[command.index("-o") + 1])
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        (frames_out_dir / "00000001.png").write_bytes(b"fake-frame-out")

    @staticmethod
    def _write_dummy_output(command: list[str]) -> None:
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-output-video")


class FailingPipelineVideoUpscaler(VideoUpscaler):
    """Simulates an engine crash mid-pipeline, after work_dir already has content on disk."""

    async def _run_process(self, command: list[str]) -> None:
        if "-fps_mode" in command:
            frames_in_dir = Path(command[-1]).parent
            frames_in_dir.mkdir(parents=True, exist_ok=True)
            (frames_in_dir / "00000001.png").write_bytes(b"fake-frame-in")
            return
        raise RuntimeError("simulated engine crash")


def make_video_job(source_path: Path) -> VideoUpscaleJob:
    return VideoUpscaleJob(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )


class FakeImageEngine(UpscaleEngine):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        output_path = self.settings.outputs_path / f"{job.id}.{job.output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-output-image")
        return output_path


class FailingImageEngine(UpscaleEngine):
    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        raise RuntimeError("simulated engine crash")


class FailingVideoUpscaler:
    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        raise RuntimeError("simulated engine crash")


class FakeGenerationEngine:
    async def run(self, **kwargs: object) -> Path:
        raise RuntimeError("not invoked in retention-sweeper tests")


def make_generation_job_manager(settings: Settings) -> GenerationJobManager:
    return GenerationJobManager(
        settings,
        FakeGenerationEngine(),
        DeviceSemaphores(settings),
        registry=ModelRegistry(settings),
        upscale_engine=FakeImageEngine(settings),
    )


async def test_video_upscaler_removes_work_dir_after_success(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    StorageService(settings)
    upscaler = FakePipelineVideoUpscaler(settings, FakeVideoEngine(), FakeMediaTools())

    source_path = settings.uploads_path / "clip.mp4"
    source_path.write_bytes(b"fake-video-bytes")
    job = make_video_job(source_path)

    output_path = await upscaler.run(job)

    work_dir = settings.video_work_path / job.id
    assert not work_dir.exists()
    assert output_path.exists()


async def test_video_upscaler_removes_work_dir_after_failure(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    StorageService(settings)
    upscaler = FailingPipelineVideoUpscaler(settings, FakeVideoEngine(), FakeMediaTools())

    source_path = settings.uploads_path / "clip.mp4"
    source_path.write_bytes(b"fake-video-bytes")
    job = make_video_job(source_path)

    with pytest.raises(RuntimeError, match="simulated engine crash"):
        await upscaler.run(job)

    work_dir = settings.video_work_path / job.id
    assert not work_dir.exists()


async def test_video_job_worker_removes_source_and_keeps_output_on_success(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    StorageService(settings)
    upscaler = FakePipelineVideoUpscaler(settings, FakeVideoEngine(), FakeMediaTools())
    video_jobs = VideoJobManager(
        settings, upscaler, FakeMediaTools(), DeviceSemaphores(settings)
    )

    source_path = settings.uploads_path / "clip.mp4"
    source_path.write_bytes(b"fake-video-bytes")

    await video_jobs.start()
    try:
        job = await video_jobs.create_job(
            source_path=source_path,
            original_filename="clip.mp4",
            model_name="realesr-animevideov3-x2",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=False,
        )
        await video_jobs.queue.join()
    finally:
        await video_jobs.stop()

    finished_job = video_jobs.get_job(job.id)
    assert finished_job is not None
    assert finished_job.status == JobStatus.completed
    assert not source_path.exists()
    assert not (settings.video_work_path / job.id).exists()
    assert finished_job.output_path is not None
    assert finished_job.output_path.exists()


async def test_video_job_worker_removes_source_on_failure(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    StorageService(settings)
    video_jobs = VideoJobManager(
        settings,
        FailingVideoUpscaler(),
        FakeMediaTools(),
        DeviceSemaphores(settings),
    )

    source_path = settings.uploads_path / "clip.mp4"
    source_path.write_bytes(b"fake-video-bytes")

    await video_jobs.start()
    try:
        job = await video_jobs.create_job(
            source_path=source_path,
            original_filename="clip.mp4",
            model_name="realesr-animevideov3-x2",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=False,
        )
        await video_jobs.queue.join()
    finally:
        await video_jobs.stop()

    finished_job = video_jobs.get_job(job.id)
    assert finished_job is not None
    assert finished_job.status == JobStatus.failed
    assert not source_path.exists()


async def test_image_job_worker_removes_source_and_keeps_output_on_success(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    StorageService(settings)
    jobs = JobManager(
        settings, FakeImageEngine(settings), DeviceSemaphores(settings)
    )

    source_path = settings.uploads_path / "photo.png"
    source_path.write_bytes(make_png_bytes())

    await jobs.start()
    try:
        job = await jobs.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
        )
        await jobs.queue.join()
    finally:
        await jobs.stop()

    finished_job = jobs.get_job(job.id)
    assert finished_job is not None
    assert finished_job.status == JobStatus.completed
    assert not source_path.exists()
    assert finished_job.output_path is not None
    assert finished_job.output_path.exists()


async def test_image_job_worker_removes_source_on_failure(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    StorageService(settings)
    jobs = JobManager(
        settings, FailingImageEngine(), DeviceSemaphores(settings)
    )

    source_path = settings.uploads_path / "photo.png"
    source_path.write_bytes(make_png_bytes())

    await jobs.start()
    try:
        job = await jobs.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
        )
        await jobs.queue.join()
    finally:
        await jobs.stop()

    finished_job = jobs.get_job(job.id)
    assert finished_job is not None
    assert finished_job.status == JobStatus.failed
    assert not source_path.exists()


def test_retention_sweeper_deletes_expired_outputs_and_keeps_fresh_ones(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    job_manager = JobManager(
        settings, FakeImageEngine(settings), DeviceSemaphores(settings)
    )
    video_job_manager = VideoJobManager(
        settings,
        FailingVideoUpscaler(),
        FakeMediaTools(),
        DeviceSemaphores(settings),
    )
    sweeper = RetentionSweeper(settings, job_manager, video_job_manager)

    stale_output = settings.outputs_path / "stale.png"
    stale_output.write_bytes(b"stale")
    fresh_output = settings.outputs_path / "fresh.png"
    fresh_output.write_bytes(b"fresh")

    stale_mtime = time.time() - 2 * 3600
    os.utime(stale_output, (stale_mtime, stale_mtime))

    sweeper.sweep_once()

    assert not stale_output.exists()
    assert fresh_output.exists()


def test_retention_sweeper_prunes_old_finished_jobs_but_keeps_recent_and_running(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    job_manager = JobManager(
        settings, FakeImageEngine(settings), DeviceSemaphores(settings)
    )
    video_job_manager = VideoJobManager(
        settings,
        FailingVideoUpscaler(),
        FakeMediaTools(),
        DeviceSemaphores(settings),
    )
    sweeper = RetentionSweeper(settings, job_manager, video_job_manager)

    def make_job(status: JobStatus, finished_at) -> UpscaleJob:
        job = UpscaleJob(
            source_path=tmp_path / "unused.png",
            original_filename="unused.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
        )
        job.status = status
        job.finished_at = finished_at
        return job

    old_job = make_job(JobStatus.completed, utc_now() - timedelta(hours=2))
    recent_job = make_job(JobStatus.completed, utc_now())
    running_job = make_job(JobStatus.running, None)

    job_manager.jobs[old_job.id] = old_job
    job_manager.jobs[recent_job.id] = recent_job
    job_manager.jobs[running_job.id] = running_job

    sweeper.sweep_once()

    assert old_job.id not in job_manager.jobs
    assert recent_job.id in job_manager.jobs
    assert running_job.id in job_manager.jobs


def test_retention_sweeper_prunes_old_finished_generation_jobs_but_keeps_recent_and_running(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    job_manager = JobManager(
        settings, FakeImageEngine(settings), DeviceSemaphores(settings)
    )
    video_job_manager = VideoJobManager(
        settings,
        FailingVideoUpscaler(),
        FakeMediaTools(),
        DeviceSemaphores(settings),
    )
    generation_job_manager = make_generation_job_manager(settings)
    sweeper = RetentionSweeper(
        settings, job_manager, video_job_manager, generation_job_manager=generation_job_manager
    )

    def make_generation_job(status: JobStatus, finished_at) -> GenerationJob:
        job = GenerationJob(prompt="a red apple", model_id="gen--amd--sd15")
        job.status = status
        job.finished_at = finished_at
        return job

    old_job = make_generation_job(JobStatus.completed, utc_now() - timedelta(hours=2))
    recent_job = make_generation_job(JobStatus.completed, utc_now())
    running_job = make_generation_job(JobStatus.running, None)

    generation_job_manager.jobs[old_job.id] = old_job
    generation_job_manager.jobs[recent_job.id] = recent_job
    generation_job_manager.jobs[running_job.id] = running_job

    sweeper.sweep_once()

    assert old_job.id not in generation_job_manager.jobs
    assert recent_job.id in generation_job_manager.jobs
    assert running_job.id in generation_job_manager.jobs


def test_lifespan_starts_and_stops_retention_sweeper() -> None:
    with TestClient(app):
        assert app.state.retention_sweeper is not None
        assert app.state.retention_sweeper.sweep_task is not None
    assert app.state.retention_sweeper.sweep_task is None


def make_sweeper(settings: Settings) -> RetentionSweeper:
    job_manager = JobManager(
        settings, FakeImageEngine(settings), DeviceSemaphores(settings)
    )
    video_job_manager = VideoJobManager(
        settings,
        FailingVideoUpscaler(),
        FakeMediaTools(),
        DeviceSemaphores(settings),
    )
    return RetentionSweeper(settings, job_manager, video_job_manager)


async def test_sweeper_runs_first_sweep_immediately_on_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retention_sweeper_module, "SWEEP_INTERVAL_SECONDS", 3600)
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    sweeper = make_sweeper(settings)

    stale_output = settings.outputs_path / "stale.png"
    stale_output.write_bytes(b"stale")
    fresh_output = settings.outputs_path / "fresh.png"
    fresh_output.write_bytes(b"fresh")
    stale_mtime = time.time() - 2 * 3600
    os.utime(stale_output, (stale_mtime, stale_mtime))

    await sweeper.start()
    # The boot sweep now runs via asyncio.to_thread, so it completes a few loop
    # cycles after start() rather than synchronously; poll for the deletion.
    for _ in range(200):
        if not stale_output.exists():
            break
        await asyncio.sleep(0.01)
    await sweeper.stop()

    assert not stale_output.exists()
    assert fresh_output.exists()


async def test_sweeper_loop_survives_a_failing_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retention_sweeper_module, "SWEEP_INTERVAL_SECONDS", 0.01)
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    sweeper = make_sweeper(settings)

    sweep_calls: list[int] = []

    def flaky_sweep_once() -> None:
        sweep_calls.append(len(sweep_calls))
        if len(sweep_calls) == 1:
            raise OSError("simulated locked file on Windows")

    monkeypatch.setattr(sweeper, "sweep_once", flaky_sweep_once)

    await sweeper.start()
    for _ in range(200):
        if len(sweep_calls) >= 2:
            break
        await asyncio.sleep(0.01)
    await sweeper.stop()

    assert len(sweep_calls) >= 2, "the sweep loop died after the first sweep_once raised"


async def test_sweeper_stop_waits_for_in_flight_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # stop() must not return while a sweep thread is still running (a detached
    # thread would keep deleting files after the app thinks it stopped).
    monkeypatch.setattr(retention_sweeper_module, "SWEEP_INTERVAL_SECONDS", 3600)
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    sweeper = make_sweeper(settings)

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_sweep_once() -> None:
        entered.set()
        release.wait(timeout=5)  # hold the worker thread inside the sweep
        finished.set()

    monkeypatch.setattr(sweeper, "sweep_once", slow_sweep_once)

    await sweeper.start()
    # wait until the worker thread is actually inside the sweep
    for _ in range(200):
        if entered.is_set():
            break
        await asyncio.sleep(0.01)
    assert entered.is_set()

    # kick off stop() concurrently, then let the sweep finish; stop() must only
    # return AFTER the worker completed.
    stop_task = asyncio.create_task(sweeper.stop())
    await asyncio.sleep(0.05)
    assert not stop_task.done(), "stop() returned while the sweep thread was still running"
    release.set()
    await stop_task
    assert finished.is_set()


# ---------------------------------------------------------------------------
# Retention sweeper: also covers uploads/ and video-work/ (not just outputs/)
# ---------------------------------------------------------------------------


def test_retention_sweeper_deletes_expired_uploads_and_keeps_fresh_ones(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    sweeper = make_sweeper(settings)

    stale_upload = settings.uploads_path / "stale-upload.png"
    stale_upload.write_bytes(b"stale")
    fresh_upload = settings.uploads_path / "fresh-upload.png"
    fresh_upload.write_bytes(b"fresh")

    stale_mtime = time.time() - 2 * 3600
    os.utime(stale_upload, (stale_mtime, stale_mtime))

    sweeper.sweep_once()

    assert not stale_upload.exists()
    assert fresh_upload.exists()


def test_retention_sweeper_keeps_stale_upload_referenced_by_queued_image_job(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    sweeper = make_sweeper(settings)

    active_upload = settings.uploads_path / "active-image-upload.png"
    active_upload.write_bytes(b"active")
    stale_mtime = time.time() - 2 * 3600
    os.utime(active_upload, (stale_mtime, stale_mtime))

    queued_job = UpscaleJob(
        source_path=active_upload,
        original_filename="active-image-upload.png",
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
    )
    sweeper.job_manager.jobs[queued_job.id] = queued_job

    sweeper.sweep_once()

    assert active_upload.exists(), "upload referenced by a queued image job must survive the sweep"


def test_retention_sweeper_keeps_stale_upload_referenced_by_running_video_job(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    sweeper = make_sweeper(settings)

    active_upload = settings.uploads_path / "active-video-upload.mp4"
    active_upload.write_bytes(b"active")
    stale_mtime = time.time() - 2 * 3600
    os.utime(active_upload, (stale_mtime, stale_mtime))

    running_job = VideoUpscaleJob(
        source_path=active_upload,
        original_filename="active-video-upload.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )
    running_job.status = JobStatus.running
    sweeper.video_job_manager.jobs[running_job.id] = running_job

    sweeper.sweep_once()

    assert active_upload.exists(), "upload referenced by a running video job must survive the sweep"


def test_retention_sweeper_removes_expired_video_work_dirs_and_keeps_fresh_ones(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    sweeper = make_sweeper(settings)

    stale_work_dir = settings.video_work_path / "stale-job-id"
    (stale_work_dir / "frames-in").mkdir(parents=True)
    (stale_work_dir / "frames-in" / "00000001.png").write_bytes(b"frame")
    stale_mtime = time.time() - 2 * 3600
    os.utime(stale_work_dir, (stale_mtime, stale_mtime))

    fresh_work_dir = settings.video_work_path / "fresh-job-id"
    fresh_work_dir.mkdir(parents=True)

    sweeper.sweep_once()

    assert not stale_work_dir.exists()
    assert fresh_work_dir.exists()


def test_retention_sweeper_keeps_stale_work_dir_of_running_video_job(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, output_ttl_hours=1)
    StorageService(settings)
    sweeper = make_sweeper(settings)

    running_job = make_video_job(tmp_path / "source.mp4")
    running_job.status = JobStatus.running
    sweeper.video_job_manager.jobs[running_job.id] = running_job

    active_work_dir = settings.video_work_path / running_job.id
    active_work_dir.mkdir(parents=True)
    stale_mtime = time.time() - 2 * 3600
    os.utime(active_work_dir, (stale_mtime, stale_mtime))

    sweeper.sweep_once()

    assert active_work_dir.exists(), "work dir of a running video job must survive the sweep"
