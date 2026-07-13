from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from app.models import JobStatus, UpscaleJob, VideoUpscaleJob
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.video_job_manager import VideoJobManager


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), GPU_CONCURRENCY=1)


class FakeSubprocess:
    """Stands in for the child process the guarded runner would kill on cancel."""

    def __init__(self) -> None:
        self.kill_called = False

    def kill(self) -> None:
        self.kill_called = True


class HangingImageEngine(UpscaleEngine):
    """Mimics run_guarded_process's contract: kill the subprocess and re-raise on cancel."""

    def __init__(self) -> None:
        self.spawned_processes: list[FakeSubprocess] = []

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        process = FakeSubprocess()
        self.spawned_processes.append(process)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            process.kill()
            raise
        return job.source_path


class HangingVideoUpscaler:
    def __init__(self) -> None:
        self.spawned_processes: list[FakeSubprocess] = []

    async def run(self, job: VideoUpscaleJob) -> Path:
        process = FakeSubprocess()
        self.spawned_processes.append(process)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            process.kill()
            raise
        return job.source_path


class FakeMediaTools:
    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video"}]}


async def wait_until(condition, timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)


def make_image_job(source_path: Path) -> UpscaleJob:
    return UpscaleJob(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
    )


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


async def test_job_manager_worker_cancel_kills_subprocess_and_marks_job_failed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = HangingImageEngine()
    semaphore = asyncio.Semaphore(1)
    manager = JobManager(settings, engine, semaphore)

    source_path = tmp_path / "input.png"
    source_path.write_bytes(b"fake-image-bytes")
    job = make_image_job(source_path)
    manager.jobs[job.id] = job
    await manager.queue.put(job)

    await manager.start()
    worker_task = manager.worker_tasks[0]

    await wait_until(lambda: engine.spawned_processes)
    assert job.status == JobStatus.running

    worker_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker_task

    assert worker_task.cancelled(), "worker task did not actually end cancelled"
    assert engine.spawned_processes[0].kill_called, "subprocess was never killed on cancel"
    assert job.status == JobStatus.failed, "job was left in an impossible running-but-finished state"
    assert job.error
    assert job.finished_at is not None


async def test_video_job_manager_worker_cancel_kills_subprocess_and_marks_job_failed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    upscaler = HangingVideoUpscaler()
    semaphore = asyncio.Semaphore(1)
    manager = VideoJobManager(settings, upscaler, FakeMediaTools(), semaphore)

    source_path = tmp_path / "input.mp4"
    source_path.write_bytes(b"fake-video-bytes")
    job = make_video_job(source_path)
    manager.jobs[job.id] = job
    await manager.queue.put(job)

    await manager.start()
    worker_task = manager.worker_tasks[0]

    await wait_until(lambda: upscaler.spawned_processes)
    assert job.status == JobStatus.running

    worker_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker_task

    assert worker_task.cancelled(), "worker task did not actually end cancelled"
    assert upscaler.spawned_processes[0].kill_called, "subprocess was never killed on cancel"
    assert job.status == JobStatus.failed, "job was left in an impossible running-but-finished state"
    assert job.error
    assert job.finished_at is not None
