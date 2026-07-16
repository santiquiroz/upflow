from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.models import JobStatus, UpscaleJob, VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.video_job_manager import VideoJobManager


def make_png_bytes(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()

# ---------------------------------------------------------------------------
# A transient PermissionError while unlinking the source upload (Windows AV /
# indexer holding the file open) must never kill the worker task nor mask
# CancelledError. The worker must log and move on to the next queued job.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), PER_DEVICE_GPU_CONCURRENCY=1)


class FakeImageEngine(UpscaleEngine):
    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        return job.source_path


class FakeVideoUpscaler:
    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        return job.source_path


class FakeMediaTools:
    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video"}]}


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


def patch_unlink_to_fail_for(monkeypatch: pytest.MonkeyPatch, locked_name: str) -> None:
    """Makes Path.unlink raise PermissionError only for a file matching `locked_name`."""
    original_unlink = Path.unlink

    def flaky_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == locked_name:
            raise PermissionError(f"[WinError 32] The process cannot access the file: '{self}'")
        return original_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)


async def test_image_worker_survives_locked_source_unlink_and_processes_next_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeImageEngine(), DeviceSemaphores(settings))

    locked_source = tmp_path / "locked.png"
    locked_source.write_bytes(make_png_bytes())
    ok_source = tmp_path / "ok.png"
    ok_source.write_bytes(make_png_bytes())

    patch_unlink_to_fail_for(monkeypatch, "locked.png")

    await manager.start()
    try:
        with caplog.at_level(logging.ERROR):
            locked_job = await manager.create_job(
                source_path=locked_source,
                original_filename="locked.png",
                model_name="realesrgan-x4plus",
                scale=4,
                output_format="png",
            )
            ok_job = await manager.create_job(
                source_path=ok_source,
                original_filename="ok.png",
                model_name="realesrgan-x4plus",
                scale=4,
                output_format="png",
            )
            await manager.queue.join()
    finally:
        await manager.stop()

    assert manager.get_job(locked_job.id).status == JobStatus.completed
    assert manager.get_job(ok_job.id).status == JobStatus.completed
    assert locked_source.exists(), "locked file must remain since unlink was guarded, not retried"
    assert not ok_source.exists(), "unlocked file must still be deleted normally"
    assert any(record.levelno >= logging.ERROR for record in caplog.records), (
        "a PermissionError on unlink must be logged"
    )


async def test_video_worker_survives_locked_source_unlink_and_processes_next_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    settings = make_settings(tmp_path)
    manager = VideoJobManager(settings, FakeVideoUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))

    locked_source = tmp_path / "locked.mp4"
    locked_source.write_bytes(b"fake-video-bytes")
    ok_source = tmp_path / "ok.mp4"
    ok_source.write_bytes(b"fake-video-bytes")

    patch_unlink_to_fail_for(monkeypatch, "locked.mp4")

    await manager.start()
    try:
        with caplog.at_level(logging.ERROR):
            locked_job = await manager.create_job(
                source_path=locked_source,
                original_filename="locked.mp4",
                model_name="realesr-animevideov3-x2",
                scale=2,
                output_container="mp4",
                video_codec="libx264",
                video_preset="medium",
                crf=18,
                keep_audio=False,
            )
            ok_job = await manager.create_job(
                source_path=ok_source,
                original_filename="ok.mp4",
                model_name="realesr-animevideov3-x2",
                scale=2,
                output_container="mp4",
                video_codec="libx264",
                video_preset="medium",
                crf=18,
                keep_audio=False,
            )
            await manager.queue.join()
    finally:
        await manager.stop()

    assert manager.get_job(locked_job.id).status == JobStatus.completed
    assert manager.get_job(ok_job.id).status == JobStatus.completed
    assert locked_source.exists(), "locked file must remain since unlink was guarded, not retried"
    assert not ok_source.exists(), "unlocked file must still be deleted normally"
    assert any(record.levelno >= logging.ERROR for record in caplog.records), (
        "a PermissionError on unlink must be logged"
    )


async def test_worker_cancellation_is_not_masked_by_a_failing_source_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PermissionError raised inside the finally-block unlink must never
    replace a propagating CancelledError."""
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeImageEngine(), DeviceSemaphores(settings))

    source_path = tmp_path / "locked-during-cancel.png"
    source_path.write_bytes(b"fake-image-bytes")
    patch_unlink_to_fail_for(monkeypatch, "locked-during-cancel.png")

    class HangingEngine(UpscaleEngine):
        def available(self) -> bool:
            return True

        async def run(self, job: UpscaleJob) -> Path:
            await asyncio.Event().wait()
            return job.source_path

    manager.engine = HangingEngine()
    job = make_image_job(source_path)
    manager.jobs[job.id] = job
    await manager.queue.put(job)

    await manager.start()
    worker_task = manager.worker_tasks[0]

    for _ in range(200):
        if job.status == JobStatus.running:
            break
        await asyncio.sleep(0.01)

    worker_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await worker_task

    assert worker_task.cancelled(), "the failing unlink must not mask CancelledError"
    assert job.status == JobStatus.failed
    assert job.error == "Job cancelled"
