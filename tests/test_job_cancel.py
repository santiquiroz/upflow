from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.routes import cancel_audio_job, cancel_job, cancel_video_job
from app.config import Settings
from app.models import AudioJob, JobStatus, UpscaleJob, VideoUpscaleJob
from app.services.audio_job_manager import AudioJobManager
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.video_job_manager import VideoJobManager


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), PER_DEVICE_GPU_CONCURRENCY=1, MAX_CONCURRENT_JOBS=1)


async def wait_until(condition, timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)


# ---------------------------------------------------------------------------
# Fake engines: hang (cancellably) for a chosen set of job ids, complete others.
# ---------------------------------------------------------------------------
class SelectiveHangImageEngine(UpscaleEngine):
    def __init__(self, hang_ids: set[str]) -> None:
        self.hang_ids = hang_ids
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.completed: list[str] = []

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        self.started.append(job.id)
        if job.id in self.hang_ids:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(job.id)
                raise
        self.completed.append(job.id)
        return job.source_path


class SelectiveHangVideoUpscaler:
    def __init__(self, hang_ids: set[str]) -> None:
        self.hang_ids = hang_ids
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.completed: list[str] = []

    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        self.started.append(job.id)
        if job.id in self.hang_ids:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(job.id)
                raise
        self.completed.append(job.id)
        return job.source_path


class SelectiveHangAudioPipeline:
    def __init__(self, hang_ids: set[str]) -> None:
        self.hang_ids = hang_ids
        self.started: list[str] = []
        self.cancelled: list[str] = []
        self.completed: list[str] = []

    async def run(self, job: AudioJob) -> Path:
        self.started.append(job.id)
        if job.id in self.hang_ids:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.append(job.id)
                raise
        self.completed.append(job.id)
        return job.source_path


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video"}]}


# ---------------------------------------------------------------------------
# Job factories.
# ---------------------------------------------------------------------------
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


def make_audio_job(source_path: Path) -> AudioJob:
    return AudioJob(source_path=source_path, original_filename=source_path.name, denoise="deepfilternet")


def _register(manager, job) -> None:
    manager.jobs[job.id] = job


def _write_source(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake-bytes")
    return path


# ===========================================================================
# cancel_job() unit semantics (shared across the three managers).
# ===========================================================================
def _image_manager(settings: Settings, engine: UpscaleEngine) -> JobManager:
    return JobManager(settings, engine, DeviceSemaphores(settings))


def _video_manager(settings: Settings, upscaler) -> VideoJobManager:
    return VideoJobManager(settings, upscaler, FakeMediaTools(), DeviceSemaphores(settings))


def _audio_manager(settings: Settings, pipeline) -> AudioJobManager:
    return AudioJobManager(settings, pipeline, DeviceSemaphores(settings))


def test_cancel_job_returns_false_for_missing_id(tmp_path: Path) -> None:
    manager = _image_manager(make_settings(tmp_path), SelectiveHangImageEngine(set()))
    assert manager.cancel_job("nope") is False


@pytest.mark.parametrize("terminal", [JobStatus.completed, JobStatus.failed, JobStatus.cancelled])
def test_cancel_job_returns_false_for_finished_job(tmp_path: Path, terminal: JobStatus) -> None:
    manager = _image_manager(make_settings(tmp_path), SelectiveHangImageEngine(set()))
    job = make_image_job(tmp_path / "x.png")
    job.status = terminal
    _register(manager, job)
    assert manager.cancel_job(job.id) is False
    assert job.status == terminal


# ===========================================================================
# QUEUED cancel: engine never runs, source unlinked, task_done balanced.
# ===========================================================================
async def test_image_cancel_queued_job_skips_processing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = SelectiveHangImageEngine(set())
    manager = _image_manager(settings, engine)
    source = _write_source(tmp_path, "in.png")
    job = make_image_job(source)
    _register(manager, job)
    await manager.queue.put(job)

    assert manager.cancel_job(job.id) is True
    assert job.status == JobStatus.cancelled
    assert job.finished_at is not None

    await manager.start()
    await asyncio.wait_for(manager.queue.join(), timeout=2.0)

    assert engine.started == []
    assert not source.exists()
    await manager.stop()


async def test_video_cancel_queued_job_skips_processing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    upscaler = SelectiveHangVideoUpscaler(set())
    manager = _video_manager(settings, upscaler)
    source = _write_source(tmp_path, "in.mp4")
    job = make_video_job(source)
    _register(manager, job)
    await manager.queue.put(job)

    assert manager.cancel_job(job.id) is True
    assert job.status == JobStatus.cancelled

    await manager.start()
    await asyncio.wait_for(manager.queue.join(), timeout=2.0)

    assert upscaler.started == []
    assert not source.exists()
    await manager.stop()


async def test_audio_cancel_queued_job_skips_processing(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = SelectiveHangAudioPipeline(set())
    manager = _audio_manager(settings, pipeline)
    source = _write_source(tmp_path, "in.wav")
    job = make_audio_job(source)
    _register(manager, job)
    await manager.queue.put(job)

    assert manager.cancel_job(job.id) is True
    assert job.status == JobStatus.cancelled

    await manager.start()
    await asyncio.wait_for(manager.queue.join(), timeout=2.0)

    assert pipeline.started == []
    assert not source.exists()
    await manager.stop()


# ===========================================================================
# RUNNING cancel: worker SURVIVES and processes the next queued job.
# ===========================================================================
async def test_image_cancel_running_job_keeps_worker_alive(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    source1 = _write_source(tmp_path, "hang.png")
    source2 = _write_source(tmp_path, "next.png")
    job1 = make_image_job(source1)
    job2 = make_image_job(source2)
    engine = SelectiveHangImageEngine({job1.id})
    manager = _image_manager(settings, engine)
    _register(manager, job1)
    _register(manager, job2)
    await manager.queue.put(job1)
    await manager.queue.put(job2)

    await manager.start()
    worker_task = manager.worker_tasks[0]

    await wait_until(lambda: job1.id in engine.started)
    assert job1.status == JobStatus.running

    assert manager.cancel_job(job1.id) is True

    await wait_until(lambda: job1.status == JobStatus.cancelled)
    await wait_until(lambda: job2.status == JobStatus.completed)

    assert job1.id in engine.cancelled, "engine never received CancelledError"
    assert job1.error is None
    assert job1.finished_at is not None
    assert not source1.exists()
    assert not worker_task.done(), "worker died after a per-job cancel"
    assert engine.completed == [job2.id]

    await asyncio.wait_for(manager.queue.join(), timeout=2.0)
    await manager.stop()


async def test_video_cancel_running_job_keeps_worker_alive(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    source1 = _write_source(tmp_path, "hang.mp4")
    source2 = _write_source(tmp_path, "next.mp4")
    job1 = make_video_job(source1)
    job2 = make_video_job(source2)
    upscaler = SelectiveHangVideoUpscaler({job1.id})
    manager = _video_manager(settings, upscaler)
    _register(manager, job1)
    _register(manager, job2)
    await manager.queue.put(job1)
    await manager.queue.put(job2)

    await manager.start()
    worker_task = manager.worker_tasks[0]

    await wait_until(lambda: job1.id in upscaler.started)
    assert job1.status == JobStatus.running

    assert manager.cancel_job(job1.id) is True

    await wait_until(lambda: job1.status == JobStatus.cancelled)
    await wait_until(lambda: job2.status == JobStatus.completed)

    assert job1.id in upscaler.cancelled
    assert not source1.exists()
    assert not worker_task.done(), "worker died after a per-job cancel"
    assert upscaler.completed == [job2.id]

    await asyncio.wait_for(manager.queue.join(), timeout=2.0)
    await manager.stop()


async def test_audio_cancel_running_job_keeps_worker_alive(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    source1 = _write_source(tmp_path, "hang.wav")
    source2 = _write_source(tmp_path, "next.wav")
    job1 = make_audio_job(source1)
    job2 = make_audio_job(source2)
    pipeline = SelectiveHangAudioPipeline({job1.id})
    manager = _audio_manager(settings, pipeline)
    _register(manager, job1)
    _register(manager, job2)
    await manager.queue.put(job1)
    await manager.queue.put(job2)

    await manager.start()
    worker_task = manager.worker_tasks[0]

    await wait_until(lambda: job1.id in pipeline.started)
    assert job1.status == JobStatus.running

    assert manager.cancel_job(job1.id) is True

    await wait_until(lambda: job1.status == JobStatus.cancelled)
    await wait_until(lambda: job2.status == JobStatus.completed)

    assert job1.id in pipeline.cancelled
    assert not source1.exists()
    assert not worker_task.done(), "worker died after a per-job cancel"
    assert pipeline.completed == [job2.id]

    await asyncio.wait_for(manager.queue.join(), timeout=2.0)
    await manager.stop()


# ===========================================================================
# Endpoints: 200 (updated response), 404 (missing), 409 (already finished).
# ===========================================================================
async def test_cancel_job_endpoint_returns_updated_response(tmp_path: Path) -> None:
    manager = _image_manager(make_settings(tmp_path), SelectiveHangImageEngine(set()))
    job = make_image_job(tmp_path / "e.png")
    _register(manager, job)
    await manager.queue.put(job)

    response = await cancel_job(job_id=job.id, jobs=manager)

    assert response.job_id == job.id
    assert response.status == JobStatus.cancelled


async def test_cancel_job_endpoint_404_for_missing(tmp_path: Path) -> None:
    manager = _image_manager(make_settings(tmp_path), SelectiveHangImageEngine(set()))
    with pytest.raises(HTTPException) as exc_info:
        await cancel_job(job_id="missing", jobs=manager)
    assert exc_info.value.status_code == 404


async def test_cancel_job_endpoint_409_for_finished(tmp_path: Path) -> None:
    manager = _image_manager(make_settings(tmp_path), SelectiveHangImageEngine(set()))
    job = make_image_job(tmp_path / "done.png")
    job.status = JobStatus.completed
    _register(manager, job)
    with pytest.raises(HTTPException) as exc_info:
        await cancel_job(job_id=job.id, jobs=manager)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == "Job already finished"


async def test_cancel_video_job_endpoint_paths(tmp_path: Path) -> None:
    manager = _video_manager(make_settings(tmp_path), SelectiveHangVideoUpscaler(set()))
    job = make_video_job(tmp_path / "e.mp4")
    _register(manager, job)
    await manager.queue.put(job)

    response = await cancel_video_job(job_id=job.id, video_jobs=manager)
    assert response.status == JobStatus.cancelled

    with pytest.raises(HTTPException) as missing:
        await cancel_video_job(job_id="missing", video_jobs=manager)
    assert missing.value.status_code == 404

    finished = make_video_job(tmp_path / "done.mp4")
    finished.status = JobStatus.failed
    _register(manager, finished)
    with pytest.raises(HTTPException) as already:
        await cancel_video_job(job_id=finished.id, video_jobs=manager)
    assert already.value.status_code == 409


async def test_cancel_audio_job_endpoint_paths(tmp_path: Path) -> None:
    manager = _audio_manager(make_settings(tmp_path), SelectiveHangAudioPipeline(set()))
    job = make_audio_job(tmp_path / "e.wav")
    _register(manager, job)
    await manager.queue.put(job)

    response = await cancel_audio_job(job_id=job.id, audio_jobs=manager)
    assert response.status == JobStatus.cancelled

    with pytest.raises(HTTPException) as missing:
        await cancel_audio_job(job_id="missing", audio_jobs=manager)
    assert missing.value.status_code == 404

    finished = make_audio_job(tmp_path / "done.wav")
    finished.status = JobStatus.completed
    _register(manager, finished)
    with pytest.raises(HTTPException) as already:
        await cancel_audio_job(job_id=finished.id, audio_jobs=manager)
    assert already.value.status_code == 409
