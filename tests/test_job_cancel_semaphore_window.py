"""Regresion: cancelar un job ya sacado de la cola pero ESPERANDO el permiso
del device no debe resucitarlo.

Escenario real (default de produccion: 4 workers, 1 permiso GPU): el worker
saca el job de la cola y se bloquea en device_semaphores.acquire() con el job
todavia en status=queued. Un cancel en esa ventana marcaba cancelled sin nada
que cancelar, y al liberarse el permiso _execute_job pisaba el status con
running → completed, quemando GPU en un job que el usuario creia muerto.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.models import JobStatus, UpscaleJob, VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.video_job_manager import VideoJobManager


def make_settings(tmp_path: Path) -> Settings:
    # 2 workers y 1 permiso: exactamente la configuracion que abre la ventana.
    return Settings(RUNTIME_DIR=str(tmp_path), PER_DEVICE_GPU_CONCURRENCY=1, MAX_CONCURRENT_JOBS=2)


async def wait_until(condition, timeout: float = 2.0) -> None:
    async def _poll() -> None:
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)


class HoldableImageEngine(UpscaleEngine):
    """Se cuelga (cancelable) en los ids indicados; completa el resto."""

    def __init__(self, hang_ids: set[str]) -> None:
        self.hang_ids = hang_ids
        self.started: list[str] = []

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        self.started.append(job.id)
        if job.id in self.hang_ids:
            await asyncio.Event().wait()
        return job.source_path


class HoldableVideoUpscaler:
    def __init__(self, hang_ids: set[str]) -> None:
        self.hang_ids = hang_ids
        self.started: list[str] = []

    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        self.started.append(job.id)
        if job.id in self.hang_ids:
            await asyncio.Event().wait()
        return job.source_path


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video"}]}


def _write_source(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_bytes(b"fake-bytes")
    return path


def _image_job(source: Path) -> UpscaleJob:
    return UpscaleJob(
        source_path=source,
        original_filename=source.name,
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
    )


def _video_job(source: Path) -> VideoUpscaleJob:
    return VideoUpscaleJob(
        source_path=source,
        original_filename=source.name,
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )


async def test_image_cancel_while_waiting_for_permit_does_not_resurrect(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    source_a = _write_source(tmp_path, "hang.png")
    source_b = _write_source(tmp_path, "victim.png")
    job_a = _image_job(source_a)
    job_b = _image_job(source_b)
    engine = HoldableImageEngine({job_a.id})
    manager = JobManager(settings, engine, DeviceSemaphores(settings))
    manager.jobs[job_a.id] = job_a
    manager.jobs[job_b.id] = job_b
    await manager.queue.put(job_a)
    await manager.queue.put(job_b)

    await manager.start()
    # A corre y retiene el unico permiso; B fue sacado de la cola por el
    # segundo worker y quedo bloqueado en acquire() con status=queued.
    await wait_until(lambda: job_a.id in engine.started)
    await wait_until(lambda: manager.queue.qsize() == 0)
    assert job_b.status == JobStatus.queued

    assert manager.cancel_job(job_b.id) is True
    assert job_b.status == JobStatus.cancelled

    # Se libera el permiso: A se cancela corriendo, el worker de B despierta.
    assert manager.cancel_job(job_a.id) is True
    await wait_until(lambda: job_a.status == JobStatus.cancelled)
    await asyncio.wait_for(manager.queue.join(), timeout=2.0)

    assert job_b.status == JobStatus.cancelled, "el job cancelado resucito"
    assert job_b.id not in engine.started, "el motor corrio un job cancelado"
    assert not source_b.exists(), "la fuente del job cancelado quedo huerfana"
    await manager.stop()


async def test_video_cancel_while_waiting_for_permit_does_not_resurrect(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    source_a = _write_source(tmp_path, "hang.mp4")
    source_b = _write_source(tmp_path, "victim.mp4")
    job_a = _video_job(source_a)
    job_b = _video_job(source_b)
    upscaler = HoldableVideoUpscaler({job_a.id})
    manager = VideoJobManager(settings, upscaler, FakeMediaTools(), DeviceSemaphores(settings))
    manager.jobs[job_a.id] = job_a
    manager.jobs[job_b.id] = job_b
    await manager.queue.put(job_a)
    await manager.queue.put(job_b)

    await manager.start()
    await wait_until(lambda: job_a.id in upscaler.started)
    await wait_until(lambda: manager.queue.qsize() == 0)
    assert job_b.status == JobStatus.queued

    assert manager.cancel_job(job_b.id) is True
    assert manager.cancel_job(job_a.id) is True
    await wait_until(lambda: job_a.status == JobStatus.cancelled)
    await asyncio.wait_for(manager.queue.join(), timeout=2.0)

    assert job_b.status == JobStatus.cancelled, "el job cancelado resucito"
    assert job_b.id not in upscaler.started
    assert not source_b.exists()
    await manager.stop()
