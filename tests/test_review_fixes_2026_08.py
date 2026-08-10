"""Regresiones de la auditoria 2026-08-08.

Cubre: (1) cancel de shape3d no resucita al terminar el hilo, (2) el
DownloadJobManager descuenta cuota, (3) el RetentionSweeper protege las
fuentes de transcripcion y poda los dicts de los tres managers nuevos.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import timedelta
from pathlib import Path

import numpy as np

from app.config import Settings
from app.models import DownloadJob, JobStatus, TranscribeJob, utc_now
from app.services.auth.quotas import QuotaService
from app.services.download_job_manager import DownloadJobManager
from app.services.retention_sweeper import RetentionSweeper
from app.services.shape3d_job_manager import Shape3dJobManager
from app.services.storage import StorageService


def tetrahedron(scale: float = 10.0) -> np.ndarray:
    a, b, c, d = (0.0, 0.0, 0.0), (scale, 0.0, 0.0), (0.0, scale, 0.0), (0.0, 0.0, scale)
    return np.array([[a, c, b], [a, b, d], [a, d, c], [b, c, d]], dtype=np.float64)


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


# ---------------------------------------------------------------------------
# (1) shape3d: cancel a mitad de generacion no resucita a completed.
# ---------------------------------------------------------------------------
class BlockingEngine:
    """Bloquea el hilo de generacion hasta que el test lo libere."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def available(self) -> bool:
        return True

    def generate_from_text(self, prompt: str, **_kwargs) -> np.ndarray:
        self.started.set()
        self.release.wait(timeout=5)
        return tetrahedron()


async def test_shape3d_cancel_mid_run_does_not_resurrect(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = BlockingEngine()
    manager = Shape3dJobManager(settings, engine)
    job = await manager.create_job(prompt="una pieza")

    run = asyncio.create_task(manager._process_next())
    await asyncio.to_thread(engine.started.wait, 5)

    assert manager.cancel_job(job.id) is True
    cancelled_at = job.finished_at
    assert job.status is JobStatus.cancelled

    engine.release.set()
    await asyncio.wait_for(run, timeout=5)

    assert job.status is JobStatus.cancelled, "el cancel se piso al terminar el hilo"
    assert job.finished_at == cancelled_at, "finished_at del cancel fue pisado"


async def test_shape3d_failure_after_cancel_stays_cancelled(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    class FailingBlockingEngine(BlockingEngine):
        def generate_from_text(self, prompt: str, **_kwargs) -> np.ndarray:
            self.started.set()
            self.release.wait(timeout=5)
            raise RuntimeError("boom")

    engine = FailingBlockingEngine()
    manager = Shape3dJobManager(settings, engine)
    job = await manager.create_job(prompt="otra pieza")

    run = asyncio.create_task(manager._process_next())
    await asyncio.to_thread(engine.started.wait, 5)
    manager.cancel_job(job.id)
    engine.release.set()
    await asyncio.wait_for(run, timeout=5)

    assert job.status is JobStatus.cancelled
    assert job.error is None


# ---------------------------------------------------------------------------
# (2) download: la cuota se descuenta al terminar el job.
# ---------------------------------------------------------------------------
async def test_download_job_records_quota_usage(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    quota = QuotaService(settings)
    manager = DownloadJobManager(settings, quota_service=quota)
    job = DownloadJob(url="https://example.com/v", owner_id="user-1")
    manager.jobs[job.id] = job

    monkeypatch.setattr(manager, "_download_blocking", lambda *_args: None)
    await manager._run_job(job)

    assert job.status is JobStatus.completed
    record = quota._usage.get("user-1")
    assert record is not None and record.jobs == 1, "la descarga no descontó cuota"


async def test_download_failed_job_also_records_quota_usage(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    quota = QuotaService(settings)
    manager = DownloadJobManager(settings, quota_service=quota)
    job = DownloadJob(url="https://example.com/v", owner_id="user-1")
    manager.jobs[job.id] = job

    def _boom(*_args):
        raise RuntimeError("network down")

    monkeypatch.setattr(manager, "_download_blocking", _boom)
    await manager._run_job(job)

    assert job.status is JobStatus.failed
    record = quota._usage.get("user-1")
    assert record is not None and record.jobs == 1


# ---------------------------------------------------------------------------
# (3) sweeper: protege fuentes de transcripcion activas y poda los dicts
# de transcribe/shape3d/download.
# ---------------------------------------------------------------------------
class _StubManager:
    def __init__(self) -> None:
        self.jobs: dict[str, object] = {}


def _sweeper_with_stubs(settings: Settings, **named_managers) -> tuple[RetentionSweeper, dict]:
    stubs = {
        "job_manager": _StubManager(),
        "video_job_manager": _StubManager(),
        "audio_job_manager": _StubManager(),
        "generation_job_manager": _StubManager(),
        "transcribe_job_manager": _StubManager(),
        "shape3d_job_manager": _StubManager(),
        "download_job_manager": _StubManager(),
    }
    stubs.update(named_managers)
    sweeper = RetentionSweeper(
        settings,
        stubs["job_manager"],
        stubs["video_job_manager"],
        stubs["audio_job_manager"],
        generation_job_manager=stubs["generation_job_manager"],
        transcribe_job_manager=stubs["transcribe_job_manager"],
        shape3d_job_manager=stubs["shape3d_job_manager"],
        download_job_manager=stubs["download_job_manager"],
    )
    return sweeper, stubs


def _old_transcribe_job(source: Path, status: JobStatus) -> TranscribeJob:
    job = TranscribeJob(source_path=source, original_filename=source.name, model_id="m")
    job.status = status
    if status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
        job.finished_at = utc_now() - timedelta(hours=999)
    return job


def _make_old_upload(settings: Settings, name: str) -> Path:
    import os
    import time

    path = settings.uploads_path / name
    path.write_bytes(b"x")
    stale = time.time() - 999 * 3600
    os.utime(path, (stale, stale))
    return path


def test_sweeper_protects_queued_transcribe_source(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    sweeper, stubs = _sweeper_with_stubs(settings)
    source = _make_old_upload(settings, "pendiente.wav")
    job = _old_transcribe_job(source, JobStatus.queued)
    stubs["transcribe_job_manager"].jobs[job.id] = job

    sweeper.sweep_once()

    assert source.exists(), "el sweep borro la fuente de un job de transcripcion activo"


# ---------------------------------------------------------------------------
# (4) doble loudnorm: con mastering activo, el paso `loudness` de la cadena de
# voz se omite (normalizar dos veces — la primera single-pass — bombea).
# ---------------------------------------------------------------------------
def _audio_pipeline(tmp_path: Path):
    from app.services.audio_pipeline import AudioPipeline

    return AudioPipeline(make_settings(tmp_path), audio_enhancers={}, restorers={})


def _voice_job(tmp_path: Path, *, master: str | None, steps: list[str]):
    from app.models import AudioJob

    return AudioJob(
        source_path=tmp_path / "in.wav",
        original_filename="in.wav",
        voice_steps=steps,
        master=master,
    )


def test_loudness_step_is_skipped_when_mastering_is_active(tmp_path: Path) -> None:
    pipeline = _audio_pipeline(tmp_path)
    job = _voice_job(tmp_path, master="streaming", steps=["denoise", "loudness"])

    steps = pipeline._voice_steps_without_double_loudnorm(job)

    assert steps == ["denoise"]
    assert "voiceLoudnessSkipped" in job.metadata


def test_loudness_step_stays_without_mastering(tmp_path: Path) -> None:
    pipeline = _audio_pipeline(tmp_path)
    job = _voice_job(tmp_path, master=None, steps=["denoise", "loudness"])

    steps = pipeline._voice_steps_without_double_loudnorm(job)

    assert steps == ["denoise", "loudness"]
    assert "voiceLoudnessSkipped" not in job.metadata


def _stage_keys(job) -> list[str]:
    from app.services.progress import build_audio_stages

    return [stage.key for stage in build_audio_stages(job)]


def test_stage_map_hides_voice_and_mastering_when_not_requested(tmp_path: Path) -> None:
    # Reporte de campo (2026-08-10): un job denoise+restore mostraba
    # "Enhancing voice" en el stepper sin haber pedido cadena de voz.
    from app.models import AudioJob

    job = AudioJob(
        source_path=tmp_path / "in.wav",
        original_filename="in.wav",
        denoise="deepfilternet",
        restore="audiosr",
    )

    keys = _stage_keys(job)

    assert "voicing" not in keys
    assert "mastering" not in keys
    assert keys == ["decoding", "denoising", "restoring", "finalizing"]


def test_stage_map_shows_the_steps_that_were_requested(tmp_path: Path) -> None:
    from app.models import AudioJob

    job = AudioJob(
        source_path=tmp_path / "in.wav",
        original_filename="in.wav",
        denoise="deepfilternet",
        voice_steps=["compress"],
        master="streaming",
    )

    keys = _stage_keys(job)

    assert keys == ["decoding", "denoising", "voicing", "mastering", "finalizing"]


def test_stage_map_hides_voicing_when_mastering_swallows_the_only_step(tmp_path: Path) -> None:
    # El pipeline descarta `loudness` con mastering activo: pintarlo seria
    # mostrar una etapa que nunca corre.
    from app.models import AudioJob

    job = AudioJob(
        source_path=tmp_path / "in.wav",
        original_filename="in.wav",
        denoise="deepfilternet",
        voice_steps=["loudness"],
        master="streaming",
    )

    keys = _stage_keys(job)

    assert "voicing" not in keys
    assert "mastering" in keys


def test_only_loudness_plus_mastering_skips_the_voice_stage_entirely(tmp_path: Path) -> None:
    pipeline = _audio_pipeline(tmp_path)
    job = _voice_job(tmp_path, master="streaming", steps=["loudness"])

    assert pipeline._voice_steps_without_double_loudnorm(job) == []


def test_sweeper_prunes_finished_jobs_of_new_managers(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    sweeper, stubs = _sweeper_with_stubs(settings)

    old_transcribe = _old_transcribe_job(settings.uploads_path / "t.wav", JobStatus.completed)
    stubs["transcribe_job_manager"].jobs[old_transcribe.id] = old_transcribe

    old_download = DownloadJob(url="https://example.com/v")
    old_download.status = JobStatus.completed
    old_download.finished_at = utc_now() - timedelta(hours=999)
    stubs["download_job_manager"].jobs[old_download.id] = old_download

    sweeper.sweep_once()

    assert stubs["transcribe_job_manager"].jobs == {}, "transcribe no se poda"
    assert stubs["download_job_manager"].jobs == {}, "download no se poda"
