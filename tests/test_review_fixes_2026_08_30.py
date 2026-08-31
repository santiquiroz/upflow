"""Regresiones de la auditoria 2026-08-30.

Cubre: el RetentionSweeper protege la fuente de un karaoke en render y poda
su dict de jobs terminados -- el mismo olvido que la auditoria 2026-08-08
encontro para transcribe/shape3d/download (`test_review_fixes_2026_08.py`),
esta vez con karaoke, que se agrego mas tarde y quedo afuera igual.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path

from app.config import Settings
from app.models import JobStatus, KaraokeJob, utc_now
from app.services.retention_sweeper import RetentionSweeper
from app.services.storage import StorageService


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


class _StubManager:
    def __init__(self) -> None:
        self.jobs: dict[str, object] = {}


def _sweeper_with_karaoke_stub(settings: Settings) -> tuple[RetentionSweeper, _StubManager]:
    karaoke_stub = _StubManager()
    sweeper = RetentionSweeper(
        settings,
        _StubManager(),
        _StubManager(),
        karaoke_job_manager=karaoke_stub,
    )
    return sweeper, karaoke_stub


def _old_karaoke_job(source: Path, status: JobStatus) -> KaraokeJob:
    job = KaraokeJob(source_path=source, original_filename=source.name, asr_model_id="m")
    job.status = status
    if status in (JobStatus.completed, JobStatus.failed, JobStatus.cancelled):
        job.finished_at = utc_now() - timedelta(hours=999)
    return job


def _make_old_upload(settings: Settings, name: str) -> Path:
    path = settings.uploads_path / name
    path.write_bytes(b"x")
    stale = time.time() - 999 * 3600
    os.utime(path, (stale, stale))
    return path


def test_sweeper_protects_rendering_karaoke_source(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    sweeper, karaoke_stub = _sweeper_with_karaoke_stub(settings)
    source = _make_old_upload(settings, "cancion.mp3")
    job = _old_karaoke_job(source, JobStatus.running)
    karaoke_stub.jobs[job.id] = job

    sweeper.sweep_once()

    assert source.exists(), "el sweep borro la fuente de un karaoke en render"


def test_sweeper_prunes_finished_karaoke_jobs(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    sweeper, karaoke_stub = _sweeper_with_karaoke_stub(settings)
    old_job = _old_karaoke_job(settings.uploads_path / "vieja.mp3", JobStatus.completed)
    karaoke_stub.jobs[old_job.id] = old_job

    sweeper.sweep_once()

    assert karaoke_stub.jobs == {}, "karaoke no se poda"


def test_sweeper_without_karaoke_manager_still_works(tmp_path: Path) -> None:
    # karaoke_job_manager es opcional (None por default): un caller que no lo
    # pase -- como los tests viejos de RetentionSweeper -- no debe romper.
    settings = make_settings(tmp_path)
    sweeper = RetentionSweeper(settings, _StubManager(), _StubManager())

    sweeper.sweep_once()
