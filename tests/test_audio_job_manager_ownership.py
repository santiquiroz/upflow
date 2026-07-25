from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.exceptions import QuotaExceededError
from app.models import JobStatus
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import ROLE_PERMISSIONS, Role
from app.services.auth.quotas import QuotaService
from app.services.audio_job_manager import AudioJobManager
from app.services.device_semaphores import DeviceSemaphores


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), ENABLE_AUDIO_ENHANCE=True)


def make_user(user_id: str = "u1", role: Role = Role.user, overrides: dict | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id, username="alice", role=role, permissions=ROLE_PERMISSIONS[role],
        must_change_password=False, quota_overrides=overrides or {},
    )


def make_manager(tmp_path: Path, quota_service: QuotaService | None = None) -> AudioJobManager:
    settings = make_settings(tmp_path)
    return AudioJobManager(settings, pipeline=None, device_semaphores=DeviceSemaphores(settings), quota_service=quota_service)


async def test_create_job_with_owner_stamps_owner_id(tmp_path: Path, monkeypatch) -> None:
    manager = make_manager(tmp_path)
    monkeypatch.setattr(type(manager.settings), "audio_enhance_available", lambda self, mode: True)
    source = tmp_path / "source.wav"
    source.write_bytes(b"fake")

    job = await manager.create_job(
        source_path=source, original_filename="source.wav", denoise="deepfilter", owner=make_user("u1"),
    )

    assert job.owner_id == "u1"


async def test_create_job_calls_quota_check_admission_when_configured(tmp_path: Path, monkeypatch) -> None:
    settings = make_settings(tmp_path)
    quota_service = QuotaService(settings)
    manager = AudioJobManager(settings, pipeline=None, device_semaphores=DeviceSemaphores(settings), quota_service=quota_service)
    monkeypatch.setattr(type(manager.settings), "audio_enhance_available", lambda self, mode: True)
    quota_service.attach_managers(manager)
    manager.jobs["existing"] = type("FakeJob", (), {"owner_id": "u1", "status": JobStatus.running})()
    source = tmp_path / "source.wav"
    source.write_bytes(b"fake")

    with pytest.raises(QuotaExceededError):
        await manager.create_job(
            source_path=source, original_filename="source.wav", denoise="deepfilter", owner=make_user("u1"),
        )
