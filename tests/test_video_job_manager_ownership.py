from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.exceptions import QuotaExceededError
from app.models import JobStatus
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import ROLE_PERMISSIONS, Role
from app.services.auth.quotas import QuotaService
from app.services.device_semaphores import DeviceSemaphores
from app.services.video_job_manager import VideoJobManager


class FakeUpscaler:
    async def run(self, job, fps_multiplier: int = 1):
        return job.source_path


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video", "avg_frame_rate": "24/1"}]}


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_user(user_id: str = "u1", role: Role = Role.user, overrides: dict | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id, username="alice", role=role, permissions=ROLE_PERMISSIONS[role],
        must_change_password=False, quota_overrides=overrides or {},
    )


def make_manager(tmp_path: Path, quota_service: QuotaService | None = None) -> VideoJobManager:
    settings = make_settings(tmp_path)
    return VideoJobManager(
        settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings), quota_service=quota_service,
    )


async def test_create_job_without_owner_leaves_owner_id_none(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")

    job = await manager.create_job(
        source_path=source, original_filename="source.mp4", model_name="realesr-animevideov3-x2",
        scale=2, output_container="mp4", video_codec="libx264", video_preset="medium", crf=18,
        keep_audio=False,
    )

    assert job.owner_id is None


async def test_create_job_with_owner_stamps_owner_id(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")

    job = await manager.create_job(
        source_path=source, original_filename="source.mp4", model_name="realesr-animevideov3-x2",
        scale=2, output_container="mp4", video_codec="libx264", video_preset="medium", crf=18,
        keep_audio=False, owner=make_user("u1"),
    )

    assert job.owner_id == "u1"


async def test_create_job_calls_quota_check_admission_when_configured(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    quota_service = QuotaService(settings)
    manager = make_manager(tmp_path, quota_service=quota_service)
    quota_service.attach_managers(manager)
    manager.jobs["existing"] = type("FakeJob", (), {"owner_id": "u1", "status": JobStatus.running})()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake")

    with pytest.raises(QuotaExceededError):
        await manager.create_job(
            source_path=source, original_filename="source.mp4", model_name="realesr-animevideov3-x2",
            scale=2, output_container="mp4", video_codec="libx264", video_preset="medium", crf=18,
            keep_audio=False, owner=make_user("u1"),
        )
