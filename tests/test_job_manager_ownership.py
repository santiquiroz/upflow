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
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager


class FakeEngine(UpscaleEngine):
    def available(self) -> bool:
        return True

    async def run(self, job):
        return job.source_path


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_user(user_id: str = "u1", role: Role = Role.user, overrides: dict | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id, username="alice", role=role, permissions=ROLE_PERMISSIONS[role],
        must_change_password=False, quota_overrides=overrides or {},
    )


def make_image_file(tmp_path: Path) -> Path:
    from PIL import Image

    path = tmp_path / "source.png"
    Image.new("RGB", (8, 8)).save(path)
    return path


async def test_create_job_without_owner_leaves_owner_id_none(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeEngine(), DeviceSemaphores(settings))

    job = await manager.create_job(
        source_path=make_image_file(tmp_path), original_filename="a.png",
        model_name="realesrgan-x4plus", scale=4, output_format="png",
    )

    assert job.owner_id is None


async def test_create_job_with_owner_stamps_owner_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeEngine(), DeviceSemaphores(settings))

    job = await manager.create_job(
        source_path=make_image_file(tmp_path), original_filename="a.png",
        model_name="realesrgan-x4plus", scale=4, output_format="png", owner=make_user("u1"),
    )

    assert job.owner_id == "u1"


async def test_create_job_calls_quota_check_admission_when_configured(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    quota_service = QuotaService(settings)
    manager = JobManager(settings, FakeEngine(), DeviceSemaphores(settings), quota_service=quota_service)
    quota_service.attach_managers(manager)
    manager.jobs["existing"] = type(
        "FakeJob", (), {"owner_id": "u1", "status": JobStatus.running}
    )()

    with pytest.raises(QuotaExceededError):
        await manager.create_job(
            source_path=make_image_file(tmp_path), original_filename="a.png",
            model_name="realesrgan-x4plus", scale=4, output_format="png", owner=make_user("u1"),
        )


async def test_create_job_skips_quota_check_when_owner_is_none(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    quota_service = QuotaService(settings)
    manager = JobManager(settings, FakeEngine(), DeviceSemaphores(settings), quota_service=quota_service)
    quota_service.attach_managers(manager)

    job = await manager.create_job(
        source_path=make_image_file(tmp_path), original_filename="a.png",
        model_name="realesrgan-x4plus", scale=4, output_format="png",
    )

    assert job.owner_id is None
