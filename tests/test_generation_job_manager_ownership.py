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
from app.services.generation_job_manager import GenerationJobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry

MODEL_ID = "test-generation-model"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_user(user_id: str = "u1", role: Role = Role.user, overrides: dict | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id, username="alice", role=role, permissions=ROLE_PERMISSIONS[role],
        must_change_password=False, quota_overrides=overrides or {},
    )


class FakeEngine:
    async def run(self, **kwargs):
        return Path(kwargs["output_path"])


def make_manager(tmp_path: Path, quota_service: QuotaService | None = None):
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(ModelEntry(
        id=MODEL_ID, name="Test", kind=ModelKind.diffusion_onnx, source="local", size_bytes=0,
        file_path=MODEL_ID,
    ))
    (settings.models_path / MODEL_ID).mkdir(parents=True, exist_ok=True)
    manager = GenerationJobManager(
        settings, FakeEngine(), DeviceSemaphores(settings), registry=registry,
        upscale_engine=object(), quota_service=quota_service,
    )
    return manager


async def test_create_job_with_owner_stamps_owner_id(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    job = await manager.create_job(prompt="a cat", model_id=MODEL_ID, owner=make_user("u1"))

    assert job.owner_id == "u1"


async def test_create_job_calls_quota_check_admission_when_configured(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    quota_service = QuotaService(settings)
    manager = make_manager(tmp_path, quota_service=quota_service)
    quota_service.attach_managers(manager)
    manager.jobs["existing"] = type("FakeJob", (), {"owner_id": "u1", "status": JobStatus.running})()

    with pytest.raises(QuotaExceededError):
        await manager.create_job(prompt="a cat", model_id=MODEL_ID, owner=make_user("u1"))
