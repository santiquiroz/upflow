# tests/test_quotas.py
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.exceptions import QuotaExceededError
from app.models import JobStatus
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import ROLE_PERMISSIONS, Role
from app.services.auth.quotas import QuotaService


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_user(user_id: str = "u1", role: Role = Role.user, overrides: dict | None = None) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id, username="alice", role=role, permissions=ROLE_PERMISSIONS[role],
        must_change_password=False, quota_overrides=overrides or {},
    )


class FakeJob:
    def __init__(self, owner_id: str | None, status: JobStatus) -> None:
        self.owner_id = owner_id
        self.status = status


class FakeManager:
    def __init__(self, jobs: dict[str, FakeJob]) -> None:
        self.jobs = jobs


def test_check_admission_passes_for_user_with_no_jobs(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({}))

    service.check_admission(make_user())  # should not raise


def test_check_admission_raises_when_concurrency_limit_reached(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({"j1": FakeJob("u1", JobStatus.running)}))

    with pytest.raises(QuotaExceededError, match="corriendo"):
        service.check_admission(make_user())  # user default max_concurrent=1


def test_check_admission_raises_when_queue_limit_reached(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    queued_jobs = {f"j{i}": FakeJob("u1", JobStatus.queued) for i in range(5)}
    service.attach_managers(FakeManager(queued_jobs))

    with pytest.raises(QuotaExceededError, match="cola"):
        service.check_admission(make_user())  # user default max_queued=5


def test_check_admission_ignores_other_users_jobs(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({"j1": FakeJob("someone-else", JobStatus.running)}))

    service.check_admission(make_user())  # should not raise


def test_check_admission_skips_off_mode_pseudo_user(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    running = {f"j{i}": FakeJob(None, JobStatus.running) for i in range(10)}
    service.attach_managers(FakeManager(running))

    off_mode_user = AuthenticatedUser(
        id=None, username="local", role=Role.admin, permissions=ROLE_PERMISSIONS[Role.admin],
        must_change_password=False, quota_overrides={},
    )
    service.check_admission(off_mode_user)  # should not raise: id is None


def test_check_admission_admin_role_is_unlimited(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    running = {f"j{i}": FakeJob("admin-1", JobStatus.running) for i in range(50)}
    service.attach_managers(FakeManager(running))

    service.check_admission(make_user("admin-1", role=Role.admin))  # should not raise


def test_check_admission_respects_quota_override(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({"j1": FakeJob("u1", JobStatus.running)}))

    service.check_admission(make_user(overrides={"max_concurrent": 5}))  # should not raise


def test_record_usage_then_check_admission_raises_at_daily_job_limit(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({}))
    user = make_user(overrides={"max_jobs_per_day": 2, "max_concurrent": 0, "max_queued": 0})

    service.record_usage(user.id, gpu_seconds=1.0)
    service.record_usage(user.id, gpu_seconds=1.0)

    with pytest.raises(QuotaExceededError, match="diario"):
        service.check_admission(user)


def test_record_usage_accumulates_gpu_seconds_and_raises_at_limit(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.attach_managers(FakeManager({}))
    user = make_user(overrides={"max_gpu_seconds_per_day": 10, "max_concurrent": 0, "max_queued": 0})

    service.record_usage(user.id, gpu_seconds=6.0)
    service.record_usage(user.id, gpu_seconds=5.0)

    with pytest.raises(QuotaExceededError, match="GPU"):
        service.check_admission(user)


def test_record_usage_none_user_id_is_a_no_op(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    service.record_usage(None, gpu_seconds=100.0)  # should not raise, nothing persisted
    assert not service._path.exists()


def test_status_for_reports_used_and_max_values(tmp_path: Path) -> None:
    service = QuotaService(make_settings(tmp_path))
    user = make_user()
    service.record_usage(user.id, gpu_seconds=42.0)

    status = service.status_for(user)

    assert status.used_jobs_today == 1
    assert status.used_gpu_seconds_today == 42.0
    assert status.max_concurrent == 1
    assert status.max_jobs_per_day == 50


def test_usage_persists_across_instances(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = QuotaService(settings)
    service.record_usage("u1", gpu_seconds=10.0)

    reloaded = QuotaService(settings)
    status = reloaded.status_for(make_user())

    assert status.used_gpu_seconds_today == 10.0


def test_usage_resets_lazily_on_new_day(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    service = QuotaService(settings)
    service.record_usage("u1", gpu_seconds=10.0)
    # Simulate a day rollover by rewriting yesterday's date directly on disk.
    payload = json.loads(settings.usage_file_path.read_text(encoding="utf-8"))
    payload["u1"]["date"] = "2000-01-01"
    settings.usage_file_path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = QuotaService(settings)
    status = reloaded.status_for(make_user())

    assert status.used_jobs_today == 0
    assert status.used_gpu_seconds_today == 0.0


def test_corrupt_usage_file_is_backed_up_and_reset(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings.usage_file_path.parent.mkdir(parents=True, exist_ok=True)
    settings.usage_file_path.write_text("not json", encoding="utf-8")

    service = QuotaService(settings)

    assert service.status_for(make_user()).used_jobs_today == 0
    backups = list(settings.usage_file_path.parent.glob("usage.json.corrupt-*"))
    assert len(backups) == 1
