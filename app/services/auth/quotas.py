# app/services/auth/quotas.py
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.exceptions import QuotaExceededError
from app.models import JobStatus, utc_now
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import Role
from app.services.json_store import backup_corrupt_file, write_json_atomically

logger = logging.getLogger(__name__)

OVERRIDE_KEYS = ("max_concurrent", "max_queued", "max_jobs_per_day", "max_gpu_seconds_per_day")


@dataclass(frozen=True, slots=True)
class RoleQuota:
    max_concurrent: int
    max_queued: int
    max_jobs_per_day: int
    max_gpu_seconds_per_day: int


DEFAULT_ROLE_QUOTAS: dict[Role, RoleQuota] = {
    Role.user: RoleQuota(max_concurrent=1, max_queued=5, max_jobs_per_day=50, max_gpu_seconds_per_day=3600),
    Role.admin: RoleQuota(max_concurrent=0, max_queued=0, max_jobs_per_day=0, max_gpu_seconds_per_day=0),
}


@dataclass(slots=True)
class UsageRecord:
    date: str
    jobs: int = 0
    gpu_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class QuotaStatus:
    max_concurrent: int
    max_queued: int
    max_jobs_per_day: int
    max_gpu_seconds_per_day: int
    used_jobs_today: int
    used_gpu_seconds_today: float


def _today() -> str:
    return utc_now().date().isoformat()


def _effective_quota(role: Role, overrides: dict[str, int]) -> RoleQuota:
    base = DEFAULT_ROLE_QUOTAS[role]
    values = {key: overrides.get(key, getattr(base, key)) for key in OVERRIDE_KEYS}
    return RoleQuota(**values)


class QuotaService:
    def __init__(self, settings: Settings) -> None:
        self._path = settings.usage_file_path
        self._lock = threading.Lock()
        self._usage: dict[str, UsageRecord] = self._load()
        self._managers: tuple[Any, ...] = ()

    def attach_managers(self, *managers: Any) -> None:
        self._managers = managers

    def check_admission(self, user: AuthenticatedUser) -> None:
        if user.id is None:
            return
        quota = _effective_quota(user.role, user.quota_overrides)
        self._check_concurrency(user.id, quota)
        self._check_queue(user.id, quota)
        self._check_daily(user.id, quota)

    def _check_concurrency(self, user_id: str, quota: RoleQuota) -> None:
        if not quota.max_concurrent:
            return
        running = self._count_owned(user_id, JobStatus.running)
        if running >= quota.max_concurrent:
            raise QuotaExceededError(
                f"Tenés {running} job(s) corriendo y tu límite es {quota.max_concurrent}."
            )

    def _check_queue(self, user_id: str, quota: RoleQuota) -> None:
        if not quota.max_queued:
            return
        queued = self._count_owned(user_id, JobStatus.queued)
        if queued >= quota.max_queued:
            raise QuotaExceededError(
                f"Tenés {queued} job(s) en cola y tu límite es {quota.max_queued}."
            )

    def _check_daily(self, user_id: str, quota: RoleQuota) -> None:
        usage = self._current_usage(user_id)
        if quota.max_jobs_per_day and usage.jobs >= quota.max_jobs_per_day:
            raise QuotaExceededError(
                f"Límite diario alcanzado: {quota.max_jobs_per_day} jobs. Se resetea a medianoche."
            )
        if quota.max_gpu_seconds_per_day and usage.gpu_seconds >= quota.max_gpu_seconds_per_day:
            raise QuotaExceededError(
                f"Límite diario de GPU alcanzado: {quota.max_gpu_seconds_per_day}s. Se resetea a medianoche."
            )

    def _count_owned(self, user_id: str, status: JobStatus) -> int:
        return sum(
            1
            for manager in self._managers
            for job in manager.jobs.values()
            if job.owner_id == user_id and job.status == status
        )

    def record_usage(self, user_id: str | None, gpu_seconds: float) -> None:
        if user_id is None:
            return
        with self._lock:
            usage = self._current_usage(user_id)
            self._usage[user_id] = UsageRecord(
                date=usage.date, jobs=usage.jobs + 1, gpu_seconds=usage.gpu_seconds + gpu_seconds
            )
            self._persist()

    def status_for(self, user: AuthenticatedUser) -> QuotaStatus:
        quota = _effective_quota(user.role, user.quota_overrides)
        usage = self._current_usage(user.id) if user.id is not None else UsageRecord(date=_today())
        return QuotaStatus(
            max_concurrent=quota.max_concurrent, max_queued=quota.max_queued,
            max_jobs_per_day=quota.max_jobs_per_day, max_gpu_seconds_per_day=quota.max_gpu_seconds_per_day,
            used_jobs_today=usage.jobs, used_gpu_seconds_today=usage.gpu_seconds,
        )

    def _current_usage(self, user_id: str) -> UsageRecord:
        today = _today()
        record = self._usage.get(user_id)
        if record is None or record.date != today:
            return UsageRecord(date=today)
        return record

    def _load(self) -> dict[str, UsageRecord]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {
                user_id: UsageRecord(date=item["date"], jobs=item["jobs"], gpu_seconds=item["gpu_seconds"])
                for user_id, item in raw.items()
            }
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            backup_corrupt_file(self._path, exc, logger)
            return {}

    def _persist(self) -> None:
        payload = {
            user_id: {"date": usage.date, "jobs": usage.jobs, "gpu_seconds": usage.gpu_seconds}
            for user_id, usage in self._usage.items()
        }
        write_json_atomically(self._path, payload)
