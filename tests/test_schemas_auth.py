from __future__ import annotations

from app.models import JobStatus
from app.schemas import JobResponse, MeResponse, QuotaStatusResponse


def test_job_response_serializes_owner_id_as_camel_case() -> None:
    response = JobResponse(
        job_id="j1", status=JobStatus.queued, original_filename="a.png", model_name="m", scale=4,
        output_format="png", created_at="2026-01-01T00:00:00+00:00", owner_id="user-1",
    )
    assert response.model_dump(by_alias=True)["ownerId"] == "user-1"


def test_job_response_owner_id_defaults_to_none() -> None:
    response = JobResponse(
        job_id="j1", status=JobStatus.queued, original_filename="a.png", model_name="m", scale=4,
        output_format="png", created_at="2026-01-01T00:00:00+00:00",
    )
    assert response.model_dump(by_alias=True)["ownerId"] is None


def test_me_response_serializes_camel_case() -> None:
    quota = QuotaStatusResponse(
        max_concurrent=1, max_queued=5, max_jobs_per_day=50, max_gpu_seconds_per_day=3600,
        used_jobs_today=0, used_gpu_seconds_today=0.0,
    )
    response = MeResponse(
        user_id="u1", username="alice", role="user", permissions=["jobs:create"],
        must_change_password=False, auth_mode="multi", quota=quota,
    )
    dumped = response.model_dump(by_alias=True)
    assert dumped["userId"] == "u1"
    assert dumped["mustChangePassword"] is False
    assert dumped["authMode"] == "multi"
    assert dumped["quota"]["maxConcurrent"] == 1
