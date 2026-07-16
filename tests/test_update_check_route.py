from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api.routes import get_update_service, update_check
from app.main import app
from app.models import UpdateStatus

# ---------------------------------------------------------------------------
# SP8 Task 1 - GET /api/v1/update-check. Route-level tests call the coroutine
# directly with a fake UpdateService double, plus TestClient(app) wiring tests
# that verify main.py's lifespan mounts app.state.update_service and that the
# JSON body serializes to the documented camelCase shape (200 even on error).
# ---------------------------------------------------------------------------

CHECKED_AT = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


class FakeUpdateService:
    def __init__(self, status: UpdateStatus) -> None:
        self._status = status
        self.force_calls: list[bool] = []

    async def check(self, force: bool = False) -> UpdateStatus:
        self.force_calls.append(force)
        return self._status


def update_available_status() -> UpdateStatus:
    return UpdateStatus(
        current_version="0.1.0",
        latest_version="0.2.0",
        update_available=True,
        release_url="https://github.com/santiquiroz/upflow/releases/tag/v0.2.0",
        published_at="2026-07-16T00:00:00Z",
        checked_at=CHECKED_AT,
        error=None,
    )


def errored_status() -> UpdateStatus:
    return UpdateStatus(
        current_version="0.1.0",
        latest_version=None,
        update_available=False,
        release_url=None,
        published_at=None,
        checked_at=CHECKED_AT,
        error="GitHub unreachable",
    )


async def test_update_check_returns_mapped_status() -> None:
    fake = FakeUpdateService(update_available_status())

    response = await update_check(force=False, updates=fake)

    assert fake.force_calls == [False]
    assert response.current_version == "0.1.0"
    assert response.latest_version == "0.2.0"
    assert response.update_available is True
    assert response.release_url == "https://github.com/santiquiroz/upflow/releases/tag/v0.2.0"


async def test_update_check_forwards_force_flag() -> None:
    fake = FakeUpdateService(update_available_status())

    await update_check(force=True, updates=fake)

    assert fake.force_calls == [True]


def test_endpoint_wired_returns_camel_case_shape() -> None:
    fake = FakeUpdateService(update_available_status())
    app.dependency_overrides[get_update_service] = lambda: fake
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/update-check")
    finally:
        app.dependency_overrides.pop(get_update_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["currentVersion"] == "0.1.0"
    assert body["latestVersion"] == "0.2.0"
    assert body["updateAvailable"] is True
    assert body["releaseUrl"] == "https://github.com/santiquiroz/upflow/releases/tag/v0.2.0"
    assert body["publishedAt"] == "2026-07-16T00:00:00Z"
    assert "checkedAt" in body
    assert body["error"] is None


def test_endpoint_returns_200_even_when_error_set() -> None:
    fake = FakeUpdateService(errored_status())
    app.dependency_overrides[get_update_service] = lambda: fake
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/update-check")
    finally:
        app.dependency_overrides.pop(get_update_service, None)

    assert response.status_code == 200
    body = response.json()
    assert body["updateAvailable"] is False
    assert body["error"] == "GitHub unreachable"
    assert body["latestVersion"] is None


def test_endpoint_forwards_force_query_param() -> None:
    fake = FakeUpdateService(update_available_status())
    app.dependency_overrides[get_update_service] = lambda: fake
    try:
        with TestClient(app) as client:
            client.get("/api/v1/update-check", params={"force": "true"})
    finally:
        app.dependency_overrides.pop(get_update_service, None)

    assert fake.force_calls == [True]


def test_endpoint_wired_through_real_service_default_shape() -> None:
    # No dependency override: exercises the real UpdateService built in the
    # lifespan. It may reach GitHub or fail (offline/rate-limit); either way
    # the endpoint must answer 200 with the documented keys and never 5xx.
    with TestClient(app) as client:
        response = client.get("/api/v1/update-check")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "currentVersion",
        "latestVersion",
        "updateAvailable",
        "releaseUrl",
        "publishedAt",
        "checkedAt",
        "error",
    }
    assert isinstance(body["currentVersion"], str)
