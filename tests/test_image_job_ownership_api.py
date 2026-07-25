from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import auth_routes
from app.config import get_settings
from app.main import app


@pytest.fixture(autouse=True)
def _reset_login_rate_limit() -> None:
    auth_routes._login_attempts.clear()


@pytest.fixture
def two_user_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
        client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})
        created = client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
        yield client, created["user"]["id"], created["temporaryPassword"]
    get_settings.cache_clear()


def test_off_mode_existing_image_job_flow_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("AUTH_MODE", raising=False)
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/jobs")
            assert response.status_code == 200
            assert response.json() == {"jobs": []}
    finally:
        get_settings.cache_clear()


def test_list_jobs_shows_only_own_jobs_by_default(two_user_client) -> None:
    admin_client, bob_id, bob_password = two_user_client
    admin_client.post("/api/v1/auth/logout")
    admin_client.post("/api/v1/auth/login", json={"username": "bob", "password": bob_password})
    admin_client.post(
        "/api/v1/auth/change-password",
        json={"currentPassword": bob_password, "newPassword": "bobnewpass1"},
    )

    response = admin_client.get("/api/v1/jobs")

    assert response.status_code == 200
    assert response.json()["jobs"] == []


def test_list_jobs_with_all_flag_requires_read_all_permission(two_user_client) -> None:
    admin_client, bob_id, bob_password = two_user_client
    admin_client.post("/api/v1/auth/logout")
    admin_client.post("/api/v1/auth/login", json={"username": "bob", "password": bob_password})
    admin_client.post(
        "/api/v1/auth/change-password",
        json={"currentPassword": bob_password, "newPassword": "bobnewpass1"},
    )

    response = admin_client.get("/api/v1/jobs?all=true")

    assert response.status_code == 403


def test_get_job_returns_404_for_someone_elses_job(two_user_client) -> None:
    admin_client, bob_id, _bob_password = two_user_client
    response = admin_client.get("/api/v1/jobs/some-job-id-owned-by-nobody")
    assert response.status_code == 404


def test_create_job_requires_login_in_multi_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
            response = client.post("/api/v1/jobs", files={"file": ("a.png", b"not-a-real-png", "image/png")})
            assert response.status_code == 401
    finally:
        get_settings.cache_clear()
