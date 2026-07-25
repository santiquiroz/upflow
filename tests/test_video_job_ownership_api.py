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


def test_off_mode_video_jobs_list_is_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("AUTH_MODE", raising=False)
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/video/jobs")
            assert response.status_code == 200
            assert response.json() == {"jobs": []}
    finally:
        get_settings.cache_clear()


def test_video_job_endpoints_require_login_in_multi_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
            response = client.get("/api/v1/video/jobs/some-id")
            assert response.status_code == 401
    finally:
        get_settings.cache_clear()


def test_video_jobs_list_all_requires_read_all_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
            client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})
            created = client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
            client.post("/api/v1/auth/logout")
            client.post("/api/v1/auth/login", json={"username": "bob", "password": created["temporaryPassword"]})
            client.post(
                "/api/v1/auth/change-password",
                json={"currentPassword": created["temporaryPassword"], "newPassword": "bobnewpass1"},
            )

            response = client.get("/api/v1/video/jobs?all=true")

            assert response.status_code == 403
    finally:
        get_settings.cache_clear()
