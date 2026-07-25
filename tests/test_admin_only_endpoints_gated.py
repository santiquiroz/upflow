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
def bob_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
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
        yield client
    get_settings.cache_clear()


def test_install_model_requires_models_install_permission(bob_client: TestClient) -> None:
    response = bob_client.post("/api/v1/models/install", json={"repoId": "org/model"})
    assert response.status_code == 403


def test_delete_model_requires_models_delete_permission(bob_client: TestClient) -> None:
    response = bob_client.delete("/api/v1/models/some-model-id")
    assert response.status_code == 403


def test_install_generation_model_requires_models_install_permission(bob_client: TestClient) -> None:
    response = bob_client.post("/api/v1/generation/models", json={"repoId": "org/model"})
    assert response.status_code == 403


def test_rescan_capabilities_requires_settings_write_permission(bob_client: TestClient) -> None:
    response = bob_client.post("/api/v1/capabilities/rescan")
    assert response.status_code == 403


def test_fix_lever_requires_settings_write_permission(bob_client: TestClient) -> None:
    response = bob_client.post("/api/v1/capabilities/hags/fix")
    assert response.status_code == 403


def test_scan_onnx_diagnostic_requires_settings_write_permission(bob_client: TestClient) -> None:
    response = bob_client.post("/api/v1/capabilities/onnx-diagnostics/some-model/dml:0/scan")
    assert response.status_code == 403


def test_off_mode_admin_only_endpoints_still_reachable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("AUTH_MODE", raising=False)
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/capabilities/rescan")
            assert response.status_code != 403
    finally:
        get_settings.cache_clear()
