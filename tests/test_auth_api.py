# tests/test_auth_api.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.auth_routes as auth_routes
from app.main import app
from app.services.auth.identity import LocalPasswordProvider
from app.services.auth.passwords import generate_salt, hash_password
from app.services.auth.permissions import Role
from app.services.auth.quotas import QuotaService
from app.services.auth.sessions import SESSION_COOKIE_NAME
from app.services.auth.user_store import UserStore


@pytest.fixture(autouse=True)
def _reset_login_rate_limit() -> None:
    auth_routes._login_attempts.clear()


@pytest.fixture
def multi_mode_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    from app.config import get_settings
    get_settings.cache_clear()
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        yield client
    get_settings.cache_clear()


def _create_admin(client: TestClient, username: str = "admin", password: str = "adminpass1") -> None:
    response = client.post("/api/v1/auth/setup", json={"username": username, "password": password})
    assert response.status_code == 200, response.text


def test_setup_creates_first_admin(multi_mode_client: TestClient) -> None:
    response = multi_mode_client.post(
        "/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"}
    )
    assert response.status_code == 200


def test_setup_second_time_is_forbidden(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)

    response = multi_mode_client.post(
        "/api/v1/auth/setup", json={"username": "someone-else", "password": "anotherpass1"}
    )

    assert response.status_code == 403


def test_login_succeeds_with_correct_credentials_and_sets_cookie(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)

    response = multi_mode_client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"}
    )

    assert response.status_code == 200
    assert SESSION_COOKIE_NAME in response.cookies


def test_login_fails_with_wrong_password(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)

    response = multi_mode_client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "wrong"}
    )

    assert response.status_code == 401
    assert "usuario" in response.json()["detail"].lower() or "contraseña" in response.json()["detail"].lower()


def test_login_rate_limits_after_five_failed_attempts(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)

    for _ in range(5):
        multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong"})

    response = multi_mode_client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "wrong"}
    )

    assert response.status_code == 429


def test_me_returns_authenticated_user_after_login(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)
    multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})

    response = multi_mode_client.get("/api/v1/auth/me")

    assert response.status_code == 200
    body = response.json()
    assert body["username"] == "admin"
    assert body["role"] == "admin"
    assert body["authMode"] == "multi"
    assert "users:manage" in body["permissions"]


def test_me_without_session_returns_401_not_authenticated(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)

    response = multi_mode_client.get("/api/v1/auth/me")

    assert response.status_code == 401
    assert response.json()["detail"] == "not_authenticated"


def test_me_before_setup_returns_401_setup_required(multi_mode_client: TestClient) -> None:
    response = multi_mode_client.get("/api/v1/auth/me")

    assert response.status_code == 401
    assert response.json()["detail"] == "setup_required"


def test_logout_clears_session(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)
    multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})

    multi_mode_client.post("/api/v1/auth/logout")
    response = multi_mode_client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_logout_all_invalidates_other_sessions(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)
    multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})
    old_cookie = multi_mode_client.cookies.get(SESSION_COOKIE_NAME)

    multi_mode_client.post("/api/v1/auth/logout-all")

    multi_mode_client.cookies.set(SESSION_COOKIE_NAME, old_cookie)
    response = multi_mode_client.get("/api/v1/auth/me")
    assert response.status_code == 401


def test_change_password_updates_and_clears_must_change_password(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)
    multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})

    response = multi_mode_client.post(
        "/api/v1/auth/change-password",
        json={"currentPassword": "adminpass1", "newPassword": "newpassword1"},
    )
    assert response.status_code == 200

    relogin = multi_mode_client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "newpassword1"}
    )
    assert relogin.status_code == 200


def test_change_password_rejects_wrong_current_password(multi_mode_client: TestClient) -> None:
    _create_admin(multi_mode_client)
    multi_mode_client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})

    response = multi_mode_client.post(
        "/api/v1/auth/change-password",
        json={"currentPassword": "wrong", "newPassword": "newpassword1"},
    )

    assert response.status_code == 401


def test_me_reports_off_mode_pseudo_admin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.delenv("AUTH_MODE", raising=False)
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/auth/me")
            assert response.status_code == 200
            body = response.json()
            assert body["authMode"] == "off"
            assert body["userId"] is None
            assert body["role"] == "admin"
    finally:
        get_settings.cache_clear()
