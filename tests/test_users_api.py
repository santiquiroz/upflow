# tests/test_users_api.py
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.auth_routes as auth_routes
from app.config import get_settings
from app.main import app


@pytest.fixture(autouse=True)
def _reset_login_rate_limit() -> None:
    auth_routes._login_attempts.clear()


@pytest.fixture
def admin_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
        client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})
        yield client
    get_settings.cache_clear()


def test_create_user_returns_temporary_password(admin_client: TestClient) -> None:
    response = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"})

    assert response.status_code == 201
    body = response.json()
    assert body["user"]["username"] == "bob"
    assert body["user"]["mustChangePassword"] is True
    assert len(body["temporaryPassword"]) > 0


def test_create_user_rejects_duplicate_username(admin_client: TestClient) -> None:
    admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"})

    response = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"})

    assert response.status_code == 409


def test_list_users_includes_admin_and_created_users(admin_client: TestClient) -> None:
    admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"})

    response = admin_client.get("/api/v1/users")

    assert response.status_code == 200
    usernames = {u["username"] for u in response.json()["users"]}
    assert usernames == {"admin", "bob"}


def test_update_user_role_and_disabled(admin_client: TestClient) -> None:
    created = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
    user_id = created["user"]["id"]

    response = admin_client.patch(f"/api/v1/users/{user_id}", json={"role": "admin", "disabled": True})

    assert response.status_code == 200
    body = response.json()["user"]
    assert body["role"] == "admin"
    assert body["disabled"] is True


def test_update_user_reset_password_returns_temporary_password(admin_client: TestClient) -> None:
    created = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
    user_id = created["user"]["id"]

    response = admin_client.patch(f"/api/v1/users/{user_id}", json={"resetPassword": True})

    assert response.status_code == 200
    body = response.json()
    assert body["temporaryPassword"] is not None
    assert body["user"]["mustChangePassword"] is True


def test_update_unknown_user_returns_404(admin_client: TestClient) -> None:
    response = admin_client.patch("/api/v1/users/does-not-exist", json={"disabled": True})
    assert response.status_code == 404


def test_users_endpoints_require_users_manage_permission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post("/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"})
            client.post("/api/v1/auth/login", json={"username": "admin", "password": "adminpass1"})
            created = client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
            temp_password = created["temporaryPassword"]
            client.post("/api/v1/auth/logout")

            client.post("/api/v1/auth/login", json={"username": "bob", "password": temp_password})
            response = client.get("/api/v1/users")

            assert response.status_code == 403
    finally:
        get_settings.cache_clear()


def test_get_user_jobs_returns_empty_list_for_new_user(admin_client: TestClient) -> None:
    created = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
    user_id = created["user"]["id"]

    response = admin_client.get(f"/api/v1/users/{user_id}/jobs")

    assert response.status_code == 200
    assert response.json()["jobs"] == []


def test_get_user_jobs_includes_karaoke_transcribe_shape3d_and_download(
    admin_client: TestClient,
) -> None:
    # Regresion: get_user_jobs solo agregaba image/video/audio/generation.
    # karaoke/transcribe/shape3d/download tienen owner_id igual que los
    # otros cuatro (la cuota y el ownership ya los cubren) pero un admin no
    # los veia en la vista de "trabajos de este usuario".
    from app.main import app as fastapi_app
    from app.models import DownloadJob, KaraokeJob, Shape3dJob, TranscribeJob

    created = admin_client.post("/api/v1/users", json={"username": "bob", "role": "user"}).json()
    user_id = created["user"]["id"]

    karaoke = KaraokeJob(
        source_path=Path("song.mp3"), original_filename="song.mp3", asr_model_id="m",
    )
    karaoke.owner_id = user_id
    fastapi_app.state.karaoke_jobs.jobs[karaoke.id] = karaoke

    transcribe = TranscribeJob(source_path=Path("a.mp3"), original_filename="a.mp3", model_id="m")
    transcribe.owner_id = user_id
    fastapi_app.state.transcribe_jobs.jobs[transcribe.id] = transcribe

    shape3d = Shape3dJob(prompt="a small ceramic mug")
    shape3d.owner_id = user_id
    fastapi_app.state.shape3d_jobs.jobs[shape3d.id] = shape3d

    download = DownloadJob(url="https://example.com/v")
    download.owner_id = user_id
    fastapi_app.state.download_jobs.jobs[download.id] = download

    response = admin_client.get(f"/api/v1/users/{user_id}/jobs")

    assert response.status_code == 200
    kinds = {job["kind"] for job in response.json()["jobs"]}
    assert kinds == {"karaoke", "transcribe", "shape3d", "download"}
