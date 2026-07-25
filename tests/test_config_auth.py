from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings, ensure_auth_secret


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def test_auth_mode_defaults_to_off(tmp_path: Path) -> None:
    assert make_settings(tmp_path).auth_mode == "off"


def test_auth_mode_accepts_multi(tmp_path: Path) -> None:
    assert make_settings(tmp_path, AUTH_MODE="multi").auth_mode == "multi"


def test_auth_mode_rejects_invalid_value(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="AUTH_MODE"):
        make_settings(tmp_path, AUTH_MODE="bogus")


def test_users_file_path_is_under_runtime_auth_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.users_file_path == settings.runtime_path / "auth" / "users.json"


def test_usage_file_path_is_under_runtime_auth_dir(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.usage_file_path == settings.runtime_path / "auth" / "usage.json"


def test_ensure_auth_secret_generates_and_persists_when_missing(tmp_path: Path, monkeypatch) -> None:
    import app.config as config_module

    env_path = tmp_path / ".env"
    monkeypatch.setattr(config_module, "ENV_FILE_PATH", env_path)
    settings = make_settings(tmp_path)
    assert settings.auth_secret is None

    secret = ensure_auth_secret(settings)

    assert len(secret) == 64  # 32 bytes hex-encoded
    assert settings.auth_secret == secret
    assert f"AUTH_SECRET={secret}" in env_path.read_text(encoding="utf-8")


def test_ensure_auth_secret_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    import app.config as config_module

    env_path = tmp_path / ".env"
    monkeypatch.setattr(config_module, "ENV_FILE_PATH", env_path)
    settings = make_settings(tmp_path)

    first = ensure_auth_secret(settings)
    second = ensure_auth_secret(settings)

    assert first == second
    assert env_path.read_text(encoding="utf-8").count("AUTH_SECRET=") == 1


def test_ensure_auth_secret_appends_without_clobbering_existing_content(tmp_path: Path, monkeypatch) -> None:
    import app.config as config_module

    env_path = tmp_path / ".env"
    env_path.write_text("APP_PORT=8090", encoding="utf-8")
    monkeypatch.setattr(config_module, "ENV_FILE_PATH", env_path)
    settings = make_settings(tmp_path)

    ensure_auth_secret(settings)

    content = env_path.read_text(encoding="utf-8")
    assert "APP_PORT=8090" in content
    assert "AUTH_SECRET=" in content
