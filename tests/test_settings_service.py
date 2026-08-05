import threading
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.services import settings_service
from app.services.settings_service import (
    EDITABLE_SETTINGS_WHITELIST,
    SettingNotEditableError,
    SettingValueError,
    editable_settings_status,
    register_live_settings,
    update_setting,
)


@pytest.fixture()
def env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / ".env"
    monkeypatch.setattr(settings_service, "ENV_FILE_PATH", path)
    return path


@pytest.fixture(autouse=True)
def clear_live_settings_registry():
    settings_service._LIVE_SETTINGS.clear()
    yield
    settings_service._LIVE_SETTINGS.clear()


def test_whitelist_contains_the_ui_editable_settings() -> None:
    assert EDITABLE_SETTINGS_WHITELIST == frozenset(
        {"hf_token", "rebar_confirmed", "enable_file_logging", "max_video_upload_mb"}
    )


def test_update_setting_rejects_key_outside_whitelist(env_file: Path) -> None:
    with pytest.raises(SettingNotEditableError, match="app_port"):
        update_setting("app_port", "9999")
    assert not env_file.exists()


def test_update_setting_appends_when_env_missing(env_file: Path) -> None:
    update_setting("hf_token", "hf_abc123")
    assert env_file.read_text(encoding="utf-8").strip() == "HF_TOKEN=hf_abc123"


def test_update_setting_replaces_existing_line_preserving_others(env_file: Path) -> None:
    env_file.write_text("APP_PORT=8090\nHF_TOKEN=hf_old\nDEFAULT_DEVICE=dml:0\n", encoding="utf-8")
    update_setting("hf_token", "hf_new")
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert lines == ["APP_PORT=8090", "HF_TOKEN=hf_new", "DEFAULT_DEVICE=dml:0"]


def test_update_setting_rejects_env_file_injection(env_file: Path) -> None:
    original = "APP_PORT=8090\n"
    env_file.write_text(original, encoding="utf-8")
    with pytest.raises(SettingValueError, match="Valor inválido"):
        update_setting("hf_token", "x\nAPP_HOST=evil")
    assert env_file.read_text(encoding="utf-8") == original


def test_update_setting_clears_get_settings_cache(env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # get_settings lee env_file=".env" relativo al CWD -- se apunta el CWD al
    # tmp para que la lectura y la escritura miren el mismo archivo.
    monkeypatch.chdir(env_file.parent)
    get_settings.cache_clear()
    assert get_settings().hf_token is None
    update_setting("hf_token", "hf_fresh")
    assert get_settings().hf_token == "hf_fresh"
    get_settings.cache_clear()


def test_update_setting_propagates_to_live_settings_instance(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # La instancia que los servicios de vida larga retienen es la cacheada
    # ANTES del update -- debe ver el valor nuevo sin reconstruirse.
    monkeypatch.chdir(env_file.parent)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    get_settings.cache_clear()
    live_instance = get_settings()  # como hace main.py en el lifespan
    register_live_settings(live_instance)
    assert live_instance.hf_token is None
    update_setting("hf_token", "hf_live")
    assert live_instance.hf_token == "hf_live"
    get_settings.cache_clear()


def test_register_live_settings_registers_equal_but_distinct_instances(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(env_file.parent)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    first = Settings(_env_file=None)
    second = Settings(_env_file=None)
    assert first == second and first is not second
    register_live_settings(first)
    register_live_settings(second)
    update_setting("hf_token", "hf_both")
    assert first.hf_token == "hf_both"
    assert second.hf_token == "hf_both"


def test_update_setting_propagates_across_multiple_updates(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(env_file.parent)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    get_settings.cache_clear()
    live = get_settings()
    register_live_settings(live)
    update_setting("hf_token", "token_A")
    update_setting("hf_token", "token_B")
    assert live.hf_token == "token_B"
    get_settings.cache_clear()


def test_concurrent_updates_do_not_corrupt_env(env_file: Path) -> None:
    env_file.write_text("APP_PORT=8090\n", encoding="utf-8")
    errors: list[Exception] = []

    def writer(value: str) -> None:
        try:
            for _ in range(20):
                update_setting("hf_token", value)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(f"hf_{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    lines = env_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "APP_PORT=8090"
    assert len([line for line in lines if line.startswith("HF_TOKEN=")]) == 1


def test_editable_settings_status_reports_configured_flag() -> None:
    configured = editable_settings_status(Settings(_env_file=None, HF_TOKEN="x"))
    assert configured == [
        {"key": "enable_file_logging", "configured": False},
        {"key": "hf_token", "configured": True},
        # Siempre tiene valor: es un numero con default, no una credencial.
        {"key": "max_video_upload_mb", "configured": True},
        {"key": "rebar_confirmed", "configured": False},
    ]
    empty = editable_settings_status(Settings(_env_file=None))
    assert empty == [
        {"key": "enable_file_logging", "configured": False},
        {"key": "hf_token", "configured": False},
        {"key": "max_video_upload_mb", "configured": True},
        {"key": "rebar_confirmed", "configured": False},
    ]
