"""Los flags que el CATALOG pide se prenden desde la UI, no editando el .env.

La friccion real que cubre este archivo: la tarjeta bajaba un pack de gigas de
un click y despues decia "falta configurar ENABLE_X", dejando al usuario con un
editor de texto y un reinicio por delante.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services import settings_service
from app.services.capabilities import resolve_one
from app.services.settings_service import register_live_settings, update_setting


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


def _settings_with_audiosr_pack(tmp_path: Path, **overrides: object) -> Settings:
    pack_dir = tmp_path / "audiosr"
    pack_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        RUNTIME_DIR=str(tmp_path),
        AUDIOSR_MODEL_DIR=str(pack_dir),
        _env_file=None,
        **overrides,
    )


def _settings_with_apollo_pack(tmp_path: Path, **overrides: object) -> Settings:
    model = tmp_path / "apollo.onnx"
    model.write_bytes(b"x")
    return Settings(
        RUNTIME_DIR=str(tmp_path),
        APOLLO_RESTORE_MODEL=str(model),
        _env_file=None,
        **overrides,
    )


def test_downloaded_pack_with_the_flag_off_offers_the_flag_as_activatable(tmp_path: Path) -> None:
    settings = _settings_with_audiosr_pack(tmp_path)
    resolved = resolve_one("audio.restoreSr", settings, None)

    assert resolved.status == "needs_setup"
    # El pack ya esta: lo unico que falta es el interruptor.
    assert resolved.missing_packs == ()
    assert resolved.setup_reason_key == "capability.setup.missingSetting"
    assert resolved.activatable_settings == ("enable_audiosr",)


def test_activating_the_flag_turns_the_capability_available_without_restart(
    tmp_path: Path, env_file: Path
) -> None:
    settings = _settings_with_audiosr_pack(tmp_path)
    register_live_settings(settings)
    assert resolve_one("audio.restoreSr", settings, None).status == "needs_setup"

    update_setting("enable_audiosr", "true")

    assert settings.enable_audiosr is True
    assert resolve_one("audio.restoreSr", settings, None).status == "available"
    assert "ENABLE_AUDIOSR=true" in env_file.read_text(encoding="utf-8")


def test_apollo_restore_follows_the_same_path(tmp_path: Path, env_file: Path) -> None:
    settings = _settings_with_apollo_pack(tmp_path)
    register_live_settings(settings)
    assert resolve_one("audio.restore", settings, None).activatable_settings == (
        "enable_audio_restore",
    )

    update_setting("enable_audio_restore", "true")

    assert resolve_one("audio.restore", settings, None).status == "available"


def test_an_available_capability_offers_nothing_to_activate(tmp_path: Path) -> None:
    settings = _settings_with_audiosr_pack(tmp_path, ENABLE_AUDIOSR="true")
    assert resolve_one("audio.restoreSr", settings, None).activatable_settings == ()


def test_a_missing_pack_is_not_activatable_even_with_the_flag_pending(tmp_path: Path) -> None:
    # Sin el pack, prender el flag no arregla nada: primero se descarga.
    settings = Settings(RUNTIME_DIR=str(tmp_path), AUDIOSR_MODEL_DIR=str(tmp_path / "nope"), _env_file=None)
    resolved = resolve_one("audio.restoreSr", settings, None)

    assert resolved.missing_packs == ("audiosr",)
    assert resolved.setup_reason_key == "capability.setup.missingPack"


def test_the_cad_url_is_editable_but_never_offered_as_one_click(tmp_path: Path) -> None:
    # No es un flag y ademas el cliente CAD se cablea una sola vez en el
    # lifespan: un boton "Activar" prometeria algo que no pasa hasta reiniciar.
    binary = tmp_path / "openscad.exe"
    binary.write_bytes(b"x")
    settings = Settings(
        RUNTIME_DIR=str(tmp_path), OPENSCAD_BINARY_PATH=str(binary), _env_file=None
    )
    resolved = resolve_one("print.cad", settings, None)

    assert resolved.setup_reason_key == "capability.setup.missingSetting"
    assert resolved.activatable_settings == ()


def test_the_restorer_wired_at_startup_sees_the_flag_without_being_rebuilt(
    tmp_path: Path, env_file: Path
) -> None:
    # La prueba de que "en caliente" no es una promesa: los restorers se
    # construyen UNA vez en el lifespan y leen `self.settings` en cada
    # available(), asi que mutar la instancia viva alcanza.
    from app.config import APOLLO_MODE
    from app.services.gpu_session_coordinator import GpuSessionCoordinator
    from app.services.restorer_registry import build_restorers, validate_restore_mode_ready

    settings = _settings_with_apollo_pack(tmp_path)
    register_live_settings(settings)
    restorer = build_restorers(settings, GpuSessionCoordinator())[APOLLO_MODE]
    assert restorer.available() is False
    with pytest.raises(ValueError, match="ENABLE_AUDIO_RESTORE"):
        validate_restore_mode_ready(settings, APOLLO_MODE)

    update_setting("enable_audio_restore", "true")

    assert restorer.available() is True
    validate_restore_mode_ready(settings, APOLLO_MODE)
