from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.services.missing_pack import PACK_LABELS
from app.services.realtime_service import (
    UPFLOW_MODE_NAME,
    RealtimeService,
    apply_upflow_mode,
    available_presets,
    magpie_config_path,
)

# ---------------------------------------------------------------------------
# Magpie se maneja por su config.json (verificado corriendo el binario real
# v0.12.1: deja %LOCALAPPDATA%\Magpie\config\v4\config.json con scalingModes y
# profiles[0].scalingMode como indice del modo activo). Upflow es el plano de
# control: parchea ese archivo y lanza el proceso. Sin driver, sin linkear nada
# — Magpie es GPL-3.0 y corre aparte.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), **overrides)  # type: ignore[arg-type]


def config_del_usuario() -> dict:
    """Un config como el que deja Magpie de verdad, con ajustes propios."""
    return {
        "language": "es",
        "theme": 2,
        "shortcuts": {"scale": 3137, "toolbar": 3140},
        "scalingModes": [
            {"name": "Lanczos", "effects": [{"name": "Lanczos"}]},
            {"name": "FSR", "effects": [{"name": "FSR\\FSR_EASU"}]},
        ],
        "profiles": [{"scalingMode": 1, "captureMethod": 0, "maxFrameRate": 60.0}],
        "unaClaveQueNoConocemos": {"algo": 1},
    }


# --- parcheo del config (nucleo puro y testeable) ------------------------


def test_adding_our_mode_does_not_touch_the_users_modes() -> None:
    config = config_del_usuario()
    apply_upflow_mode(config, preset="anime4k")
    nombres = [m["name"] for m in config["scalingModes"]]
    assert nombres[:2] == ["Lanczos", "FSR"]
    assert UPFLOW_MODE_NAME in nombres


def test_the_active_profile_points_at_our_mode() -> None:
    config = config_del_usuario()
    apply_upflow_mode(config, preset="anime4k")
    indice = config["profiles"][0]["scalingMode"]
    assert config["scalingModes"][indice]["name"] == UPFLOW_MODE_NAME


def test_applying_twice_updates_in_place_instead_of_duplicating() -> None:
    config = config_del_usuario()
    apply_upflow_mode(config, preset="anime4k")
    apply_upflow_mode(config, preset="fsr")
    nuestros = [m for m in config["scalingModes"] if m["name"] == UPFLOW_MODE_NAME]
    assert len(nuestros) == 1
    assert "FSR" in json.dumps(nuestros[0])


def test_unknown_keys_survive() -> None:
    """El config es de Magpie, no nuestro: una version nueva puede traer claves
    que no conocemos y borrarlas seria romperle los ajustes al usuario."""
    config = config_del_usuario()
    apply_upflow_mode(config, preset="anime4k")
    assert config["unaClaveQueNoConocemos"] == {"algo": 1}
    assert config["language"] == "es"
    assert config["shortcuts"]["scale"] == 3137


def test_the_users_other_profile_settings_are_kept() -> None:
    config = config_del_usuario()
    apply_upflow_mode(config, preset="anime4k")
    perfil = config["profiles"][0]
    assert perfil["captureMethod"] == 0
    assert perfil["maxFrameRate"] == 60.0


def test_an_empty_config_becomes_usable() -> None:
    config: dict = {}
    apply_upflow_mode(config, preset="anime4k")
    assert config["scalingModes"][config["profiles"][0]["scalingMode"]]["name"] == UPFLOW_MODE_NAME


def test_an_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="anime4k"):
        apply_upflow_mode({}, preset="no-existe")


def test_a_frame_rate_cap_is_optional_and_wired_when_asked() -> None:
    config = config_del_usuario()
    apply_upflow_mode(config, preset="anime4k", max_frame_rate=120)
    perfil = config["profiles"][0]
    assert perfil["frameRateLimiterEnabled"] is True
    assert perfil["maxFrameRate"] == 120.0


def test_presets_are_named_for_humans_not_for_magpie() -> None:
    presets = {p.id: p for p in available_presets()}
    assert "anime4k" in presets
    # La copia dejo de vivir en Python: el preset declara la clave y el frontend
    # la traduce. test_display_copy_lives_in_the_catalog verifica que la clave
    # exista de verdad en los dos catalogos.
    assert presets["anime4k"].label_key
    assert presets["anime4k"].description_key


# --- disponibilidad y arranque -------------------------------------------


def test_unavailable_without_the_binary(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, MAGPIE_BINARY=str(tmp_path / "no-esta.exe"))
    assert RealtimeService(settings).available() is False


def test_available_once_the_binary_is_there(tmp_path: Path) -> None:
    binario = tmp_path / "Magpie.exe"
    binario.write_bytes(b"exe")
    settings = make_settings(tmp_path, MAGPIE_BINARY=str(binario))
    assert RealtimeService(settings).available() is True


def test_starting_without_the_binary_dice_que_paquete_falta_sin_dictar_comandos(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path, MAGPIE_BINARY=str(tmp_path / "no-esta.exe"))
    with pytest.raises(RuntimeError) as error:
        RealtimeService(settings).start(preset="anime4k")

    mensaje = str(error.value)
    # Importa que identifique QUE falta, no la redaccion exacta.
    assert PACK_LABELS["magpie"] in mensaje
    # La regresion a evitar: el mensaje no manda al usuario a una terminal.
    assert ".ps1" not in mensaje


def test_starting_writes_the_config_and_launches(tmp_path: Path, monkeypatch) -> None:
    binario = tmp_path / "Magpie.exe"
    binario.write_bytes(b"exe")
    config = tmp_path / "cfg" / "config.json"
    config.parent.mkdir()
    config.write_text(json.dumps(config_del_usuario()), encoding="utf-8")
    settings = make_settings(tmp_path, MAGPIE_BINARY=str(binario), MAGPIE_CONFIG=str(config))

    lanzados: list = []
    service = RealtimeService(settings)
    monkeypatch.setattr(service, "_spawn", lambda cmd: lanzados.append(cmd) or 4321)

    pid = service.start(preset="anime4k")

    assert pid == 4321
    assert lanzados == [[str(binario)]]
    escrito = json.loads(config.read_text(encoding="utf-8"))
    indice = escrito["profiles"][0]["scalingMode"]
    assert escrito["scalingModes"][indice]["name"] == UPFLOW_MODE_NAME
    # Los ajustes del usuario siguen ahi.
    assert escrito["language"] == "es"


def test_a_corrupt_config_is_not_silently_overwritten(tmp_path: Path, monkeypatch) -> None:
    """Pisarle el config roto al usuario le borraria sus modos sin avisar."""
    binario = tmp_path / "Magpie.exe"
    binario.write_bytes(b"exe")
    config = tmp_path / "cfg" / "config.json"
    config.parent.mkdir()
    config.write_text("{ esto no es json", encoding="utf-8")
    settings = make_settings(tmp_path, MAGPIE_BINARY=str(binario), MAGPIE_CONFIG=str(config))
    service = RealtimeService(settings)
    monkeypatch.setattr(service, "_spawn", lambda cmd: 1)

    with pytest.raises(RuntimeError, match="config"):
        service.start(preset="anime4k")

    assert config.read_text(encoding="utf-8") == "{ esto no es json"


def test_a_missing_config_is_created(tmp_path: Path, monkeypatch) -> None:
    binario = tmp_path / "Magpie.exe"
    binario.write_bytes(b"exe")
    config = tmp_path / "cfg" / "config.json"
    settings = make_settings(tmp_path, MAGPIE_BINARY=str(binario), MAGPIE_CONFIG=str(config))
    service = RealtimeService(settings)
    monkeypatch.setattr(service, "_spawn", lambda cmd: 1)

    service.start(preset="anime4k")

    escrito = json.loads(config.read_text(encoding="utf-8"))
    assert escrito["scalingModes"][escrito["profiles"][0]["scalingMode"]]["name"] == UPFLOW_MODE_NAME


def test_the_default_config_path_is_where_magpie_really_puts_it(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, MAGPIE_CONFIG="")
    ruta = magpie_config_path(settings)
    assert ruta.name == "config.json"
    # Verificado corriendo Magpie v0.12.1 de verdad.
    assert ruta.parent.name == "v4"
    assert "Magpie" in str(ruta)
