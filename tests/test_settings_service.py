import threading
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.services import settings_service
from app.services.capabilities import CATALOG, SettingRequirement
from app.services.settings_service import (
    ACTIVATABLE_FLAG_SETTINGS,
    EDITABLE_BOOL_SETTINGS,
    EDITABLE_SETTINGS_WHITELIST,
    RESTART_REQUIRED_SETTINGS,
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
        {
            "hf_token",
            "rebar_confirmed",
            "enable_file_logging",
            "max_video_upload_mb",
            "enable_audiosr",
            "enable_audio_restore",
            "cad_llm_base_url",
        }
    )


def test_every_catalog_setting_requirement_is_editable() -> None:
    # La regla que evita volver al punto de partida: si alguien suma un
    # SettingRequirement al CATALOG sin sumarlo aca, la tarjeta vuelve a mandar
    # al usuario a editar el .env a mano.
    declared = {
        requirement.setting_attr
        for capability in CATALOG
        for requirement in capability.requirements
        if isinstance(requirement, SettingRequirement)
    }
    assert declared <= EDITABLE_SETTINGS_WHITELIST


def test_bool_settings_are_derived_from_the_declared_field_type() -> None:
    assert EDITABLE_BOOL_SETTINGS == frozenset(
        {"rebar_confirmed", "enable_file_logging", "enable_audiosr", "enable_audio_restore"}
    )


def test_activatable_flags_exclude_the_ones_that_need_a_restart() -> None:
    # cad_llm_base_url no es flag y ademas exige reiniciar: no puede llegar a un
    # boton "Activar" que promete aplicar en el acto.
    assert "cad_llm_base_url" not in ACTIVATABLE_FLAG_SETTINGS
    assert ACTIVATABLE_FLAG_SETTINGS == EDITABLE_BOOL_SETTINGS


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


def _status_by_key(settings: Settings) -> dict[str, dict]:
    return {item["key"]: item for item in editable_settings_status(settings)}


def test_editable_settings_status_reports_configured_flag() -> None:
    configured = _status_by_key(Settings(_env_file=None, HF_TOKEN="x"))
    assert [item["key"] for item in editable_settings_status(Settings(_env_file=None))] == [
        "cad_llm_base_url",
        "enable_audio_restore",
        "enable_audiosr",
        "enable_file_logging",
        "hf_token",
        "max_video_upload_mb",
        "rebar_confirmed",
    ]
    assert configured["hf_token"]["configured"] is True
    # Siempre tiene valor: es un numero con default, no una credencial.
    assert configured["max_video_upload_mb"]["configured"] is True
    assert configured["enable_audiosr"]["configured"] is False
    assert _status_by_key(Settings(_env_file=None))["hf_token"]["configured"] is False


def test_status_exposes_the_value_only_for_the_boolean_flags() -> None:
    status = _status_by_key(
        Settings(_env_file=None, HF_TOKEN="hf_secret", ENABLE_AUDIOSR="true", CAD_LLM_BASE_URL="http://x/v1")
    )
    assert status["enable_audiosr"]["value"] == "true"
    assert status["enable_audio_restore"]["value"] == "false"
    # El secreto NUNCA vuelve, ni siquiera enmascarado.
    assert status["hf_token"]["value"] is None
    assert "hf_secret" not in str(status)
    # Texto libre tampoco: puede traer una credencial en la URL.
    assert status["cad_llm_base_url"]["value"] is None
    assert status["max_video_upload_mb"]["value"] is None


def test_status_marks_which_settings_need_a_restart() -> None:
    status = _status_by_key(Settings(_env_file=None))
    assert status["cad_llm_base_url"]["requires_restart"] is True
    for key in EDITABLE_BOOL_SETTINGS:
        assert status[key]["requires_restart"] is False
    assert RESTART_REQUIRED_SETTINGS == frozenset({"cad_llm_base_url"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", "true"),
        ("True", "true"),
        (" TRUE ", "true"),
        ("1", "true"),
        ("yes", "true"),
        ("on", "true"),
        ("false", "false"),
        ("False", "false"),
        ("0", "false"),
        ("no", "false"),
        ("off", "false"),
    ],
)
def test_update_flag_normalizes_what_it_writes_to_env(env_file: Path, raw: str, expected: str) -> None:
    update_setting("enable_audiosr", raw)
    assert env_file.read_text(encoding="utf-8").strip() == f"ENABLE_AUDIOSR={expected}"


@pytest.mark.parametrize("raw", ["", "  ", "maybe", "2", "sí", "t"])
def test_update_flag_rejects_a_value_that_is_not_a_switch(env_file: Path, raw: str) -> None:
    with pytest.raises(SettingValueError, match="interruptor"):
        update_setting("enable_audiosr", raw)
    assert not env_file.exists()


def test_turning_a_flag_off_leaves_a_real_false_in_the_live_instance(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # La CADENA "false" es truthy: sin coercion, apagar el flag lo dejaba
    # prendido en todo servicio que retiene la instancia viva.
    monkeypatch.chdir(env_file.parent)
    live = Settings(_env_file=None, ENABLE_AUDIOSR="true")
    register_live_settings(live)
    update_setting("enable_audiosr", "false")
    assert live.enable_audiosr is False
    update_setting("enable_audiosr", "true")
    assert live.enable_audiosr is True
