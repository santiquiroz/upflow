import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_get_settings_lists_editable_keys(client) -> None:
    response = client.get("/api/v1/settings")
    assert response.status_code == 200
    payload = response.json()
    keys = [item["key"] for item in payload["settings"]]
    assert "hf_token" in keys
    assert "enable_file_logging" in keys  # switch de logs para diagnosticar
    assert all(isinstance(item["configured"], bool) for item in payload["settings"])


def test_patch_setting_outside_whitelist_is_400(client) -> None:
    response = client.patch("/api/v1/settings", json={"key": "app_port", "value": "1"})
    assert response.status_code == 400
    assert "no es editable" in response.json()["detail"]


def test_patch_hf_token_persists(client, tmp_path, monkeypatch) -> None:
    from app.services import settings_service

    env_path = tmp_path / ".env"
    monkeypatch.setattr(settings_service, "ENV_FILE_PATH", env_path)
    response = client.patch("/api/v1/settings", json={"key": "hf_token", "value": "hf_xyz"})
    assert response.status_code == 200
    assert response.json() == {"key": "hf_token"}
    assert "HF_TOKEN=hf_xyz" in env_path.read_text(encoding="utf-8")


def test_patch_rebar_confirmed_persists(client, tmp_path, monkeypatch) -> None:
    from app.services import settings_service

    env_path = tmp_path / ".env"
    monkeypatch.setattr(settings_service, "ENV_FILE_PATH", env_path)
    response = client.patch(
        "/api/v1/settings",
        json={"key": "rebar_confirmed", "value": "true"},
    )
    assert response.status_code == 200
    assert response.json() == {"key": "rebar_confirmed"}
    assert "REBAR_CONFIRMED=true" in env_path.read_text(encoding="utf-8")


def test_patch_hf_token_rejects_env_file_injection(client, tmp_path, monkeypatch) -> None:
    from app.services import settings_service

    env_path = tmp_path / ".env"
    monkeypatch.setattr(settings_service, "ENV_FILE_PATH", env_path)
    response = client.patch(
        "/api/v1/settings",
        json={"key": "hf_token", "value": "x\nAPP_HOST=evil"},
    )
    assert response.status_code == 400
    assert "Valor inválido" in response.json()["detail"]
    assert not env_path.exists()


def test_get_settings_exposes_the_flag_value_but_never_a_secret(client) -> None:
    payload = client.get("/api/v1/settings").json()
    by_key = {item["key"]: item for item in payload["settings"]}

    # Un interruptor necesita saber si esta en true o false; con `configured`
    # sola, apagado e inexistente se ven igual.
    assert by_key["enable_audiosr"]["value"] in {"true", "false"}
    assert by_key["enable_audio_restore"]["value"] in {"true", "false"}
    # El token nunca vuelve, ni enmascarado.
    assert by_key["hf_token"]["value"] is None
    # Texto libre tampoco: puede traer una credencial adentro.
    assert by_key["cad_llm_base_url"]["value"] is None


def test_get_settings_says_which_settings_need_a_restart(client) -> None:
    payload = client.get("/api/v1/settings").json()
    by_key = {item["key"]: item for item in payload["settings"]}

    assert by_key["cad_llm_base_url"]["requiresRestart"] is True
    assert by_key["enable_audiosr"]["requiresRestart"] is False


def test_patch_capability_flag_persists_normalized(client, tmp_path, monkeypatch) -> None:
    from app.services import settings_service

    env_path = tmp_path / ".env"
    monkeypatch.setattr(settings_service, "ENV_FILE_PATH", env_path)
    response = client.patch("/api/v1/settings", json={"key": "enable_audiosr", "value": "YES"})

    assert response.status_code == 200
    assert "ENABLE_AUDIOSR=true" in env_path.read_text(encoding="utf-8")


def test_patch_capability_flag_rejects_a_non_switch_value(client, tmp_path, monkeypatch) -> None:
    from app.services import settings_service

    env_path = tmp_path / ".env"
    monkeypatch.setattr(settings_service, "ENV_FILE_PATH", env_path)
    response = client.patch(
        "/api/v1/settings", json={"key": "enable_audio_restore", "value": "quizas"}
    )

    assert response.status_code == 400
    assert "interruptor" in response.json()["detail"]
    assert not env_path.exists()
