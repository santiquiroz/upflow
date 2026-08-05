from __future__ import annotations

import pytest

from app.config import Settings
from app.services.settings_service import (
    EDITABLE_SETTINGS_WHITELIST,
    SettingValueError,
    register_live_settings,
    update_setting,
)

# ---------------------------------------------------------------------------
# El limite de subida de video protege el DISCO, no la red: el pipeline extrae
# cada frame como imagen, asi que un video de 2 GB puede volverse cientos de GB
# de PNG. Pero es un numero de politica, no un techo tecnico, y hay maquinas con
# disco de sobra donde 2 GB queda corto.
#
# Editarlo requiere el permiso `settings_write`, igual que el resto de los
# settings editables. Lo que se prueba aca es que el valor llegue como ENTERO:
# el whitelist guardaba strings, y `"4096" * 1024 * 1024` en Python no
# multiplica, REPITE el string. El limite quedaria roto sin que nada falle.
# ---------------------------------------------------------------------------


@pytest.fixture()
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("APP_NAME=Upflow\n", encoding="utf-8")
    monkeypatch.setattr("app.services.settings_service.ENV_FILE_PATH", path)
    return path


def test_the_video_upload_limit_is_editable_from_the_ui() -> None:
    assert "max_video_upload_mb" in EDITABLE_SETTINGS_WHITELIST


def test_updating_the_limit_leaves_a_number_not_a_string(env_file, monkeypatch) -> None:
    live = Settings(_env_file=None)
    register_live_settings(live)

    update_setting("max_video_upload_mb", "4096")

    # Lo que importa es el TIPO: con el valor como string, `limit * 1024 * 1024`
    # repetiria el texto en vez de multiplicar.
    assert live.max_video_upload_mb == 4096
    assert isinstance(live.max_video_upload_mb, int)
    # No se afirma sobre `get_settings()`: tras el `cache_clear` vuelve a leer el
    # .env del directorio de configuracion real, que este fixture no controla.


def test_the_new_limit_reaches_the_env_file(env_file) -> None:
    update_setting("max_video_upload_mb", "8192")

    assert "MAX_VIDEO_UPLOAD_MB=8192" in env_file.read_text(encoding="utf-8")


@pytest.mark.parametrize("bad", ["0", "-1", "no-soy-un-numero", "1.5"])
def test_a_limit_that_would_reject_everything_is_refused(env_file, bad: str) -> None:
    """Un cero o un negativo dejaria la app rechazando cualquier archivo."""
    with pytest.raises(SettingValueError):
        update_setting("max_video_upload_mb", bad)
