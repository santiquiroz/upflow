from __future__ import annotations

import threading
from typing import TypedDict

from pydantic import ValidationError

from app.config import ENV_FILE_PATH, Settings, get_settings
from app.services.json_store import write_text_atomically

# Primer campo real de la whitelist. Crece en subproyectos futuros sin tocar
# el mecanismo (spec 2026-07-25-generation-third-party-models-design.md §5).
EDITABLE_SETTINGS_WHITELIST = frozenset({"hf_token"})

# Serializa read-modify-write del .env entre requests concurrentes.
_ENV_WRITE_LOCK = threading.Lock()


class SettingNotEditableError(ValueError):
    pass


class SettingValueError(ValueError):
    pass


class EditableSettingStatus(TypedDict):
    key: str
    configured: bool


def _env_alias(key: str) -> str:
    field = Settings.model_fields[key]
    return field.alias or key.upper()


def _validate_value(key: str, value: str) -> None:
    # Reusa la validación pydantic real del campo: un valor inválido para el
    # tipo del campo revienta acá con 400, nunca llega al .env.
    try:
        Settings(_env_file=None, **{_env_alias(key): value})
    except ValidationError as exc:
        raise SettingValueError(
            f"Valor inválido para {key}: {exc.errors()[0].get('msg', 'validación fallida')}"
        ) from exc


def _render_env_text(existing_text: str, alias: str, value: str) -> str:
    prefix = f"{alias}="
    lines = existing_text.splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[index] = f"{alias}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{alias}={value}")
    return "\n".join(lines) + "\n"


def update_setting(key: str, value: str) -> None:
    if key not in EDITABLE_SETTINGS_WHITELIST:
        raise SettingNotEditableError(f"El setting {key!r} no es editable desde la UI.")
    _validate_value(key, value)
    alias = _env_alias(key)
    with _ENV_WRITE_LOCK:
        # Extiende _append_env_var de config.py (append-si-falta) a
        # update-si-existe-o-append, con la misma escritura atómica.
        existing = ENV_FILE_PATH.read_text(encoding="utf-8") if ENV_FILE_PATH.exists() else ""
        write_text_atomically(ENV_FILE_PATH, _render_env_text(existing, alias, value))
    # Los servicios de vida larga (HfClient, installers) guardan la instancia
    # de Settings del lifespan -- que es la cacheada en get_settings() antes
    # del clear. Mutarla propaga el valor nuevo sin reiniciar (mismo patrón
    # que ensure_auth_secret con auth_secret).
    setattr(get_settings(), key, value)
    get_settings.cache_clear()


def editable_settings_status(settings: Settings) -> list[EditableSettingStatus]:
    return [
        {"key": key, "configured": bool(getattr(settings, key))}
        for key in sorted(EDITABLE_SETTINGS_WHITELIST)
    ]
