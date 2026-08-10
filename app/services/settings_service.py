from __future__ import annotations

import threading
from typing import TypedDict

from pydantic import ValidationError

from app.config import ENV_FILE_PATH, Settings, get_settings
from app.core.log_file import configure_file_logging
from app.services.json_store import write_text_atomically

# Primer campo real de la whitelist. Crece en subproyectos futuros sin tocar
# el mecanismo (spec 2026-07-25-generation-third-party-models-design.md §5).
#
# Los tres ultimos son los que el CATALOG de capabilities declara como
# SettingRequirement: sin ellos la tarjeta bajaba gigabytes de pack y despues
# mandaba al usuario a editar el .env a mano para poder usar lo que acababa de
# descargar.
EDITABLE_SETTINGS_WHITELIST = frozenset(
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

# Settings cuyo valor NO es texto. El .env guarda strings, pero la app usa el
# numero: `"4096" * 1024 * 1024` en Python REPITE el string en vez de
# multiplicar, y el limite quedaria roto sin que nada falle.
_POSITIVE_INT_SETTINGS = frozenset({"max_video_upload_mb"})


def _is_bool_field(key: str) -> bool:
    return Settings.model_fields[key].annotation is bool


# Los flags. Se DERIVA del tipo declarado en Settings en vez de listarse a mano:
# sumar un flag a la whitelist no puede olvidarse de actualizar esto. Sin el
# conjunto, `setattr(live, "enable_audiosr", "false")` dejaba la CADENA "false"
# en la instancia viva, que es truthy: apagar el flag lo dejaba prendido.
EDITABLE_BOOL_SETTINGS = frozenset(
    key for key in EDITABLE_SETTINGS_WHITELIST if _is_bool_field(key)
)

# Lo que NO aplica en caliente. El cliente del modelo CAD se cablea UNA vez en
# el lifespan (`main.py`, `cad_client=... if settings.cad_llm_base_url else
# None`): escribir la URL con el servidor corriendo no lo crea. Se declara para
# poder DECIRLO en la UI, no para prohibir el cambio.
RESTART_REQUIRED_SETTINGS = frozenset({"cad_llm_base_url"})

# Los que un boton "Activar" puede prender de verdad: flag booleano, editable y
# aplicado en caliente. Un boton que escribe el .env y deja la capacidad igual
# de rota seria peor que no ofrecerlo.
ACTIVATABLE_FLAG_SETTINGS = EDITABLE_BOOL_SETTINGS - RESTART_REQUIRED_SETTINGS

_TRUE_LITERALS = frozenset({"true", "1", "yes", "on"})
_FALSE_LITERALS = frozenset({"false", "0", "no", "off"})

# Serializa read-modify-write del .env entre requests concurrentes.
_ENV_WRITE_LOCK = threading.Lock()

# Instancias retenidas por servicios de vida larga. El lifespan registra la
# suya explícitamente para que los updates no dependan del estado de la cache.
_LIVE_SETTINGS: list[Settings] = []


class SettingNotEditableError(ValueError):
    pass


class SettingValueError(ValueError):
    pass


class EditableSettingStatus(TypedDict):
    key: str
    configured: bool
    # Presente SOLO para los flags. Ver `_visible_value`.
    value: str | None
    requires_restart: bool


def register_live_settings(settings: Settings) -> None:
    # Dedup por IDENTIDAD, no __eq__ de pydantic: dos instancias distintas con
    # valores iguales son servicios vivos distintos y ambas deben mutarse.
    if not any(existing is settings for existing in _LIVE_SETTINGS):
        _LIVE_SETTINGS.append(settings)


def _env_alias(key: str) -> str:
    field = Settings.model_fields[key]
    return field.alias or key.upper()


def _coerce_value(key: str, value: str) -> str | int | bool:
    if key in EDITABLE_BOOL_SETTINGS:
        return value == "true"
    if key in _POSITIVE_INT_SETTINGS:
        return int(value)
    return value


def _normalize_bool(key: str, value: str) -> str:
    text = value.strip().lower()
    if text in _TRUE_LITERALS:
        return "true"
    if text in _FALSE_LITERALS:
        return "false"
    raise SettingValueError(
        f"Valor inválido para {key}: es un interruptor, se espera "
        "true o false (también valen 1/0, yes/no, on/off)."
    )


def normalize_value(key: str, value: str) -> str:
    """Lo que se escribe al .env: un flag SIEMPRE queda como true/false.

    Guardar el "yes" que mando el cliente dejaria el .env valido para pydantic
    pero roto para `_coerce_value`, que compara contra "true".
    """
    return _normalize_bool(key, value) if key in EDITABLE_BOOL_SETTINGS else value


def _validate_positive_int(key: str, value: str) -> None:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SettingValueError(
            f"Valor inválido para {key}: tiene que ser un número entero."
        ) from exc
    if parsed <= 0:
        raise SettingValueError(
            f"Valor inválido para {key}: tiene que ser mayor que cero. Un cero o un "
            "negativo dejaría la app rechazando cualquier archivo."
        )


def _validate_value(key: str, value: str) -> None:
    if key in _POSITIVE_INT_SETTINGS:
        _validate_positive_int(key, value)
    if any(ch in value for ch in ("\n", "\r", "\x00")):
        raise SettingValueError(
            f"Valor inválido para {key}: no puede contener saltos de línea ni caracteres de control."
        )
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
    value = normalize_value(key, value)
    _validate_value(key, value)
    alias = _env_alias(key)
    with _ENV_WRITE_LOCK:
        # Extiende _append_env_var de config.py (append-si-falta) a
        # update-si-existe-o-append, con la misma escritura atómica.
        existing = ENV_FILE_PATH.read_text(encoding="utf-8") if ENV_FILE_PATH.exists() else ""
        write_text_atomically(ENV_FILE_PATH, _render_env_text(existing, alias, value))
        # Los servicios de vida larga (HfClient, installers) guardan la
        # instancia registrada por el lifespan. Se muta directamente sin
        # depender de qué instancia esté cacheada después de updates previos.
        coerced = _coerce_value(key, value)
        for live in _LIVE_SETTINGS:
            setattr(live, key, coerced)
        setattr(get_settings(), key, coerced)
        get_settings.cache_clear()
    # Fuera del lock: el log a archivo se engancha/desengancha en el acto para
    # que un tester pueda encenderlo y reproducir sin reiniciar el servidor.
    if key == "enable_file_logging":
        configure_file_logging(get_settings())


def _visible_value(settings: Settings, key: str) -> str | None:
    """El valor del ajuste, o None si no se puede devolver.

    Solo los flags exponen su valor: prendido o apagado no es un secreto y la UI
    necesita el estado para dibujar el interruptor (con `configured` sola, un
    flag apagado y uno inexistente se ven igual). Todo lo demas —hf_token, y
    cualquier texto libre que pueda cargar una credencial— sigue saliendo como
    `configured` y nada mas.
    """
    if key not in EDITABLE_BOOL_SETTINGS:
        return None
    return "true" if getattr(settings, key) else "false"


def editable_settings_status(settings: Settings) -> list[EditableSettingStatus]:
    return [
        {
            "key": key,
            "configured": bool(getattr(settings, key)),
            "value": _visible_value(settings, key),
            "requires_restart": key in RESTART_REQUIRED_SETTINGS,
        }
        for key in sorted(EDITABLE_SETTINGS_WHITELIST)
    ]
