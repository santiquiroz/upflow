from __future__ import annotations

# ---------------------------------------------------------------------------
# Log a archivo, opt-in, para diagnosticar reportes de otras máquinas.
#
# Sin esto, un "va lentísimo" desde la PC de otro es adivinar: el warning que
# avisa que el upscale cayó a tiling tras un OOM ya se emite, pero va a una
# consola que nadie guarda. Con el switch encendido el tester reproduce el
# problema y manda un archivo.
#
# Apagado por DEFECTO a propósito: en uso normal es ruido y disco, y esta app
# ya tuvo un bug de disco lleno. Cuando se enciende, el archivo rota con techo
# duro para que tampoco pueda crecer sin control.
# ---------------------------------------------------------------------------

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOG_FILENAME = "upflow.log"
LOG_DIRNAME = "logs"

# Marca para reconocer NUESTRO handler entre los del root (uvicorn agrega los
# suyos): así reconfigurar es idempotente y apagar sabe qué desenganchar.
_HANDLER_MARKER = "_upflow_file_handler"

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

logger = logging.getLogger(__name__)


def log_file_path(settings: Any) -> Path:
    return settings.runtime_path / LOG_DIRNAME / LOG_FILENAME


def configure_file_logging(settings: Any) -> Path | None:
    """Engancha o desengancha el handler de archivo según el setting.

    Idempotente: llamarla dos veces no duplica handlers, así que sirve tanto al
    arranque como cuando el toggle cambia en caliente. Devuelve la ruta del
    archivo cuando queda activo, None cuando queda apagado (o no se pudo).
    """
    root = logging.getLogger()
    existing = _find_handler(root)

    if not getattr(settings, "enable_file_logging", False):
        _detach(root, existing)
        return None

    if existing is not None:
        return Path(existing.baseFilename)

    path = log_file_path(settings)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path,
            maxBytes=max(1, settings.log_file_max_mb) * 1024 * 1024,
            backupCount=max(0, settings.log_file_backups),
            encoding="utf-8",
        )
    except OSError:
        # El logging es diagnóstico: que no se pueda escribir NUNCA puede
        # tumbar la app ni impedir que un job corra.
        logger.exception("no se pudo abrir el archivo de log en %s", path)
        return None

    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    # El root suele quedar en WARNING: sin esto los logger.info del pipeline
    # (etapas, decisiones de ruteo) no llegarían al archivo.
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    return path


def _find_handler(root: logging.Logger) -> RotatingFileHandler | None:
    for handler in root.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return handler  # type: ignore[return-value]
    return None


def _detach(root: logging.Logger, handler: RotatingFileHandler | None) -> None:
    if handler is None:
        return
    root.removeHandler(handler)
    handler.close()
