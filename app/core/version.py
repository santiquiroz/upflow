from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

DEFAULT_PACKAGE_NAME = "upflow"
FALLBACK_VERSION = "0.0.0"
PYPROJECT_PATH = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"


def get_app_version(package_name: str = DEFAULT_PACKAGE_NAME) -> str:
    """Resolves the running app version without ever raising.

    Prefiere el pyproject.toml que viaja JUNTO al código: es, por definición,
    la versión del código que se está ejecutando. La metadata del paquete es un
    caché que se desincroniza — `pip install -e .` corre una sola vez y después
    el código se actualiza sin que nadie la regenere (visto en instalaciones
    reales reportando 0.10.0 con código 0.14.0), y en esta app la raíz de
    instalación entra en sys.path, así que un `upflow.egg-info` viejo ahí puede
    tapar hasta el dist-info bueno de site-packages.

    La metadata queda como fallback para instalaciones normales desde wheel,
    donde no hay pyproject al lado del código. `package_name` es inyectable
    para reusar el mecanismo en otros proyectos.
    """
    return _version_from_pyproject() or _version_from_metadata(package_name) or FALLBACK_VERSION


def _version_from_metadata(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_from_pyproject() -> str | None:
    try:
        data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        return data["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None
