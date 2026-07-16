from __future__ import annotations

import importlib.metadata
import tomllib
from pathlib import Path

PACKAGE_NAME = "upflow"
FALLBACK_VERSION = "0.0.0"
PYPROJECT_PATH = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"


def get_app_version() -> str:
    """Resolves the running app version without ever raising.

    Prefers the installed package metadata (works after `pip install -e .`);
    falls back to parsing pyproject.toml for a plain source checkout; final
    fallback keeps every caller total even if both sources are missing.
    """
    return _version_from_metadata() or _version_from_pyproject() or FALLBACK_VERSION


def _version_from_metadata() -> str | None:
    try:
        return importlib.metadata.version(PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_from_pyproject() -> str | None:
    try:
        data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        return data["project"]["version"]
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None
