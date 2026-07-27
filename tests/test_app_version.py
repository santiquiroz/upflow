from __future__ import annotations

import importlib.metadata
from pathlib import Path

import pytest

from app.core import version as version_module
from app.core.version import FALLBACK_VERSION, get_app_version


def write_pyproject(path: Path, value: str) -> Path:
    path.write_text(f'[project]\nname = "upflow"\nversion = "{value}"\n', encoding="utf-8")
    return path


def test_pyproject_next_to_the_code_wins_over_stale_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Caso real: en la instalación, `pip install -e .` corre UNA sola vez, así
    # que la metadata queda congelada en la versión que se instaló primero
    # mientras el código se actualiza con cada release. El pyproject que viaja
    # junto al código sí es, por definición, la versión que está corriendo.
    monkeypatch.setattr(version_module, "PYPROJECT_PATH", write_pyproject(tmp_path / "pyproject.toml", "0.14.0"))
    monkeypatch.setattr(version_module, "_version_from_metadata", lambda name: "0.10.0")

    assert get_app_version() == "0.14.0"


def test_falls_back_to_metadata_when_no_pyproject_ships_with_the_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Instalación normal desde wheel: no hay pyproject al lado del código.
    monkeypatch.setattr(version_module, "PYPROJECT_PATH", tmp_path / "no-existe.toml")
    monkeypatch.setattr(version_module, "_version_from_metadata", lambda name: "1.2.3")

    assert get_app_version() == "1.2.3"


def test_falls_back_to_the_sentinel_when_neither_source_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(version_module, "PYPROJECT_PATH", tmp_path / "no-existe.toml")
    monkeypatch.setattr(version_module, "_version_from_metadata", lambda name: None)

    assert get_app_version() == FALLBACK_VERSION


def test_malformed_pyproject_does_not_raise_and_defers_to_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = tmp_path / "pyproject.toml"
    broken.write_text("esto no es toml valido [[[", encoding="utf-8")
    monkeypatch.setattr(version_module, "PYPROJECT_PATH", broken)
    monkeypatch.setattr(version_module, "_version_from_metadata", lambda name: "9.9.9")

    assert get_app_version() == "9.9.9"


def test_real_repo_pyproject_matches_the_running_version() -> None:
    # Guard de integración: la versión que reporta la app es la del pyproject
    # del repo, sin depender de si el paquete está instalado o no.
    import tomllib

    repo_pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.loads(repo_pyproject.read_text(encoding="utf-8"))["project"]["version"]

    assert get_app_version() == declared


def test_metadata_lookup_never_raises_for_a_missing_package() -> None:
    assert version_module._version_from_metadata("paquete-que-no-existe-xyz") is None
    with pytest.raises(importlib.metadata.PackageNotFoundError):
        importlib.metadata.version("paquete-que-no-existe-xyz")
