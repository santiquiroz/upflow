from __future__ import annotations

from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings

# ---------------------------------------------------------------------------
# 3.9 — hardcoded relative paths must resolve regardless of process CWD
# ---------------------------------------------------------------------------


def test_runtime_path_resolves_against_project_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.runtime_path.is_absolute()
    assert settings.runtime_path == PROJECT_ROOT / "runtime"
    assert settings.runtime_path != tmp_path / "runtime"


def test_absolute_runtime_dir_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(RUNTIME_DIR=str(tmp_path / "custom-runtime"))

    assert settings.runtime_path == tmp_path / "custom-runtime"


def test_templates_directory_is_absolute_and_exists() -> None:
    from app.web.routes import templates

    directory = Path(templates.env.loader.searchpath[0])

    assert directory.is_absolute()
    assert directory.exists()


def test_static_mount_directory_is_absolute_and_exists() -> None:
    from app.main import app

    static_route = next(route for route in app.routes if getattr(route, "name", None) == "static")
    directory = Path(static_route.app.directory)

    assert directory.is_absolute()
    assert directory.exists()
