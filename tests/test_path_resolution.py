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
    # The isolated_runtime_dir autouse fixture (tests/conftest.py) sets RUNTIME_DIR
    # for every test; undo it here to exercise the real, un-overridden default.
    monkeypatch.delenv("RUNTIME_DIR", raising=False)

    settings = Settings()

    assert settings.runtime_path.is_absolute()
    assert settings.runtime_path == PROJECT_ROOT / "runtime"
    assert settings.runtime_path != tmp_path / "runtime"


def test_absolute_runtime_dir_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(RUNTIME_DIR=str(tmp_path / "custom-runtime"))

    assert settings.runtime_path == tmp_path / "custom-runtime"


# ---------------------------------------------------------------------------
# Task 16 review fix — ffmpeg/ffprobe/engine binaries must resolve the same
# way as runtime_path/rife_binary_path. VideoUpscaler used to pass the raw
# forward-slash relative string (settings.ffmpeg_binary / engine_binary)
# straight into asyncio.create_subprocess_exec, which on Windows raises
# "[WinError 2] The system cannot find the file specified" — CreateProcess
# does not resolve a bare CWD-relative forward-slash executable path the way
# a Path()-normalized or absolute one does. Verified against the real
# vendored binaries during the Task 16 smoke test.
# ---------------------------------------------------------------------------


def test_ffmpeg_binary_path_resolves_against_project_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.ffmpeg_binary_path.is_absolute()
    assert settings.ffmpeg_binary_path == PROJECT_ROOT / "vendor/ffmpeg/bin/ffmpeg.exe"
    assert settings.ffmpeg_binary_path != tmp_path / "vendor/ffmpeg/bin/ffmpeg.exe"


def test_absolute_ffmpeg_binary_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(FFMPEG_BINARY=str(tmp_path / "custom-ffmpeg.exe"))

    assert settings.ffmpeg_binary_path == tmp_path / "custom-ffmpeg.exe"


def test_ffprobe_binary_path_resolves_against_project_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.ffprobe_binary_path.is_absolute()
    assert settings.ffprobe_binary_path == PROJECT_ROOT / "vendor/ffmpeg/bin/ffprobe.exe"
    assert settings.ffprobe_binary_path != tmp_path / "vendor/ffmpeg/bin/ffprobe.exe"


def test_engine_binary_path_resolves_against_project_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.engine_binary_path.is_absolute()
    assert settings.engine_binary_path == PROJECT_ROOT / "vendor/realesrgan/realesrgan-ncnn-vulkan.exe"
    assert settings.engine_binary_path != tmp_path / "vendor/realesrgan/realesrgan-ncnn-vulkan.exe"


def test_absolute_engine_binary_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(ENGINE_BINARY=str(tmp_path / "custom-engine.exe"))

    assert settings.engine_binary_path == tmp_path / "custom-engine.exe"


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
