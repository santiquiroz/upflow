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


def test_engine_models_path_resolves_against_project_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.engine_models_path.is_absolute()
    assert settings.engine_models_path == PROJECT_ROOT / "vendor/realesrgan/models"
    assert settings.engine_models_path != tmp_path / "vendor/realesrgan/models"


def test_absolute_engine_models_dir_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(ENGINE_MODELS_DIR=str(tmp_path / "custom-models"))

    assert settings.engine_models_path == tmp_path / "custom-models"


# ---------------------------------------------------------------------------
# SP1 Task 2 — models_path derives from runtime_path (not project root) so
# it always follows an overridden RUNTIME_DIR, mirroring uploads/outputs/temp.
# ---------------------------------------------------------------------------


def test_models_path_defaults_to_models_dir_under_runtime(tmp_path: Path) -> None:
    settings = Settings(RUNTIME_DIR=str(tmp_path / "runtime"))

    assert settings.models_path == tmp_path / "runtime" / "models"


def test_models_path_follows_overridden_runtime_dir(tmp_path: Path) -> None:
    settings = Settings(RUNTIME_DIR=str(tmp_path / "custom-runtime"))

    assert settings.models_path == tmp_path / "custom-runtime" / "models"


def test_absolute_models_dir_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(RUNTIME_DIR=str(tmp_path / "runtime"), MODELS_DIR=str(tmp_path / "custom-models"))

    assert settings.models_path == tmp_path / "custom-models"


# ---------------------------------------------------------------------------
# Task 16 review round 2 — the consumers themselves must hold resolved
# absolute paths. Path(settings.ffmpeg_binary) normalizes slashes (enough to
# dodge WinError 2 from the project root) but stays CWD-relative: launched
# from any other directory (packaged launcher, Windows service), ffprobe and
# the realesrgan engine would fail to find their binaries.
# ---------------------------------------------------------------------------


def test_media_tools_uses_resolved_absolute_ffmpeg_and_ffprobe_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.services.media_tools import MediaTools

    monkeypatch.chdir(tmp_path)
    settings = Settings()

    media_tools = MediaTools(settings)

    assert media_tools.ffmpeg_path.is_absolute()
    assert media_tools.ffmpeg_path == settings.ffmpeg_binary_path
    assert media_tools.ffprobe_path.is_absolute()
    assert media_tools.ffprobe_path == settings.ffprobe_binary_path


def test_realesrgan_engine_uses_resolved_absolute_binary_and_models_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine

    monkeypatch.chdir(tmp_path)
    settings = Settings()

    engine = RealEsrganNcnnEngine(settings)

    assert engine.binary_path.is_absolute()
    assert engine.binary_path == settings.engine_binary_path
    assert engine.models_dir.is_absolute()
    assert engine.models_dir == settings.engine_models_path


def test_frontend_dist_dir_is_absolute_and_resolved_against_project_root() -> None:
    from app.main import APP_DIR, FRONTEND_DIST_DIR

    assert FRONTEND_DIST_DIR.is_absolute()
    assert FRONTEND_DIST_DIR == APP_DIR.parent / "frontend" / "dist"
