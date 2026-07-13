from __future__ import annotations

from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings

# ---------------------------------------------------------------------------
# Task 10 (4.2) - RIFE settings: defaults, base-dir-relative paths,
# ALLOWED_FPS_MULTIPLIERS parsing, interpolation_available() helper.
# ---------------------------------------------------------------------------


def test_rife_settings_have_expected_defaults() -> None:
    settings = Settings()

    assert settings.rife_binary == "vendor/rife/rife-ncnn-vulkan.exe"
    assert settings.rife_models_dir == "vendor/rife/models"
    assert settings.rife_model == "rife-v4.6"
    assert settings.enable_interpolation is False
    assert settings.allowed_fps_multipliers == "2,3,4"


def test_rife_binary_path_resolves_against_project_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.rife_binary_path.is_absolute()
    assert settings.rife_binary_path == PROJECT_ROOT / "vendor/rife/rife-ncnn-vulkan.exe"
    assert settings.rife_binary_path != tmp_path / "vendor/rife/rife-ncnn-vulkan.exe"


def test_absolute_rife_binary_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(RIFE_BINARY=str(tmp_path / "custom-rife.exe"))

    assert settings.rife_binary_path == tmp_path / "custom-rife.exe"


def test_rife_models_path_resolves_against_project_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.rife_models_path.is_absolute()
    assert settings.rife_models_path == PROJECT_ROOT / "vendor/rife/models"
    assert settings.rife_models_path != tmp_path / "vendor/rife/models"


def test_absolute_rife_models_dir_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(RIFE_MODELS_DIR=str(tmp_path / "custom-models"))

    assert settings.rife_models_path == tmp_path / "custom-models"


def test_allowed_fps_multiplier_values_parses_default_csv() -> None:
    settings = Settings()

    assert settings.allowed_fps_multiplier_values == [2, 3, 4]


def test_allowed_fps_multiplier_values_parses_custom_csv() -> None:
    settings = Settings(ALLOWED_FPS_MULTIPLIERS="2,4")

    assert settings.allowed_fps_multiplier_values == [2, 4]


def test_interpolation_available_false_when_disabled(tmp_path: Path) -> None:
    binary = tmp_path / "rife-ncnn-vulkan.exe"
    binary.write_bytes(b"fake")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    settings = Settings(
        ENABLE_INTERPOLATION=False,
        RIFE_BINARY=str(binary),
        RIFE_MODELS_DIR=str(models_dir),
    )

    assert settings.interpolation_available() is False


def test_interpolation_available_false_when_binary_missing(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    settings = Settings(
        ENABLE_INTERPOLATION=True,
        RIFE_BINARY=str(tmp_path / "missing-rife.exe"),
        RIFE_MODELS_DIR=str(models_dir),
    )

    assert settings.interpolation_available() is False


def test_interpolation_available_false_when_models_dir_missing(tmp_path: Path) -> None:
    binary = tmp_path / "rife-ncnn-vulkan.exe"
    binary.write_bytes(b"fake")

    settings = Settings(
        ENABLE_INTERPOLATION=True,
        RIFE_BINARY=str(binary),
        RIFE_MODELS_DIR=str(tmp_path / "missing-models"),
    )

    assert settings.interpolation_available() is False


def test_interpolation_available_true_when_enabled_and_paths_exist(tmp_path: Path) -> None:
    binary = tmp_path / "rife-ncnn-vulkan.exe"
    binary.write_bytes(b"fake")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    settings = Settings(
        ENABLE_INTERPOLATION=True,
        RIFE_BINARY=str(binary),
        RIFE_MODELS_DIR=str(models_dir),
    )

    assert settings.interpolation_available() is True
