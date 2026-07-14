from __future__ import annotations

from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings

# ---------------------------------------------------------------------------
# Task 18 (6.1a) - Audio enhancement settings: defaults, base-dir-relative
# paths, audio_enhance_available() capability helper (deepfilter/rnnoise).
# ---------------------------------------------------------------------------


def test_audio_settings_have_expected_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.deepfilter_binary == "vendor/deepfilternet/deep-filter.exe"
    assert settings.enable_audio_enhance is False
    assert settings.rnnoise_model == "vendor/deepfilternet/models/sh.rnnn"


def test_deepfilter_binary_path_resolves_against_project_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.deepfilter_binary_path.is_absolute()
    assert settings.deepfilter_binary_path == PROJECT_ROOT / "vendor/deepfilternet/deep-filter.exe"
    assert settings.deepfilter_binary_path != tmp_path / "vendor/deepfilternet/deep-filter.exe"


def test_absolute_deepfilter_binary_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(DEEPFILTER_BINARY=str(tmp_path / "custom-deep-filter.exe"))

    assert settings.deepfilter_binary_path == tmp_path / "custom-deep-filter.exe"


def test_rnnoise_model_path_resolves_against_project_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.rnnoise_model_path.is_absolute()
    assert settings.rnnoise_model_path == PROJECT_ROOT / "vendor/deepfilternet/models/sh.rnnn"
    assert settings.rnnoise_model_path != tmp_path / "vendor/deepfilternet/models/sh.rnnn"


def test_absolute_rnnoise_model_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(RNNOISE_MODEL=str(tmp_path / "custom.rnnn"))

    assert settings.rnnoise_model_path == tmp_path / "custom.rnnn"


def _make_fake_deepfilter_binary(tmp_path: Path) -> Path:
    binary = tmp_path / "deep-filter.exe"
    binary.write_bytes(b"fake")
    return binary


def _make_fake_rnnoise_model(tmp_path: Path) -> Path:
    model = tmp_path / "sh.rnnn"
    model.write_bytes(b"fake")
    return model


def test_audio_enhance_available_false_when_deepfilter_binary_missing(tmp_path: Path) -> None:
    settings = Settings(DEEPFILTER_BINARY=str(tmp_path / "missing-deep-filter.exe"))

    assert settings.audio_enhance_available("deepfilter") is False


def test_audio_enhance_available_true_when_deepfilter_binary_exists(tmp_path: Path) -> None:
    binary = _make_fake_deepfilter_binary(tmp_path)

    settings = Settings(DEEPFILTER_BINARY=str(binary))

    assert settings.audio_enhance_available("deepfilter") is True


def test_audio_enhance_available_false_when_rnnoise_model_missing(tmp_path: Path) -> None:
    settings = Settings(RNNOISE_MODEL=str(tmp_path / "missing.rnnn"))

    assert settings.audio_enhance_available("rnnoise") is False


def test_audio_enhance_available_true_when_rnnoise_model_exists(tmp_path: Path) -> None:
    model = _make_fake_rnnoise_model(tmp_path)

    settings = Settings(RNNOISE_MODEL=str(model))

    assert settings.audio_enhance_available("rnnoise") is True


def test_audio_enhance_available_raises_for_unknown_mode() -> None:
    settings = Settings()

    with pytest.raises(ValueError):
        settings.audio_enhance_available("not-a-real-mode")


def test_audio_enhance_available_is_capability_only_and_ignores_enable_flag(
    tmp_path: Path,
) -> None:
    binary = _make_fake_deepfilter_binary(tmp_path)

    settings = Settings(
        ENABLE_AUDIO_ENHANCE=False,
        DEEPFILTER_BINARY=str(binary),
    )

    assert settings.audio_enhance_available("deepfilter") is True
    assert (settings.enable_audio_enhance and settings.audio_enhance_available("deepfilter")) is False
