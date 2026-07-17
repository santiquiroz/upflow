from __future__ import annotations

from pathlib import Path

import pytest

from app.config import AUDIO_RESTORE_MODES, PROJECT_ROOT, Settings

# ---------------------------------------------------------------------------
# SP9 - Apollo audio restoration settings: defaults, project-root-relative
# model path, and the enable+installed audio_restore_available() gate.
# ---------------------------------------------------------------------------


def test_audio_restore_settings_have_expected_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.enable_audio_restore is False
    assert settings.apollo_restore_model == "vendor/apollo/apollo.onnx"
    assert settings.audio_restore_chunk_seconds == 1.0  # <2s por chunk -> evita el TDR de Windows en DirectML
    assert settings.max_audio_upload_mb == 200


def test_audio_restore_modes_constant() -> None:
    assert AUDIO_RESTORE_MODES == frozenset({"apollo"})


def test_apollo_restore_model_path_resolves_against_project_root_not_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings()

    assert settings.apollo_restore_model_path.is_absolute()
    assert settings.apollo_restore_model_path == PROJECT_ROOT / "vendor/apollo/apollo.onnx"


def test_absolute_apollo_model_override_is_kept_as_is(tmp_path: Path) -> None:
    settings = Settings(APOLLO_RESTORE_MODEL=str(tmp_path / "custom-apollo.onnx"))

    assert settings.apollo_restore_model_path == tmp_path / "custom-apollo.onnx"


def test_chunk_seconds_must_be_positive() -> None:
    with pytest.raises(ValueError):
        Settings(AUDIO_RESTORE_CHUNK_SECONDS=0)


def test_audio_restore_available_false_when_disabled(tmp_path: Path) -> None:
    model = tmp_path / "apollo.onnx"
    model.write_bytes(b"fake")

    settings = Settings(ENABLE_AUDIO_RESTORE=False, APOLLO_RESTORE_MODEL=str(model))

    assert settings.audio_restore_available() is False


def test_audio_restore_available_false_when_model_missing(tmp_path: Path) -> None:
    settings = Settings(ENABLE_AUDIO_RESTORE=True, APOLLO_RESTORE_MODEL=str(tmp_path / "missing.onnx"))

    # Never raises even though the model is absent -- the app must not break.
    assert settings.audio_restore_available() is False


def test_audio_restore_available_true_when_enabled_and_installed(tmp_path: Path) -> None:
    model = tmp_path / "apollo.onnx"
    model.write_bytes(b"fake")

    settings = Settings(ENABLE_AUDIO_RESTORE=True, APOLLO_RESTORE_MODEL=str(model))

    assert settings.audio_restore_available() is True
