from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings, get_settings
from app.services import restore_models
from app.services.restore_models import RestoreArtifact, RestoreBundle

RESTORE_DEFAULTS = {
    "RESTORE_PHOTO_ENABLED": ("restore_photo_enabled", False),
    "CCTV_AI_ENABLED": ("cctv_ai_enabled", False),
    "RESTORE_MODEL_DIR": ("restore_model_dir", "vendor/restore"),
    "RESTORE_CALL_BUDGET_MS": ("restore_call_budget_ms", 1200),
    "RESTORE_GPU_THROTTLE_SECONDS": ("restore_gpu_throttle_seconds", 0.0),
    "RESTORE_SESSION_CACHE_MB": ("restore_session_cache_mb", 3000),
    "RESTORE_MAX_LIVE_SESSIONS": ("restore_max_live_sessions", 3),
    "RESTORE_NCNN_HEADROOM_MB": ("restore_ncnn_headroom_mb", 2048),
    "RESTORE_MAX_INPUT_PIXELS": ("restore_max_input_pixels", 40_000_000),
    "RESTORE_MAX_OUTPUT_PIXELS": ("restore_max_output_pixels", 100_000_000),
    "RESTORE_ANALYSIS_CONCURRENCY": ("restore_analysis_concurrency", 1),
    "CCTV_X264_THREADS": ("cctv_x264_threads", 4),
    "CCTV_FFV1_SLICES": ("cctv_ffv1_slices", 4),
    "CCTV_MAX_STILL_FRAMES": ("cctv_max_still_frames", 20),
    "CCTV_ROI_MAX_FRAMES": ("cctv_roi_max_frames", 60),
    "CCTV_ROI_ECC_MIN": ("cctv_roi_ecc_min", 0.8),
}

INSTALLED_PROPERTIES = {
    "core": "restore_core_installed",
    "faces": "restore_faces_installed",
    "colorize": "restore_colorize_installed",
}


def published_bundle(name: str) -> RestoreBundle:
    artifact = RestoreArtifact(
        model_id=f"{name}-model",
        precision="fp32",
        filename=f"{name}-model.onnx",
        url=f"https://example.com/{name}-model.onnx",
        sha256="d" * 64,
        size=4,
    )
    return RestoreBundle(name, (artifact,))


def install(model_dir: Path, bundle: RestoreBundle) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    for artifact in bundle.artifacts:
        (model_dir / artifact.filename).write_bytes(b"x" * artifact.size)
    manifest = model_dir / f"{bundle.name}.installed.json"
    manifest.write_text(json.dumps({"bundle": bundle.name}), encoding="utf-8")
    return manifest


@pytest.mark.parametrize("variable, expected", RESTORE_DEFAULTS.items())
def test_restore_and_cctv_settings_have_the_spec_defaults(
    variable: str, expected: tuple[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(variable, raising=False)
    attribute, value = expected

    assert getattr(Settings(_env_file=None), attribute) == value


def test_restore_model_dir_resolves_against_the_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RESTORE_MODEL_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    assert Settings(_env_file=None).restore_model_dir_path == PROJECT_ROOT / "vendor/restore"


def test_an_absolute_restore_model_dir_is_kept(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, RESTORE_MODEL_DIR=str(tmp_path / "packs"))

    assert settings.restore_model_dir_path == tmp_path / "packs"


def test_tests_never_see_the_real_restore_folder() -> None:
    model_dir = get_settings().restore_model_dir_path

    assert model_dir != PROJECT_ROOT / "vendor/restore"
    assert not model_dir.exists()


@pytest.mark.parametrize("attribute", INSTALLED_PROPERTIES.values())
def test_nothing_is_installed_in_an_isolated_run(attribute: str) -> None:
    assert getattr(get_settings(), attribute) == ""


@pytest.mark.parametrize("bundle_name, attribute", INSTALLED_PROPERTIES.items())
def test_installed_property_returns_the_manifest_of_a_complete_bundle(
    bundle_name: str, attribute: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = published_bundle(bundle_name)
    monkeypatch.setitem(restore_models.RESTORE_BUNDLES, bundle_name, bundle)
    manifest = install(tmp_path / "restore", bundle)

    settings = Settings(_env_file=None, RESTORE_MODEL_DIR=str(tmp_path / "restore"))

    assert getattr(settings, attribute) == str(manifest)


def test_installed_property_is_empty_when_a_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = published_bundle("core")
    monkeypatch.setitem(restore_models.RESTORE_BUNDLES, "core", bundle)
    install(tmp_path / "restore", bundle)
    (tmp_path / "restore" / "core-model.onnx").unlink()

    settings = Settings(_env_file=None, RESTORE_MODEL_DIR=str(tmp_path / "restore"))

    assert settings.restore_core_installed == ""


@pytest.mark.parametrize(
    "variable, value",
    [
        ("RESTORE_CALL_BUDGET_MS", 0),
        ("RESTORE_SESSION_CACHE_MB", 0),
        ("RESTORE_MAX_LIVE_SESSIONS", 0),
        ("RESTORE_MAX_INPUT_PIXELS", 0),
        ("RESTORE_MAX_OUTPUT_PIXELS", 0),
        ("RESTORE_ANALYSIS_CONCURRENCY", 0),
        ("CCTV_X264_THREADS", 0),
        ("CCTV_FFV1_SLICES", 0),
        ("CCTV_MAX_STILL_FRAMES", 0),
        ("CCTV_ROI_MAX_FRAMES", 0),
        ("CCTV_ROI_ECC_MIN", 0.0),
        ("CCTV_ROI_ECC_MIN", 1.0),
        ("RESTORE_GPU_THROTTLE_SECONDS", -0.1),
        ("RESTORE_NCNN_HEADROOM_MB", -1),
    ],
)
def test_restore_and_cctv_settings_reject_impossible_values(variable: str, value: float) -> None:
    with pytest.raises(ValueError, match=variable):
        Settings(_env_file=None, **{variable: value})


def test_zero_throttle_and_zero_headroom_are_allowed() -> None:
    settings = Settings(_env_file=None, RESTORE_GPU_THROTTLE_SECONDS=0, RESTORE_NCNN_HEADROOM_MB=0)

    assert settings.restore_gpu_throttle_seconds == 0
    assert settings.restore_ncnn_headroom_mb == 0


def test_env_example_documents_every_restore_and_cctv_variable() -> None:
    text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#?\s*([A-Z0-9_]+)=", text, flags=re.MULTILINE))

    assert set(RESTORE_DEFAULTS) <= documented
