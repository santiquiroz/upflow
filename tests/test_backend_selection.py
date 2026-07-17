from __future__ import annotations

import pytest

from app.config import (
    UPSCALE_BACKEND_AUTO,
    UPSCALE_BACKEND_NCNN,
    UPSCALE_BACKEND_ONNX,
    Settings,
)
from app.services.backend_registry import (
    BUILTIN_ONNX_MODELS,
    UpscaleBackend,
    get_builtin_onnx_model,
    resolve_upscale_backend,
    validate_backend_choice,
)

# ---------------------------------------------------------------------------
# SP11 Task 2 - backend registry + Auto-selection rule. Pure function, no GPU
# or onnxruntime needed. The rule: onnx iff the model has a vendored ONNX
# export AND (a GPU EP is present OR device is cpu); else ncnn (safe fallback).
# ---------------------------------------------------------------------------


def resolve(
    *,
    setting_backend: str = UPSCALE_BACKEND_AUTO,
    job_backend: str | None = None,
    onnx_model_available: bool = True,
    gpu_ep_available: bool = True,
    device: str = "dml:0",
) -> UpscaleBackend:
    return resolve_upscale_backend(
        setting_backend=setting_backend,
        job_backend=job_backend,
        onnx_model_available=onnx_model_available,
        gpu_ep_available=gpu_ep_available,
        device=device,
    )


# --- Auto rule ---


def test_auto_picks_onnx_when_model_available_and_gpu_ep_present() -> None:
    assert resolve(onnx_model_available=True, gpu_ep_available=True, device="dml:0") == UpscaleBackend.onnx


def test_auto_falls_back_to_ncnn_when_no_onnx_model() -> None:
    assert resolve(onnx_model_available=False, gpu_ep_available=True, device="dml:0") == UpscaleBackend.ncnn


def test_auto_falls_back_to_ncnn_when_no_gpu_ep_on_gpu_device() -> None:
    assert resolve(onnx_model_available=True, gpu_ep_available=False, device="dml:0") == UpscaleBackend.ncnn


def test_auto_picks_onnx_on_cpu_device_when_model_available() -> None:
    # ncnn Vulkan has no cpu path; onnx-cpu is the only runtime, even with no GPU EP.
    assert resolve(onnx_model_available=True, gpu_ep_available=False, device="cpu") == UpscaleBackend.onnx


def test_auto_stays_ncnn_on_cpu_device_when_no_onnx_model() -> None:
    assert resolve(onnx_model_available=False, gpu_ep_available=False, device="cpu") == UpscaleBackend.ncnn


# --- Forced setting (global UPSCALE_BACKEND) ---


def test_setting_ncnn_forces_ncnn_even_when_onnx_available() -> None:
    assert resolve(setting_backend=UPSCALE_BACKEND_NCNN, onnx_model_available=True) == UpscaleBackend.ncnn


def test_setting_onnx_forces_onnx() -> None:
    assert resolve(setting_backend=UPSCALE_BACKEND_ONNX, gpu_ep_available=False) == UpscaleBackend.onnx


# --- Per-job override wins over the global setting ---


def test_job_override_onnx_beats_setting_ncnn() -> None:
    assert resolve(setting_backend=UPSCALE_BACKEND_NCNN, job_backend=UPSCALE_BACKEND_ONNX) == UpscaleBackend.onnx


def test_job_override_ncnn_beats_setting_onnx() -> None:
    assert resolve(setting_backend=UPSCALE_BACKEND_ONNX, job_backend=UPSCALE_BACKEND_NCNN) == UpscaleBackend.ncnn


def test_job_override_auto_defers_to_auto_rule() -> None:
    assert (
        resolve(setting_backend=UPSCALE_BACKEND_NCNN, job_backend=UPSCALE_BACKEND_AUTO, onnx_model_available=True)
        == UpscaleBackend.onnx
    )


def test_no_job_override_uses_setting() -> None:
    assert resolve(setting_backend=UPSCALE_BACKEND_ONNX, job_backend=None) == UpscaleBackend.onnx


# --- validate_backend_choice ---


def test_validate_backend_choice_accepts_none() -> None:
    assert validate_backend_choice(None) is None


@pytest.mark.parametrize("value", [UPSCALE_BACKEND_AUTO, UPSCALE_BACKEND_NCNN, UPSCALE_BACKEND_ONNX])
def test_validate_backend_choice_accepts_valid(value: str) -> None:
    assert validate_backend_choice(value) == value


def test_validate_backend_choice_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        validate_backend_choice("cuda")


# --- builtin onnx model catalog ---


def test_builtin_onnx_catalog_covers_all_builtin_engine_names() -> None:
    expected = {
        "realesr-animevideov3-x2",
        "realesr-animevideov3-x3",
        "realesr-animevideov3-x4",
        "realesrgan-x4plus",
        "realesrgan-x4plus-anime",
    }
    assert set(BUILTIN_ONNX_MODELS.keys()) == expected


def test_get_builtin_onnx_model_returns_scale_and_filename() -> None:
    model = get_builtin_onnx_model("realesr-animevideov3-x3")
    assert model is not None
    assert model.scale == 3
    assert model.filename.endswith(".onnx")


def test_get_builtin_onnx_model_unknown_is_none() -> None:
    assert get_builtin_onnx_model("does-not-exist") is None


# --- config validation ---


def test_settings_rejects_invalid_upscale_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="UPSCALE_BACKEND must be one of"):
        Settings(_env_file=None, RUNTIME_DIR=str(tmp_path), UPSCALE_BACKEND="cuda")


def test_settings_defaults_upscale_backend_to_auto(tmp_path) -> None:
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path))
    assert settings.upscale_backend == UPSCALE_BACKEND_AUTO
