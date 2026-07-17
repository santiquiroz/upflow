from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.config import (
    UPSCALE_BACKEND_AUTO,
    UPSCALE_BACKEND_NCNN,
    UPSCALE_BACKEND_ONNX,
    UPSCALE_BACKENDS,
)

# ---------------------------------------------------------------------------
# SP11 - upscale runtime registry + Auto-selection.
#
# Two runtimes upscale a BUILTIN Real-ESRGAN model's video frames:
#   - ncnn: Real-ESRGAN NCNN Vulkan (the existing, safe fallback, any GPU).
#   - onnx: the optimized ONNX Runtime engine (uint8 graph, whole-frame, IO
#     binding, threaded pipeline) -- ~2.1x faster on AMD/DirectML, but only
#     when a vendored uint8 ONNX export of that model exists AND a GPU EP is
#     present.
#
# resolve_upscale_backend is a PURE function of its inputs so the routing
# rule is unit-testable without a GPU, onnxruntime, or the model files. It
# only ever chooses the RUNTIME; the model the user picked is unchanged.
# ---------------------------------------------------------------------------

_CPU_DEVICE = "cpu"


class UpscaleBackend(str, Enum):
    ncnn = "ncnn"
    onnx = "onnx"


@dataclass(frozen=True, slots=True)
class BuiltinOnnxModel:
    engine_model_name: str
    filename: str
    scale: int


# Builtin Real-ESRGAN models that have a uint8-in/out ONNX export. Keyed by the
# concrete engine model name (the scale-specific one the job carries in
# job.model_name after VideoJobManager resolution), so a builtin model resolves
# straight to its vendored .onnx file + upscale ratio.
BUILTIN_ONNX_MODELS: dict[str, BuiltinOnnxModel] = {
    "realesr-animevideov3-x2": BuiltinOnnxModel(
        "realesr-animevideov3-x2", "realesr-animevideov3-x2-uint8.onnx", 2
    ),
    "realesr-animevideov3-x3": BuiltinOnnxModel(
        "realesr-animevideov3-x3", "realesr-animevideov3-x3-uint8.onnx", 3
    ),
    "realesr-animevideov3-x4": BuiltinOnnxModel(
        "realesr-animevideov3-x4", "realesr-animevideov3-x4-uint8.onnx", 4
    ),
    "realesrgan-x4plus": BuiltinOnnxModel(
        "realesrgan-x4plus", "realesrgan-x4plus-uint8.onnx", 4
    ),
    "realesrgan-x4plus-anime": BuiltinOnnxModel(
        "realesrgan-x4plus-anime", "realesrgan-x4plus-anime-uint8.onnx", 4
    ),
}


def get_builtin_onnx_model(engine_model_name: str) -> BuiltinOnnxModel | None:
    return BUILTIN_ONNX_MODELS.get(engine_model_name)


def validate_backend_choice(value: str | None) -> str | None:
    """Rejects an out-of-range per-job backend override (None = no override)."""
    if value is None:
        return None
    if value not in UPSCALE_BACKENDS:
        raise ValueError(f"backend must be one of {sorted(UPSCALE_BACKENDS)}")
    return value


def resolve_upscale_backend(
    *,
    setting_backend: str,
    job_backend: str | None,
    onnx_model_available: bool,
    gpu_ep_available: bool,
    device: str,
) -> UpscaleBackend:
    """Resolves which runtime upscales a builtin model's frames.

    Precedence: an explicit per-job `job_backend` (ncnn/onnx) wins over the
    global `setting_backend`; either being `auto` defers to the Auto rule.

    Auto rule: onnx iff the model has a vendored ONNX export available AND the
    target can run it fast (a GPU execution provider is present, or the device
    is CPU where ncnn has no path at all); otherwise ncnn, the safe fallback.
    """
    choice = job_backend or setting_backend or UPSCALE_BACKEND_AUTO
    if choice == UPSCALE_BACKEND_NCNN:
        return UpscaleBackend.ncnn
    if choice == UPSCALE_BACKEND_ONNX:
        return UpscaleBackend.onnx
    return _auto_backend(onnx_model_available, gpu_ep_available, device)


def _auto_backend(onnx_model_available: bool, gpu_ep_available: bool, device: str) -> UpscaleBackend:
    if not onnx_model_available:
        return UpscaleBackend.ncnn
    if device == _CPU_DEVICE:
        # ncnn Vulkan has no CPU path; onnx-cpu is the only runtime that fits.
        # NOTE: today VideoJobManager rejects device="cpu" for builtin models
        # (pre-SP11 rule), so this branch is only reachable once that validation
        # is relaxed to allow onnx-cpu for builtin models with an ONNX export.
        # The rule is kept correct here so enabling that is a one-line change.
        return UpscaleBackend.onnx
    if gpu_ep_available:
        return UpscaleBackend.onnx
    return UpscaleBackend.ncnn
