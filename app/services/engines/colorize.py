# Adapted from piddnad/DDColor@2adb63f2656ac41cbdf7b894cddd94121a3faf13 (Apache-2.0, © 2023 Alibaba DAMO Academy): the Lab(L,0,0) gray input and the join of the predicted ab with the photo's L of ddcolor/pipeline.py
from __future__ import annotations

import contextlib
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.services.engines.drunet_restore import CPU_DEVICE, CPU_PRECISION, CpuFallback
from app.services.engines.face_restore import (
    LOW_DISK_WARNING,
    MONOCHROME_TONES,
    WRITE_FAILED_WARNING,
    DiskProbe,
    RecomposeArtifacts,
    disk_free_bytes,
)
from app.services.engines.photo_restore_engine import PhotoRestoreEngine, ReadyInfer, is_gpu_device
from app.services.engines.tiled_restore_runner import (
    TDR_BUDGET_REASON,
    CalibrationCache,
    CalibrationKey,
    CalibrationSpec,
    Clock,
    RestoreCancelled,
    calibrate_tile,
)
from app.services.photo_dsp import BAND_ROWS, neutral_gray
from app.services.photo_restore_presets import ToneKind

COLORIZE_MODEL_ID = "ddcolor-tiny"
RENDER_SIZE = 512
RGB_CHANNELS = 3
AB_CHANNELS = 2
# Medido: con un render de 384 el ab predicho llega a 139,9, fuera de lo que Lab representa bien.
AB_LIMIT = 110.0
MAX_SATURATION = 2.0
FP16 = "fp16"
AB_ARTIFACT_NAME = "ab_512.npy"
NPY_HEADER_BYTES = 128
COLORIZE_NEEDS_MONO_CODE = "restore.error.colorizeNeedsMono"
COLORIZE_NEEDS_MONO_MESSAGE = "Colorize only works on black-and-white or toned photos."
FP16_REJECTED_REASON = "fp16Rejected"
FP16_UNAVAILABLE_REASON = "fp16Unavailable"

# Las mismas constantes de sRGB D65 y Lab que OpenCV, que es el Lab con el que se exporto DDColor.
SRGB_TO_XYZ = np.array(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]],
    dtype=np.float32,
)
XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ.astype(np.float64)).astype(np.float32)
LAB_WHITE = SRGB_TO_XYZ.sum(axis=1)
LAB_A_SCALE = 500.0
LAB_B_SCALE = 200.0
LAB_EPSILON = 0.008856
LAB_LINEAR_SLOPE = 7.787
LAB_F_OFFSET = 16.0 / 116.0
LAB_F_KNEE = LAB_EPSILON ** (1.0 / 3.0)
SRGB_KNEE = 0.04045
LINEAR_KNEE = 0.0031308
SRGB_LOW_SLOPE = 12.92
SRGB_OFFSET = 0.055
SRGB_GAMMA = 2.4


class ColorizeNeedsMonoError(ValueError):
    code = COLORIZE_NEEDS_MONO_CODE

    def __init__(self, tone_kind: str) -> None:
        super().__init__(COLORIZE_NEEDS_MONO_MESSAGE)
        self.tone_kind = tone_kind


@dataclass(frozen=True, slots=True)
class ColorizeOptions:
    strength: float = 1.0
    saturation: float = 1.0

    def __post_init__(self) -> None:
        _require_within(self.strength, 1.0, "Color strength")
        _require_within(self.saturation, MAX_SATURATION, "Saturation")


@dataclass(frozen=True, slots=True)
class ColorizeContext:
    engine: PhotoRestoreEngine
    calibrations: CalibrationCache
    budget_ms: float
    clock: Clock = time.perf_counter
    cancel_event: threading.Event | None = None


@dataclass(frozen=True, slots=True)
class ColorizeModel:
    ready: ReadyInfer
    cpu_fallback: CpuFallback | None = None


@dataclass(frozen=True, slots=True)
class ColorizeResult:
    image: np.ndarray
    ab_512: np.ndarray
    options: ColorizeOptions
    model_id: str
    device: str
    precision: str
    cpu_fallback: CpuFallback | None = None


def colorize(
    context: ColorizeContext,
    image: np.ndarray,
    *,
    tone_kind: ToneKind,
    device: str,
    options: ColorizeOptions | None = None,
    model_id: str = COLORIZE_MODEL_ID,
) -> ColorizeResult:
    require_colorizable(tone_kind)
    _require_rgb_float(image)
    options = options or ColorizeOptions()
    _raise_if_cancelled(context.cancel_event)
    sample = model_input(image)
    model = prepare_colorize_model(context, model_id, device, sample)
    ab_512 = predict_ab(model.ready, sample)
    colored = apply_ab(image, ab_512, options, cancel_event=context.cancel_event)
    return ColorizeResult(
        image=colored,
        ab_512=ab_512,
        options=options,
        model_id=model_id,
        device=model.ready.device,
        precision=model.ready.precision,
        cpu_fallback=model.cpu_fallback,
    )


def require_colorizable(tone_kind: str) -> None:
    if tone_kind not in MONOCHROME_TONES:
        raise ColorizeNeedsMonoError(tone_kind)


def model_input(image: np.ndarray) -> np.ndarray:
    return neutral_gray(cv2.resize(image, (RENDER_SIZE, RENDER_SIZE), interpolation=render_interpolation(image.shape)))


def render_interpolation(shape: tuple[int, ...]) -> int:
    # Achicar promediando evita el aliasing; agrandar queda en bilineal, como el pipeline oficial.
    return cv2.INTER_AREA if min(shape[:2]) >= RENDER_SIZE else cv2.INTER_LINEAR


def predict_ab(ready: ReadyInfer, sample: np.ndarray) -> np.ndarray:
    ab = ready.infer(sample)
    _require_ab_512(ab)
    return clip_ab(ab)


def clip_ab(ab: np.ndarray) -> np.ndarray:
    return np.clip(ab, -AB_LIMIT, AB_LIMIT).astype(np.float32, copy=False)


def upsample_ab(ab_512: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    # El codigo oficial usa nearest; la bicubica baja el ΔE p99 de 15,2 a 10,6 (medido).
    return cv2.resize(ab_512, (width, height), interpolation=cv2.INTER_CUBIC)


def apply_ab(
    image: np.ndarray,
    ab_512: np.ndarray,
    options: ColorizeOptions,
    *,
    cancel_event: threading.Event | None = None,
) -> np.ndarray:
    _require_rgb_float(image)
    _require_ab_512(ab_512)
    ab = upsample_ab(ab_512, image.shape)
    ab *= np.float32(options.saturation)
    output = np.empty_like(image)
    for start in range(0, image.shape[0], BAND_ROWS):
        _raise_if_cancelled(cancel_event)
        rows = slice(start, start + BAND_ROWS)
        output[rows] = colorize_band(image[rows], ab[rows], options.strength)
    return output


def colorize_band(rgb: np.ndarray, ab: np.ndarray, strength: float) -> np.ndarray:
    # Todo en luz lineal con la Y de la foto: el ajuste de gama y la mezcla con el gris no mueven la L.
    luminance = relative_luminance(rgb)
    colored = fit_to_gamut(linear_rgb_from_luminance_ab(luminance, ab), luminance)
    gray = luminance[..., np.newaxis]
    mixed = gray + np.float32(strength) * (colored - gray)
    return linear_to_srgb(mixed)


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    return np.clip(srgb_to_linear(rgb) @ SRGB_TO_XYZ[1], 0.0, 1.0).astype(np.float32, copy=False)


def linear_rgb_from_luminance_ab(luminance: np.ndarray, ab: np.ndarray) -> np.ndarray:
    f_y = lab_f(luminance)
    x = LAB_WHITE[0] * lab_f_inverse(f_y + ab[..., 0] / LAB_A_SCALE)
    z = LAB_WHITE[2] * lab_f_inverse(f_y - ab[..., 1] / LAB_B_SCALE)
    return (np.stack([x, luminance, z], axis=-1) @ XYZ_TO_SRGB.T).astype(np.float32, copy=False)


def fit_to_gamut(linear: np.ndarray, luminance: np.ndarray) -> np.ndarray:
    # Recortar canal por canal cambia la Y; se acerca el color al gris de su misma Y hasta que entra.
    gray = luminance[..., np.newaxis]
    offset = linear - gray
    with np.errstate(divide="ignore", invalid="ignore"):
        room = np.where(offset > 0.0, (1.0 - gray) / offset, np.where(offset < 0.0, -gray / offset, np.inf))
    scale = np.minimum(room.min(axis=-1, keepdims=True), 1.0)
    return np.clip(gray + scale * offset, 0.0, 1.0).astype(np.float32, copy=False)


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    low = values / SRGB_LOW_SLOPE
    high = np.power((np.maximum(values, SRGB_KNEE) + SRGB_OFFSET) / (1.0 + SRGB_OFFSET), SRGB_GAMMA)
    return np.where(values <= SRGB_KNEE, low, high).astype(np.float32, copy=False)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    low = clipped * SRGB_LOW_SLOPE
    high = (1.0 + SRGB_OFFSET) * np.power(np.maximum(clipped, LINEAR_KNEE), 1.0 / SRGB_GAMMA) - SRGB_OFFSET
    return np.where(clipped <= LINEAR_KNEE, low, high).astype(np.float32, copy=False)


def lab_f(t: np.ndarray) -> np.ndarray:
    return np.where(t > LAB_EPSILON, np.cbrt(t), LAB_LINEAR_SLOPE * t + LAB_F_OFFSET).astype(np.float32, copy=False)


def lab_f_inverse(f: np.ndarray) -> np.ndarray:
    linear = (f - LAB_F_OFFSET) / LAB_LINEAR_SLOPE
    return np.where(f > LAB_F_KNEE, f * f * f, linear).astype(np.float32, copy=False)


def prepare_colorize_model(
    context: ColorizeContext, model_id: str, device: str, sample: np.ndarray
) -> ColorizeModel:
    # DDColor solo tiene pesos fp16: el canario se compara contra CPU, que es tambien su vuelta.
    ready = context.engine.ready_infer(model_id, device, sample, reference_device=CPU_DEVICE, clamp=False)
    if not is_gpu_device(device):
        return ColorizeModel(ready)
    reason = gpu_fallback_reason(context, ready)
    if reason is None:
        return ColorizeModel(ready)
    return _cpu_colorize_model(context, model_id, reason)


def gpu_fallback_reason(context: ColorizeContext, ready: ReadyInfer) -> str | None:
    if ready.precision != FP16:
        return _fp16_refusal_reason(context.engine, ready)
    calibration = context.calibrations.get_or_calibrate(
        CalibrationKey(ready.model_id, ready.device, ready.precision),
        lambda: calibrate_tile(
            ready.infer,
            fixed_render_calibration_spec(),
            precision=ready.precision,
            budget_ms=context.budget_ms,
            clock=context.clock,
        ),
    )
    if calibration.tile is not None:
        return None
    return calibration.cpu_fallback_reason or TDR_BUDGET_REASON


def fixed_render_calibration_spec() -> CalibrationSpec:
    return CalibrationSpec(tile_min=RENDER_SIZE, fixed_shape=True, channels=RGB_CHANNELS)


def ab_artifact_bytes() -> int:
    return RENDER_SIZE * RENDER_SIZE * AB_CHANNELS * np.dtype(np.float32).itemsize + NPY_HEADER_BYTES


def save_ab_artifact(
    directory: Path, ab_512: np.ndarray, *, free_bytes: DiskProbe = disk_free_bytes
) -> RecomposeArtifacts:
    _require_ab_512(ab_512)
    free = free_bytes(directory)
    if free is not None and free < ab_artifact_bytes():
        return RecomposeArtifacts(None, LOW_DISK_WARNING)
    try:
        _write_ab(directory, ab_512)
    except OSError:
        _remove_ab(directory)
        return RecomposeArtifacts(None, WRITE_FAILED_WARNING)
    return RecomposeArtifacts(directory)


def load_ab_artifact(directory: Path) -> np.ndarray:
    with (directory / AB_ARTIFACT_NAME).open("rb") as handle:
        ab = np.load(handle, allow_pickle=False)
    _require_ab_512(ab)
    return ab.astype(np.float32, copy=False)


def colorize_metadata(result: ColorizeResult, artifacts: RecomposeArtifacts) -> dict[str, object]:
    return {
        "model": result.model_id,
        "renderSize": RENDER_SIZE,
        "strength": result.options.strength,
        "saturation": result.options.saturation,
        "abReusedOnRecompose": artifacts.available,
    }


def _fp16_refusal_reason(engine: PhotoRestoreEngine, ready: ReadyInfer) -> str:
    rejected = any(
        rejection.model_id == ready.model_id and rejection.device == ready.device
        for rejection in engine.fp16_rejections()
    )
    return FP16_REJECTED_REASON if rejected else FP16_UNAVAILABLE_REASON


def _cpu_colorize_model(context: ColorizeContext, model_id: str, reason: str) -> ColorizeModel:
    context.engine.begin_phase(CPU_DEVICE)
    infer = context.engine.tile_infer(model_id, CPU_DEVICE, CPU_PRECISION, clamp=False)
    return ColorizeModel(ReadyInfer(model_id, CPU_DEVICE, CPU_PRECISION, infer), CpuFallback(model_id, reason))


def _write_ab(directory: Path, ab_512: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / AB_ARTIFACT_NAME).open("wb") as handle:
        np.save(handle, ab_512.astype(np.float32, copy=False), allow_pickle=False)


def _remove_ab(directory: Path) -> None:
    # Solo lo propio: las caras guardan sus recortes en el mismo directorio.
    with contextlib.suppress(OSError):
        (directory / AB_ARTIFACT_NAME).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        directory.rmdir()


def _require_rgb_float(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != RGB_CHANNELS or image.dtype != np.float32:
        raise ValueError(f"Colorize needs a float32 RGB image, got {image.dtype} {image.shape}")


def _require_ab_512(ab: np.ndarray) -> None:
    if ab.shape != (RENDER_SIZE, RENDER_SIZE, AB_CHANNELS):
        raise ValueError(f"Expected ab of shape {(RENDER_SIZE, RENDER_SIZE, AB_CHANNELS)}, got {ab.shape}")


def _require_within(value: float, upper: float, name: str) -> None:
    if not (math.isfinite(value) and 0.0 <= value <= upper):
        raise ValueError(f"{name} must be within [0, {upper}], got {value}")


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RestoreCancelled("Restoration cancelled")
