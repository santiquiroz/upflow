from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from app.services.engines.photo_restore_engine import InferFactory, PhotoRestoreEngine, ReadyInfer, is_gpu_device
from app.services.engines.tiled_restore_runner import (
    TDR_BUDGET_REASON,
    CalibrationCache,
    CalibrationKey,
    CalibrationSpec,
    Clock,
    TileCalibration,
    TileInfer,
    TilePlan,
    TileProgress,
    calibrate_tile,
    channel_passes,
    pad_to_requirements,
    run_tiled,
    session_tile_infer,
)
from app.services.photo_dsp import keep_grain
from app.services.restore_models import RestoreModelSpec

DEBLOCK_MODEL_ID = "drunet-deblock-color"
DENOISE_MODEL_ID = "drunet-color"
CPU_DEVICE = "cpu"
CPU_PRECISION = "fp32"
DRUNET_MULTIPLE = 8
DRUNET_CHANNELS = 3
# Escalas confirmadas en el export (M-02, DPIR): deblock = (100 - calidad JPEG)/100 entrenado
# para calidad 95..10; denoise = sigma/255 con sigma acotado a [2, 50] (§3.4.4).
DEBLOCK_LEVEL_RANGE = (0.05, 0.90)
DENOISE_SIGMA_RANGE = (2.0, 50.0)
U8_PEAK = 255.0


@dataclass(frozen=True, slots=True)
class CpuFallback:
    model_id: str
    reason: str

    def to_metadata(self) -> dict[str, str]:
        return {"model": self.model_id, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class DrunetResult:
    image: np.ndarray
    model_id: str
    device: str
    precision: str
    tile: int
    cpu_fallback: CpuFallback | None = None


@dataclass(frozen=True, slots=True)
class DrunetContext:
    engine: PhotoRestoreEngine
    calibrations: CalibrationCache
    budget_ms: float
    clock: Clock = time.perf_counter
    cancel_event: threading.Event | None = None
    on_progress: TileProgress | None = None


def deblock_level(strength: float) -> float:
    _require_unit_interval(strength, "Strength")
    low, high = DEBLOCK_LEVEL_RANGE
    return low + strength * (high - low)


def denoise_sigma(strength: float, estimated_sigma: float) -> float:
    _require_unit_interval(strength, "Strength")
    if not estimated_sigma >= 0.0:
        raise ValueError(f"Estimated noise sigma must be non-negative, got {estimated_sigma}")
    return float(np.clip(strength * estimated_sigma * U8_PEAK, *DENOISE_SIGMA_RANGE))


def denoise_level(strength: float, estimated_sigma: float) -> float:
    return denoise_sigma(strength, estimated_sigma) / U8_PEAK


def with_strength_channel(tile: np.ndarray, level: float) -> np.ndarray:
    strength = np.full((*tile.shape[:2], 1), level, dtype=np.float32)
    return np.concatenate([tile.astype(np.float32, copy=False), strength], axis=2)


def strength_infer_factory(level: float, base: InferFactory = session_tile_infer) -> InferFactory:
    def factory(session: object) -> TileInfer:
        infer = base(session)
        return lambda tile: infer(with_strength_channel(tile, level))

    return factory


def canary_sample(image: np.ndarray, tile: int) -> np.ndarray:
    hwc = image[:, :, np.newaxis] if image.ndim == 2 else image
    color = channel_passes(hwc, DRUNET_CHANNELS)[0]
    top, left = (max(0, (size - tile) // 2) for size in color.shape[:2])
    crop = color[top : top + tile, left : left + tile]
    padded, _ = pad_to_requirements(crop, minimum=tile, multiple=DRUNET_MULTIPLE, square=False)
    return np.ascontiguousarray(padded[:tile, :tile], dtype=np.float32)


def deblock_photo(context: DrunetContext, image: np.ndarray, strength: float, device: str) -> DrunetResult:
    return run_drunet(context, DEBLOCK_MODEL_ID, image, deblock_level(strength), device)


def denoise_photo(
    context: DrunetContext,
    image: np.ndarray,
    strength: float,
    estimated_sigma: float,
    device: str,
    *,
    keep_grain_amount: float = 0.0,
) -> DrunetResult:
    result = run_drunet(context, DENOISE_MODEL_ID, image, denoise_level(strength, estimated_sigma), device)
    if keep_grain_amount == 0.0:
        return result
    return replace(result, image=keep_grain(image.astype(np.float32, copy=False), result.image, keep_grain_amount))


def run_drunet(context: DrunetContext, model_id: str, image: np.ndarray, level: float, device: str) -> DrunetResult:
    _require_unit_interval(level, "Strength map level")
    spec = context.engine.model_spec(model_id)
    factory = strength_infer_factory(level)
    ready = _ready_infer(context, spec, image, device, factory)
    if not is_gpu_device(device):
        return _run(context, spec, ready, image, cpu_tile(spec))
    calibration = _calibration(context, spec, ready)
    if calibration.tile is None:
        return _run_cpu_fallback(context, spec, image, factory, calibration)
    return _run(context, spec, ready, image, calibration.tile)


def cpu_tile(spec: RestoreModelSpec) -> int:
    # Sin TDR en CPU: el tile mas grande de la escalera, sin el techo medido en GPU.
    return CalibrationSpec(spec.tile_min, spec.tile_candidates, channels=DRUNET_CHANNELS).ladder()[-1]


def calibration_spec(spec: RestoreModelSpec, precision: str) -> CalibrationSpec:
    return CalibrationSpec(
        tile_min=spec.tile_min,
        tile_candidates=spec.tile_candidates,
        ceiling=spec.tile_by_precision.get(precision),
        channels=DRUNET_CHANNELS,
    )


def reference_device(calibrations: CalibrationCache, model_id: str, device: str) -> str:
    # El fp32 del device todavia sin tile calibrado podria pasarse del presupuesto: la referencia va a CPU.
    calibrated = calibrations.get(CalibrationKey(model_id, device, "fp32")) is not None
    return device if calibrated else CPU_DEVICE


def _ready_infer(
    context: DrunetContext, spec: RestoreModelSpec, image: np.ndarray, device: str, factory: InferFactory
) -> ReadyInfer:
    return context.engine.ready_infer(
        spec.id,
        device,
        canary_sample(image, spec.tile_min),
        reference_device=reference_device(context.calibrations, spec.id, device),
        infer_for=factory,
    )


def _calibration(context: DrunetContext, spec: RestoreModelSpec, ready: ReadyInfer) -> TileCalibration:
    key = CalibrationKey(spec.id, ready.device, ready.precision)
    return context.calibrations.get_or_calibrate(
        key,
        lambda: calibrate_tile(
            ready.infer,
            calibration_spec(spec, ready.precision),
            precision=ready.precision,
            budget_ms=context.budget_ms,
            clock=context.clock,
        ),
    )


def _run_cpu_fallback(
    context: DrunetContext,
    spec: RestoreModelSpec,
    image: np.ndarray,
    factory: InferFactory,
    calibration: TileCalibration,
) -> DrunetResult:
    context.engine.begin_phase(CPU_DEVICE)
    infer = context.engine.tile_infer(spec.id, CPU_DEVICE, CPU_PRECISION, infer_for=factory)
    ready = ReadyInfer(spec.id, CPU_DEVICE, CPU_PRECISION, infer)
    result = _run(context, spec, ready, image, cpu_tile(spec))
    reason = calibration.cpu_fallback_reason or TDR_BUDGET_REASON
    return replace(result, cpu_fallback=CpuFallback(spec.id, reason))


def _run(
    context: DrunetContext, spec: RestoreModelSpec, ready: ReadyInfer, image: np.ndarray, tile: int
) -> DrunetResult:
    plan = TilePlan(tile=tile, overlap=spec.overlap, multiple=DRUNET_MULTIPLE, channels=DRUNET_CHANNELS)
    restored = run_tiled(
        ready.infer, image, plan, cancel_event=context.cancel_event, on_progress=context.on_progress
    )
    return DrunetResult(restored, spec.id, ready.device, ready.precision, tile)


def _require_unit_interval(value: float, name: str) -> None:
    if not (math.isfinite(value) and 0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be within [0, 1], got {value}")
