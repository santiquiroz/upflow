"""Mediciones de spike_restore_dml.py fuera de la tabla de fotos: el grafo uint8 de video
de DRUNet (cuadros enteros de CCTV, con el mismo FrameInference y la misma regla TDR del
carril IA) y los detectores que corren en CPU por diseño, medidos en DML solo como dato."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.engines.frame_model_runner import (
    FrameInference,
    OrtValueFactory,
    level_for_strength,
    pad_frame,
    to_uint8,
    whole_frame_ms,
)
from app.services.engines.tiled_restore_runner import (
    TARGET_BUDGET_FRACTION,
    CalibrationSpec,
    Clock,
    TileCalibration,
    TileInfer,
    calibrate_tile,
)

U8_INPUT = "tensor(uint8)"
FRAME_INPUT = "input"
STRENGTH_INPUT = "strength"
U8_PEAK = 255.0
NHWC_RANK = 4
RGB = 3
# Los dos cuadros del smoke del carril IA (§4.15 y P3-GPU-1): 1080p y la mitad de un 4K lado a lado.
VIDEO_FRAME_SHAPES: tuple[tuple[int, int], ...] = ((1080, 1920), (1080, 960))
VIDEO_STRENGTH_PERCENT = 50
FRAME_CALLS = 5
# Lado corto 256 y múltiplo de 16 (BOPBTL, §3.4.2); lado mayor 1280 (RetinaFace, §3.4.7). Foto 4:3.
DETECTOR_SHAPES: Mapping[str, tuple[int, int, int, int]] = {
    "bopbtl-scratch-detector": (1, 1, 256, 336),
    "retinaface-r34": (1, 3, 960, 1280),
}
RETINAFACE_MEAN_BGR = (104.0, 117.0, 123.0)
DETECTOR_CALLS = 5


@dataclass(frozen=True, slots=True)
class FrameTiming:
    shape: tuple[int, int]
    padded: tuple[int, int]
    predicted_ms: float
    # None = el runner arranca con el cuadro entero; si no, el tile con el que arranca.
    start_tile: int | None
    median_ms: float | None


@dataclass(frozen=True, slots=True)
class DetectorProbe:
    model_id: str
    file_name: str
    sha256: str
    input_shape: tuple[int, ...]
    providers: tuple[str, ...] = ()
    cpu_nodes: tuple[str, ...] = ()
    median_ms: float | None = None
    max_abs: Mapping[str, float] | None = None
    non_finite: bool = False
    error: str | None = None


# --- grafo uint8 de video ------------------------------------------------------


def is_video_contract(inputs: Sequence[Any]) -> bool:
    by_name = {item.name: item for item in inputs}
    frame = by_name.get(FRAME_INPUT)
    if frame is None or STRENGTH_INPUT not in by_name or len(by_name) != 2:
        return False
    shape = list(frame.shape)
    return frame.type == U8_INPUT and len(shape) == NHWC_RANK and shape[-1] == RGB


def video_level() -> float:
    return level_for_strength(VIDEO_STRENGTH_PERCENT)


def u8_canary(canary: np.ndarray) -> np.ndarray:
    return to_uint8(canary[:, :, :RGB] * U8_PEAK)


def frame_inference(session: Any, device: str, ortvalue_factory: OrtValueFactory) -> FrameInference:
    return FrameInference(session, video_level(), device, ortvalue_factory)


def canary_output(inference: FrameInference, canary_u8: np.ndarray) -> np.ndarray:
    # En [0, 1] como el canario del runner: sobre 0..255, psnr_db recortaria todo a 1.
    return inference(np.ascontiguousarray(canary_u8[np.newaxis]))[0].astype(np.float32) / np.float32(U8_PEAK)


def probe_infer(inference: FrameInference) -> TileInfer:
    return lambda tile: inference(np.ascontiguousarray(to_uint8(tile * U8_PEAK)[np.newaxis]))[0]


def calibrate_video(
    inference: FrameInference, spec: CalibrationSpec, precision: str, budget_ms: float, clock: Clock
) -> TileCalibration:
    return calibrate_tile(probe_infer(inference), spec, precision=precision, budget_ms=budget_ms, clock=clock)


def median_tile_ms(inference: FrameInference, tile: int | None, clock: Clock) -> float | None:
    if tile is None:
        return None
    batch = np.full((1, tile, tile, RGB), 128, dtype=np.uint8)
    return _median_ms(lambda: inference(batch), FRAME_CALLS, clock)


def padded_shape(shape: tuple[int, int]) -> tuple[int, int]:
    return pad_frame(np.zeros((*shape, RGB), dtype=np.uint8))[0].shape[:2]


def frame_timing(
    inference: FrameInference, calibration: TileCalibration, shape: tuple[int, int], budget_ms: float, clock: Clock
) -> FrameTiming:
    padded = padded_shape(shape)
    predicted = whole_frame_ms(calibration.ms_per_mpx, padded)
    # La misma regla que tdr_start_tile: solo se llama con el cuadro entero si la prediccion entra.
    if predicted > budget_ms * TARGET_BUDGET_FRACTION:
        return FrameTiming(shape, padded, predicted, calibration.tile, None)
    batch = _frame_batch(padded)
    inference(batch)
    return FrameTiming(shape, padded, predicted, None, _median_ms(lambda: inference(batch), FRAME_CALLS, clock))


def _frame_batch(padded: tuple[int, int]) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(padded[0]), np.arange(padded[1]), indexing="ij")
    ramp = ((xs + ys) % 256).astype(np.uint8)
    return np.ascontiguousarray(np.stack([ramp, ramp[::-1], ramp[:, ::-1]], axis=-1)[np.newaxis])


def frame_issues(timings: Sequence[FrameTiming], budget_ms: float) -> tuple[str, ...]:
    return tuple(issue for timing in timings if (issue := _frame_issue(timing, budget_ms)) is not None)


def _frame_issue(timing: FrameTiming, budget_ms: float) -> str | None:
    target_ms = budget_ms * TARGET_BUDGET_FRACTION
    label = f"cuadro {timing.shape[1]}x{timing.shape[0]}"
    if timing.start_tile is None and timing.median_ms is None:
        return f"{label}: ni el tile mínimo entra en el presupuesto"
    if timing.median_ms is not None and timing.median_ms >= target_ms:
        return f"{label}: mediana {timing.median_ms:.0f} ms > {target_ms:.0f} ms"
    return None


def iobinding_issue(device: str, inference: FrameInference, cpu_device: str) -> str | None:
    if device == cpu_device or inference.uses_iobinding:
        return None
    return "IOBinding cayó a run común"


def frame_rows(model_id: str, precision: str, timings: Sequence[FrameTiming], io_binding: bool) -> list[tuple[str, ...]]:
    return [_frame_row(model_id, precision, timing, io_binding) for timing in timings]


def _frame_row(model_id: str, precision: str, timing: FrameTiming, io_binding: bool) -> tuple[str, ...]:
    start = "cuadro entero" if timing.start_tile is None else f"tiles de {timing.start_tile}"
    if timing.start_tile is None and timing.median_ms is None:
        start = "no arranca (TdrBudgetExceeded)"
    return (
        model_id,
        precision,
        f"{timing.shape[1]}x{timing.shape[0]}",
        f"{timing.padded[1]}x{timing.padded[0]}",
        f"{timing.predicted_ms:.0f}",
        "n/d" if timing.median_ms is None else f"{timing.median_ms:.0f}",
        start,
        "sí" if io_binding else "no",
    )


FRAME_COLUMNS = ("Modelo", "Precisión", "Cuadro", "Con padding", "Previsto ms", "Mediana ms", "Arranque", "IOBinding")


# --- detectores en CPU por diseño, medidos en DML como dato --------------------


def detector_feed(model_id: str, input_name: str, shape: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
    _, channels, height, width = shape
    ys, xs = np.meshgrid(np.linspace(0.0, 1.0, height), np.linspace(0.0, 1.0, width), indexing="ij")
    plane = (0.5 + 0.35 * np.sin(6.0 * xs) * np.cos(4.0 * ys)).astype(np.float32)
    image = np.stack([plane] * channels)[np.newaxis]
    if model_id.startswith("retinaface"):
        image = image * np.float32(U8_PEAK) - np.array(RETINAFACE_MEAN_BGR, dtype=np.float32).reshape(1, 3, 1, 1)
    return {input_name: np.ascontiguousarray(image, dtype=np.float32)}


def run_outputs(session: Any, feed: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = [item.name for item in session.get_outputs()]
    return dict(zip(names, session.run(names, dict(feed)), strict=True))


def output_max_abs(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {
        name: float(np.max(np.abs(np.asarray(candidate[name], np.float64) - np.asarray(ref, np.float64))))
        for name, ref in reference.items()
    }


def probe_detector(
    model_id: str,
    reference_session: Any,
    device_session: Any,
    cpu_nodes: tuple[str, ...],
    clock: Clock,
) -> dict[str, Any]:
    shape = DETECTOR_SHAPES[model_id]
    feed = detector_feed(model_id, reference_session.get_inputs()[0].name, shape)
    reference = run_outputs(reference_session, feed)
    run_outputs(device_session, feed)
    median = _median_ms(lambda: device_session.run(None, feed), DETECTOR_CALLS, clock)
    outputs = run_outputs(device_session, feed)
    return {
        "input_shape": shape,
        "providers": tuple(device_session.get_providers()),
        "cpu_nodes": cpu_nodes,
        "median_ms": median,
        "max_abs": output_max_abs(reference, outputs),
        "non_finite": not all(np.isfinite(value).all() for value in outputs.values()),
    }


DETECTOR_COLUMNS = ("Modelo", "Archivo", "SHA-256", "Entrada", "Providers", "Nodos en CPU", "Mediana ms", "máx. |Δ| vs CPU", "NaN/Inf")


def detector_row(probe: DetectorProbe) -> tuple[str, ...]:
    if probe.error is not None:
        return (probe.model_id, probe.file_name, probe.sha256, "x".join(map(str, probe.input_shape)),
                "n/d", "n/d", "n/d", f"error: {probe.error}", "n/d")
    nodes = str(len(probe.cpu_nodes)) if not probe.cpu_nodes else f"{len(probe.cpu_nodes)}: {', '.join(sorted(set(probe.cpu_nodes)))}"
    deltas = ", ".join(f"{name} {value:.2e}" for name, value in (probe.max_abs or {}).items())
    return (
        probe.model_id,
        probe.file_name,
        probe.sha256,
        "x".join(map(str, probe.input_shape)),
        ", ".join(probe.providers),
        nodes,
        "n/d" if probe.median_ms is None else f"{probe.median_ms:.0f}",
        deltas or "n/d",
        "sí" if probe.non_finite else "no",
    )


def _median_ms(call: Callable[[], Any], calls: int, clock: Clock) -> float:
    return statistics.median(_timed_ms(call, clock) for _ in range(calls))


def _timed_ms(call: Callable[[], Any], clock: Clock) -> float:
    started = clock()
    call()
    return (clock() - started) * 1000.0
