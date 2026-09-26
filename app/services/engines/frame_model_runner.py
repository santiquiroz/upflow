from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.dml_device import try_parse_dml_device_id
from app.services.engines.drunet_restore import deblock_level
from app.services.engines.onnx_video_upscaler import is_device_removed_error, is_oom_error
from app.services.engines.photo_restore_engine import InferFactory, PhotoRestoreEngine
from app.services.engines.tiled_restore_runner import (
    CalibrationSpec,
    Padding,
    RestoreCancelled,
    TileInfer,
    TilePlan,
    crop_padding,
    pad_to_requirements,
    run_tiled,
)
from app.services.restore_models import RestoreModelSpec

logger = logging.getLogger(__name__)

VIDEO_DEBLOCK_MODEL_ID = "drunet-deblock-color-u8"
# Contrato del export (M-02): uint8 NHWC 1xHxWx3 + escalar float32 "strength" -> uint8 NHWC.
FRAME_INPUT = "input"
STRENGTH_INPUT = "strength"
FRAME_OUTPUT = "output"
FRAME_MULTIPLE = 8
FRAME_CHANNELS = 3
STRENGTH_PERCENT_MAX = 100
CPU_DEVICE = "cpu"
DML_DEVICE_TYPE = "dml"
U8_PEAK = 255.0

OrtValueFactory = Callable[[np.ndarray, str, int], Any]
RemovalGuard = Callable[[], AbstractContextManager[None]]
Batch = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class FrameModelReport:
    model_id: str
    device: str
    precision: str
    tile: int | None
    io_binding: bool


def level_for_strength(strength_percent: int) -> float:
    if not 0 <= strength_percent <= STRENGTH_PERCENT_MAX:
        raise ValueError(f"Strength must be within [0, {STRENGTH_PERCENT_MAX}], got {strength_percent}")
    return deblock_level(strength_percent / STRENGTH_PERCENT_MAX)


def dml_ortvalue(array: np.ndarray, device_type: str, device_id: int) -> Any:
    import onnxruntime as ort

    return ort.OrtValue.ortvalue_from_numpy(array, device_type, device_id)


def as_frame(frame: np.ndarray) -> np.ndarray:
    image = frame[0] if frame.ndim == 4 and frame.shape[0] == 1 else frame
    if image.ndim != 3 or image.shape[2] != FRAME_CHANNELS or image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 RGB frame (1xHxWx3 or HxWx3), got {frame.dtype} {frame.shape}")
    return image


def pad_frame(image: np.ndarray) -> tuple[np.ndarray, Padding]:
    return pad_to_requirements(image, minimum=FRAME_MULTIPLE, multiple=FRAME_MULTIPLE, square=False)


def canary_crop(image: np.ndarray, side: int) -> np.ndarray:
    top, left = (max(0, (size - side) // 2) for size in image.shape[:2])
    crop = image[top : top + side, left : left + side]
    padded, _ = pad_to_requirements(crop, minimum=side, multiple=FRAME_MULTIPLE, square=False)
    return np.ascontiguousarray(padded[:side, :side])


def oom_tile_ladder(spec: RestoreModelSpec, precision: str, padded_shape: tuple[int, int]) -> tuple[int, ...]:
    ladder = CalibrationSpec(
        spec.tile_min, spec.tile_candidates, ceiling=spec.tile_by_precision.get(precision)
    ).ladder()
    # Un tile tan grande como el cuadro repite la misma asignacion que acaba de fallar.
    longest = max(padded_shape)
    return tuple(tile for tile in reversed(ladder) if tile % FRAME_MULTIPLE == 0 and tile < longest)


def strength_feed(level: float) -> np.ndarray:
    return np.array(level, dtype=np.float32)


def plain_run(session: Any, batch: np.ndarray, level: float) -> np.ndarray:
    return session.run([FRAME_OUTPUT], {FRAME_INPUT: batch, STRENGTH_INPUT: strength_feed(level)})[0]


def iobinding_run(
    session: Any, batch: np.ndarray, level: float, device_id: int, ortvalue_factory: OrtValueFactory
) -> np.ndarray:
    binding = session.io_binding()
    binding.bind_ortvalue_input(FRAME_INPUT, ortvalue_factory(batch, DML_DEVICE_TYPE, device_id))
    binding.bind_cpu_input(STRENGTH_INPUT, strength_feed(level))
    binding.bind_output(FRAME_OUTPUT, DML_DEVICE_TYPE, device_id)
    session.run_with_iobinding(binding)
    return binding.copy_outputs_to_cpu()[0]


def canary_infer_factory(level: float) -> InferFactory:
    # El canario puntua en [0, 1] como el de fotos: sobre 0..255, psnr_db recortaria todo a 1.
    def factory(session: Any) -> TileInfer:
        return lambda tile: plain_run(session, np.ascontiguousarray(tile[np.newaxis]), level)[0] / np.float32(U8_PEAK)

    return factory


def require_video_contract(session: Any) -> None:
    inputs = sorted(item.name for item in session.get_inputs())
    outputs = [item.name for item in session.get_outputs()]
    if inputs != sorted((FRAME_INPUT, STRENGTH_INPUT)) or outputs != [FRAME_OUTPUT]:
        raise RuntimeError(
            f"Video model must take {FRAME_INPUT!r} and {STRENGTH_INPUT!r} and return {FRAME_OUTPUT!r}; "
            f"got inputs {inputs} and outputs {outputs}"
        )


def require_output_shape(output: np.ndarray, batch: np.ndarray) -> np.ndarray:
    if output.shape != batch.shape:
        raise RuntimeError(f"Video model returned shape {output.shape} for a {batch.shape} frame")
    return np.asarray(output, dtype=np.uint8)


def to_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


class FrameInference:
    def __init__(self, session: Any, level: float, device: str, ortvalue_factory: OrtValueFactory) -> None:
        self._session = session
        self._level = level
        self._device_id = try_parse_dml_device_id(device)
        self._ortvalue_factory = ortvalue_factory
        self.uses_iobinding = self._device_id is not None and hasattr(session, "io_binding")

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if self.uses_iobinding:
            bound = self._bound_run(batch)
            if bound is not None:
                return require_output_shape(bound, batch)
        return require_output_shape(plain_run(self._session, batch, self._level), batch)

    def _bound_run(self, batch: np.ndarray) -> np.ndarray | None:
        try:
            return iobinding_run(self._session, batch, self._level, self._device_id, self._ortvalue_factory)
        except Exception as exc:
            if is_oom_error(exc) or is_device_removed_error(exc):
                raise
            # Sticky: si falla una vez fallaria en cada cuadro, y el aviso se da una sola vez.
            self.uses_iobinding = False
            logger.warning("IO binding failed on the video model; using plain runs for this job", exc_info=True)
            return None


class FrameModelRunner:
    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        precision: str,
        frame_shape: tuple[int, int],
        inference: FrameInference,
        tile_ladder: tuple[int, ...],
        overlap: int,
        removal_guard: RemovalGuard,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._precision = precision
        self._frame_shape = frame_shape
        self._inference = inference
        self._tile_ladder = tile_ladder
        self._overlap = overlap
        self._removal_guard = removal_guard
        self._cancel_event = cancel_event
        self._tile_index: int | None = None

    def __call__(self, frame_nhwc: np.ndarray) -> np.ndarray:
        self._raise_if_cancelled()
        padded, padding = pad_frame(self._checked(frame_nhwc))
        with self._removal_guard():
            restored = self._restore(np.ascontiguousarray(padded))
        return np.ascontiguousarray(crop_padding(restored, padding)[np.newaxis])

    def report(self) -> FrameModelReport:
        tile = None if self._tile_index is None else self._tile_ladder[self._tile_index]
        return FrameModelReport(self._model_id, self._device, self._precision, tile, self._inference.uses_iobinding)

    def _checked(self, frame_nhwc: np.ndarray) -> np.ndarray:
        image = as_frame(frame_nhwc)
        if image.shape[:2] != self._frame_shape:
            raise ValueError(f"Frame shape {image.shape[:2]} differs from the job's fixed shape {self._frame_shape}")
        return image

    def _restore(self, padded: np.ndarray) -> np.ndarray:
        if self._tile_index is None:
            try:
                return self._inference(padded[np.newaxis])[0]
            except Exception as exc:
                self._step_down_on_oom(exc)
        return self._tiled(padded)

    def _tiled(self, padded: np.ndarray) -> np.ndarray:
        while True:
            try:
                return self._run_tiles(padded, self._tile_ladder[self._tile_index])
            except Exception as exc:
                self._step_down_on_oom(exc)

    def _run_tiles(self, padded: np.ndarray, tile: int) -> np.ndarray:
        plan = TilePlan(tile=tile, overlap=self._overlap, multiple=FRAME_MULTIPLE, channels=FRAME_CHANNELS)
        infer = self._tile_infer()
        return to_uint8(run_tiled(infer, padded, plan, cancel_event=self._cancel_event))

    def _tile_infer(self) -> TileInfer:
        return lambda tile: self._inference(np.ascontiguousarray(tile[np.newaxis]))[0].astype(np.float32)

    def _step_down_on_oom(self, exc: Exception) -> None:
        next_index = 0 if self._tile_index is None else self._tile_index + 1
        if not is_oom_error(exc) or next_index >= len(self._tile_ladder):
            raise exc
        # Sticky para el resto del job, como el motor de video: reintentar el cuadro entero volveria a fallar.
        logger.warning(
            "Video model hit an out-of-memory error on %s; using %d px tiles for the rest of the job",
            self._device,
            self._tile_ladder[next_index],
        )
        self._tile_index = next_index

    def _raise_if_cancelled(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise RestoreCancelled("Restoration cancelled")


def build_frame_model_runner(
    engine: PhotoRestoreEngine,
    device: str,
    sample_frame: np.ndarray,
    level: float,
    *,
    model_id: str = VIDEO_DEBLOCK_MODEL_ID,
    reference_device: str = CPU_DEVICE,
    ortvalue_factory: OrtValueFactory = dml_ortvalue,
    cancel_event: threading.Event | None = None,
) -> FrameModelRunner:
    _require_level(level)
    image = as_frame(sample_frame)
    spec = engine.model_spec(model_id)
    engine.begin_phase(device)
    precision = engine.precision_for(
        model_id,
        device,
        canary_crop(image, spec.tile_min),
        reference_device=reference_device,
        infer_for=canary_infer_factory(level),
    )
    # La sesion se toma una vez: el escalador de la etapa compuesta es otro dueno del mismo device.
    session = engine.session(model_id, device, precision)
    require_video_contract(session)
    padded_shape = pad_frame(image)[0].shape[:2]
    return FrameModelRunner(
        model_id=model_id,
        device=device,
        precision=precision,
        frame_shape=image.shape[:2],
        inference=FrameInference(session, level, device, ortvalue_factory),
        tile_ladder=oom_tile_ladder(spec, precision, padded_shape),
        overlap=spec.overlap,
        removal_guard=lambda: engine.removal_classified(device),
        cancel_event=cancel_event,
    )


def _require_level(level: float) -> None:
    if not (math.isfinite(level) and 0.0 <= level <= 1.0):
        raise ValueError(f"Strength map level must be within [0, 1], got {level}")
