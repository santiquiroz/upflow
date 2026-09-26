from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.engines.frame_model_runner import (
    FrameModelReport,
    as_frame,
    build_frame_model_runner,
    level_for_strength,
)
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.inpaint_mask import soft_composite

Box = tuple[int, int, int, int]
FrameFn = Callable[[np.ndarray], np.ndarray]
UpscalerFactory = Callable[[], FrameFn]
RestoreReporter = Callable[[], FrameModelReport]
RunnerBuilder = Callable[..., Any]

OSD_OPAQUE = 255
OSD_EDGE_ALPHA = 128
FRAME_DIGEST_BYTES = 16


@dataclass(frozen=True, slots=True)
class ComposedStageReport:
    frames: int
    duplicates_reused: int
    upscaled: bool
    osd_boxes: int
    restore: FrameModelReport | None


def validated_box(box: Sequence[int]) -> Box:
    x, y, w, h = (int(value) for value in box)
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"OSD box {tuple(box)} needs a non-negative origin and a positive size")
    return x, y, w, h


def validated_boxes(boxes: Sequence[Sequence[int]]) -> tuple[Box, ...]:
    return tuple(validated_box(box) for box in boxes)


def require_box_inside(box: Box, frame_hw: tuple[int, int]) -> Box:
    x, y, w, h = box
    height, width = frame_hw
    if x + w > width or y + h > height:
        raise ValueError(f"OSD box {box} is outside the {width}x{height} frame")
    return box


def output_factor(source_hw: tuple[int, int], output_hw: tuple[int, int]) -> int:
    factor = output_hw[0] // source_hw[0]
    if factor < 1 or output_hw != (source_hw[0] * factor, source_hw[1] * factor):
        raise ValueError(f"Output {output_hw} is not an integer scale of the decoded frame {source_hw}")
    return factor


def enlarge_nearest(image: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return image
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def osd_edge_mask(height: int, width: int) -> np.ndarray:
    mask = np.full((height, width), OSD_EDGE_ALPHA, dtype=np.uint8)
    mask[1:-1, 1:-1] = OSD_OPAQUE
    return mask


def paste_osd(processed: np.ndarray, decoded: np.ndarray, boxes: tuple[Box, ...]) -> np.ndarray:
    if not boxes:
        return processed
    source = as_frame(decoded)
    # Copia propia: la salida del escalador puede ser un buffer de su anillo de readback.
    target = as_frame(processed).copy()
    factor = output_factor(source.shape[:2], target.shape[:2])
    for box in boxes:
        _paste_box_into(target, source, require_box_inside(box, source.shape[:2]), factor)
    return target[np.newaxis]


def _paste_box_into(target: np.ndarray, source: np.ndarray, box: Box, factor: int) -> None:
    x, y, w, h = box
    patch = enlarge_nearest(source[y : y + h, x : x + w], factor)
    rows = slice(y * factor, (y + h) * factor)
    columns = slice(x * factor, (x + w) * factor)
    target[rows, columns] = soft_composite(patch, target[rows, columns], osd_edge_mask(*patch.shape[:2]))


def frame_digest(frame: np.ndarray) -> bytes:
    digest = hashlib.blake2b(repr(frame.shape).encode(), digest_size=FRAME_DIGEST_BYTES)
    digest.update(np.ascontiguousarray(frame).data)
    return digest.digest()


class ComposedStage:
    def __init__(
        self,
        restore: FrameFn,
        upscale: FrameFn | None,
        osd_boxes: tuple[Box, ...],
        describe_restore: RestoreReporter | None,
    ) -> None:
        self._restore = restore
        self._upscale = upscale
        self._osd_boxes = osd_boxes
        self._describe_restore = describe_restore
        self._last_digest: bytes | None = None
        self._last_output: np.ndarray | None = None
        self._frames = 0
        self._duplicates_reused = 0

    def process(self, frame: np.ndarray) -> list[np.ndarray]:
        return [self._reused_or_composed(frame)]

    def flush(self) -> list[np.ndarray]:
        return []

    def report(self) -> ComposedStageReport:
        restore = None if self._describe_restore is None else self._describe_restore()
        return ComposedStageReport(
            frames=self._frames,
            duplicates_reused=self._duplicates_reused,
            upscaled=self._upscale is not None,
            osd_boxes=len(self._osd_boxes),
            restore=restore,
        )

    def _reused_or_composed(self, frame: np.ndarray) -> np.ndarray:
        digest = frame_digest(frame)
        if digest == self._last_digest:
            self._duplicates_reused += 1
        else:
            self._last_output = self._compose(frame)
            self._last_digest = digest
        self._frames += 1
        return self._last_output

    def _compose(self, frame: np.ndarray) -> np.ndarray:
        restored = self._restore(frame)
        enhanced = restored if self._upscale is None else self._upscale(restored)
        return paste_osd(enhanced, frame, self._osd_boxes)


def build_composed_stage(
    restore: FrameFn,
    upscale: FrameFn | None = None,
    *,
    osd_boxes: Sequence[Sequence[int]] = (),
    describe_restore: RestoreReporter | None = None,
) -> ComposedStage:
    return ComposedStage(restore, upscale, validated_boxes(osd_boxes), describe_restore)


class FrameRestorer:
    def __init__(
        self, engine: PhotoRestoreEngine, *, runner_builder: RunnerBuilder = build_frame_model_runner
    ) -> None:
        self._engine = engine
        self._runner_builder = runner_builder

    def build_stage(
        self,
        device: str,
        sample_frame: np.ndarray,
        strength_percent: int,
        *,
        upscaler_factory: UpscalerFactory | None = None,
        osd_boxes: Sequence[Sequence[int]] = (),
        cancel_event: threading.Event | None = None,
    ) -> ComposedStage:
        level = level_for_strength(strength_percent)
        boxes = validated_boxes(osd_boxes)
        runner = self._runner_builder(self._engine, device, sample_frame, level, cancel_event=cancel_event)
        # El escalador toma el device despues: el runner ya corrio su canario y retiene su sesion.
        upscale = None if upscaler_factory is None else upscaler_factory()
        return build_composed_stage(runner, upscale, osd_boxes=boxes, describe_restore=runner.report)
