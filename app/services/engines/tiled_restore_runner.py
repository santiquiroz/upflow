from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.services.engines.onnx_common import blend_tiles_float, detect_scale, tile_starts

TileInfer = Callable[[np.ndarray], np.ndarray]
TileProgress = Callable[[int, int], None]
BlendTile = tuple[int, int, int, int, np.ndarray]


class RestoreCancelled(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Padding:
    bottom: int
    right: int


@dataclass(frozen=True, slots=True)
class TilePlan:
    tile: int
    overlap: int
    multiple: int = 1
    channels: int = 3

    def __post_init__(self) -> None:
        if self.tile <= 0 or self.multiple <= 0 or self.channels <= 0:
            raise ValueError(f"Tile, multiple and channels must be positive: {self}")
        if not 0 <= self.overlap < self.tile:
            raise ValueError(f"Tile overlap must be within [0, tile): {self}")
        if self.tile % self.multiple != 0:
            raise ValueError(f"Tile {self.tile} is not a multiple of {self.multiple}")


def pad_to_requirements(
    arr: np.ndarray, minimum: int, multiple: int, square: bool
) -> tuple[np.ndarray, Padding]:
    if minimum <= 0 or multiple <= 0:
        raise ValueError(f"Padding requirements must be positive (minimum={minimum}, multiple={multiple})")
    height, width = arr.shape[:2]
    target_h = _required_length(height, minimum, multiple)
    target_w = _required_length(width, minimum, multiple)
    if square:
        target_h = target_w = max(target_h, target_w)
    padding = Padding(bottom=target_h - height, right=target_w - width)
    return _reflect_pad(arr, padding), padding


def crop_padding(arr: np.ndarray, padding: Padding, scale: int = 1) -> np.ndarray:
    height, width = arr.shape[:2]
    return arr[: height - padding.bottom * scale, : width - padding.right * scale]


def session_tile_infer(session: Any) -> TileInfer:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Sin traducir errores: la calibracion necesita el texto nativo para
    # distinguir OOM de remocion del device.
    def infer(tile: np.ndarray) -> np.ndarray:
        batch = np.ascontiguousarray(np.transpose(tile, (2, 0, 1))[np.newaxis, ...], dtype=np.float32)
        result = session.run([output_name], {input_name: batch})[0]
        return np.transpose(np.asarray(result[0], dtype=np.float32), (1, 2, 0))

    return infer


def run_tiled(
    infer: TileInfer,
    image: np.ndarray,
    plan: TilePlan,
    *,
    cancel_event: threading.Event | None = None,
    on_progress: TileProgress | None = None,
) -> np.ndarray:
    hwc = image[:, :, np.newaxis] if image.ndim == 2 else image
    passes = channel_passes(hwc, plan.channels)
    counter = _ProgressCounter(len(passes) * _tiles_per_pass(hwc, plan), on_progress)
    outputs = [_run_pass(infer, frame, plan, cancel_event, counter) for frame in passes]
    merged = merge_passes(outputs, hwc.shape[2])
    return merged[:, :, 0] if image.ndim == 2 else merged


def channel_passes(image: np.ndarray, model_channels: int) -> list[np.ndarray]:
    channels = image.shape[2]
    if channels == model_channels:
        return [image]
    if channels == 1 and model_channels == 3:
        return [np.repeat(image, 3, axis=2)]
    if channels == 3 and model_channels == 1:
        return [image[:, :, index : index + 1] for index in range(3)]
    raise ValueError(f"Cannot feed a {channels}-channel image to a {model_channels}-channel model")


def merge_passes(outputs: list[np.ndarray], image_channels: int) -> np.ndarray:
    if len(outputs) > 1:
        return np.concatenate(outputs, axis=2)
    output = outputs[0]
    if image_channels == 1 and output.shape[2] == 3:
        return output.mean(axis=2, keepdims=True, dtype=np.float32)
    return output


class _ProgressCounter:
    def __init__(self, total: int, on_progress: TileProgress | None) -> None:
        self.total = total
        self.done = 0
        self.on_progress = on_progress

    def advance(self) -> None:
        self.done += 1
        if self.on_progress is not None:
            self.on_progress(self.done, self.total)


def _run_pass(
    infer: TileInfer,
    image: np.ndarray,
    plan: TilePlan,
    cancel_event: threading.Event | None,
    counter: _ProgressCounter,
) -> np.ndarray:
    padded, padding = pad_to_requirements(image, minimum=plan.tile, multiple=plan.multiple, square=False)
    tiles = _infer_tiles(infer, padded, plan, cancel_event, counter)
    scale = detect_scale(plan.tile, plan.tile, tiles[0][4])
    height, width = padded.shape[:2]
    blended = blend_tiles_float(tiles, height, width, tiles[0][4].shape[2], scale, feather=scale * plan.overlap)
    return crop_padding(blended, padding, scale)


def _infer_tiles(
    infer: TileInfer,
    padded: np.ndarray,
    plan: TilePlan,
    cancel_event: threading.Event | None,
    counter: _ProgressCounter,
) -> list[BlendTile]:
    tiles: list[BlendTile] = []
    for y0, x0 in _tile_origins(padded.shape[0], padded.shape[1], plan):
        _raise_if_cancelled(cancel_event)
        source = padded[y0 : y0 + plan.tile, x0 : x0 + plan.tile]
        tiles.append((y0, x0, plan.tile, plan.tile, infer(source)))
        counter.advance()
    return tiles


def _tile_origins(height: int, width: int, plan: TilePlan) -> list[tuple[int, int]]:
    starts_y = tile_starts(height, plan.tile, plan.overlap)
    starts_x = tile_starts(width, plan.tile, plan.overlap)
    return [(y0, x0) for y0 in starts_y for x0 in starts_x]


def _tiles_per_pass(image: np.ndarray, plan: TilePlan) -> int:
    height = _required_length(image.shape[0], plan.tile, plan.multiple)
    width = _required_length(image.shape[1], plan.tile, plan.multiple)
    return len(_tile_origins(height, width, plan))


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RestoreCancelled("Restoration cancelled")


def _required_length(length: int, minimum: int, multiple: int) -> int:
    at_least = max(length, minimum)
    return -(-at_least // multiple) * multiple


def _reflect_pad(arr: np.ndarray, padding: Padding) -> np.ndarray:
    if padding.bottom == 0 and padding.right == 0:
        return arr
    widths = [(0, padding.bottom), (0, padding.right)] + [(0, 0)] * (arr.ndim - 2)
    return np.pad(arr, widths, mode="reflect")
