"""Overlapping-chunk scheduling, mirroring MSST's `utils/model_utils.demix`.

Reimplemented in numpy for the `batch_size = 1` path, which is the one the
reference itself annotates as correct ("using clone() fixes the clicks at chunk
edges when using batch_size=1") and the only one a fixed-batch ONNX graph can run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


def windowing_array(chunk_size: int, fade_size: int) -> np.ndarray:
    """Linear fade-in / fade-out, ones in between (MSST `_getWindowingArray`)."""
    window = np.ones(chunk_size, dtype=np.float64)
    window[-fade_size:] = np.linspace(1, 0, fade_size)
    window[:fade_size] = np.linspace(0, 1, fade_size)
    return window


@dataclass(frozen=True)
class ChunkPlan:
    """Everything the overlap-add loop needs, derived once from the mix length."""

    chunk_size: int
    step: int
    fade_size: int
    border: int
    padded_length: int
    original_length: int
    was_padded: bool

    @property
    def num_chunks(self) -> int:
        return len(range(0, self.padded_length, self.step))


def plan_chunks(length: int, chunk_size: int, num_overlap: int) -> ChunkPlan:
    fade_size = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    was_padded = length > 2 * border and border > 0
    padded_length = length + 2 * border if was_padded else length
    return ChunkPlan(chunk_size, step, fade_size, border, padded_length, length, was_padded)


def pad_mix(mix: np.ndarray, plan: ChunkPlan) -> np.ndarray:
    if not plan.was_padded:
        return mix
    return np.pad(mix, ((0, 0), (plan.border, plan.border)), mode="reflect")


def _chunk_window(plan: ChunkPlan, start: int, next_start: int, total: int) -> np.ndarray:
    window = windowing_array(plan.chunk_size, plan.fade_size)
    if start == 0:
        window[: plan.fade_size] = 1.0
    elif next_start >= total:
        window[-plan.fade_size :] = 1.0
    return window


def iter_chunks(mix: np.ndarray, plan: ChunkPlan) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield (start, seg_len, padded_chunk [C, chunk_size], window [chunk_size])."""
    total = mix.shape[1]
    for start in range(0, total, plan.step):
        part = mix[:, start : start + plan.chunk_size]
        seg_len = part.shape[1]
        pad_mode = "reflect" if seg_len > plan.chunk_size // 2 else "constant"
        if seg_len < plan.chunk_size:
            part = np.pad(part, ((0, 0), (0, plan.chunk_size - seg_len)), mode=pad_mode)
        yield start, seg_len, part, _chunk_window(plan, start, start + plan.step, total)


def unpad_result(result: np.ndarray, counter: np.ndarray, plan: ChunkPlan) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        out = result / counter
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if plan.was_padded:
        out = out[..., plan.border : -plan.border]
    return out
