from __future__ import annotations

from typing import Callable

import numpy as np

RestoreMonoFn = Callable[[np.ndarray], np.ndarray]


def restore_multichannel(audio: np.ndarray, restore_mono: RestoreMonoFn) -> np.ndarray:
    """audio: (samples, channels) float32. Restaura preservando imagen espacial.

    1 canal: pasa directo por restore_mono.
    2 canales: decodifica Mid/Side, restaura solo Mid, side queda intacto.
    Otros layouts: ver multichannel_layouts.py (Task 7).
    """
    channels = audio.shape[1]
    if channels == 1:
        restored = restore_mono(audio[:, 0])
        return _rms_match(restored, audio[:, 0]).reshape(-1, 1)
    if channels == 2:
        return _restore_stereo_mid_side(audio, restore_mono)
    raise NotImplementedError(f"{channels}-channel audio requires multichannel_layouts (Task 7)")


def _restore_stereo_mid_side(audio: np.ndarray, restore_mono: RestoreMonoFn) -> np.ndarray:
    left, right = audio[:, 0], audio[:, 1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    restored_mid = restore_mono(mid)
    restored_mid = _rms_match(restored_mid, mid)
    left_out = restored_mid + side
    right_out = restored_mid - side
    return np.stack([left_out, right_out], axis=1).astype(np.float32)


def _rms_match(restored: np.ndarray, original: np.ndarray) -> np.ndarray:
    original_rms = np.sqrt(np.mean(original.astype(np.float64) ** 2))
    restored_rms = np.sqrt(np.mean(restored.astype(np.float64) ** 2))
    if restored_rms < 1e-9 or original_rms < 1e-9:
        return restored
    return (restored * (original_rms / restored_rms)).astype(np.float32)
