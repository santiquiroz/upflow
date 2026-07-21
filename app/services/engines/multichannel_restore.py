from __future__ import annotations

import logging
from typing import Callable

import numpy as np

RestoreMonoFn = Callable[[np.ndarray], np.ndarray]

logger = logging.getLogger(__name__)


def restore_multichannel(audio: np.ndarray, restore_mono: RestoreMonoFn) -> np.ndarray:
    """audio: (samples, channels) float32. Restaura preservando imagen espacial.

    1 canal: pasa directo por restore_mono.
    2 canales: decodifica Mid/Side, restaura solo Mid, side queda intacto.
    6/8 canales: 5.1/7.1 via multichannel_layouts (frente+rears por par, centro
    directo, LFE intacto).
    Otros conteos: fallback a downmix mono con warning explicito (nunca silencioso).
    """
    channels = audio.shape[1]
    if channels == 1:
        restored = restore_mono(audio[:, 0])
        return _rms_match(restored, audio[:, 0]).reshape(-1, 1)
    if channels == 2:
        return _restore_stereo_mid_side(audio, restore_mono)
    if channels == 6:
        from app.services.engines.multichannel_layouts import restore_surround

        return restore_surround(audio, "5.1", restore_mono)
    if channels == 8:
        from app.services.engines.multichannel_layouts import restore_surround

        return restore_surround(audio, "7.1", restore_mono)
    return _restore_unknown_layout_as_mono(audio, restore_mono, channels)


def _restore_unknown_layout_as_mono(
    audio: np.ndarray, restore_mono: RestoreMonoFn, channels: int
) -> np.ndarray:
    logger.warning(
        "Unrecognized channel layout (%d channels); falling back to mono restoration", channels
    )
    mono = audio.mean(axis=1)
    restored = _rms_match(restore_mono(mono), mono)
    return np.tile(restored.reshape(-1, 1), (1, channels)).astype(np.float32)


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
