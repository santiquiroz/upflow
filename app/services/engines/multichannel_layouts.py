from __future__ import annotations

import numpy as np

from app.services.engines.multichannel_restore import RestoreMonoFn, _rms_match

# Orden estandar ffmpeg: FL FR FC LFE BL BR [SL SR]
_LAYOUT_PAIRS = {
    "5.1": {"front": (0, 1), "center": 2, "lfe": 3, "rear": (4, 5)},
    "7.1": {"front": (0, 1), "center": 2, "lfe": 3, "rear": (4, 5), "side": (6, 7)},
}


def _restore_pair_mid_side(
    left: np.ndarray, right: np.ndarray, restore_mono: RestoreMonoFn
) -> tuple[np.ndarray, np.ndarray]:
    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    restored_mid = _rms_match(restore_mono(mid), mid)
    return restored_mid + side, restored_mid - side


def restore_surround(audio: np.ndarray, layout: str, restore_mono: RestoreMonoFn) -> np.ndarray:
    spec = _LAYOUT_PAIRS.get(layout)
    if spec is None:
        raise ValueError(f"Unsupported surround layout: {layout!r}")

    out = audio.copy()
    fl, fr = spec["front"]
    out[:, fl], out[:, fr] = _restore_pair_mid_side(audio[:, fl], audio[:, fr], restore_mono)

    # El centro (FC) ya es contenido mono de dialogo: se restaura directo, sin M/S,
    # pero se RMS-matchea a la señal original (como todos los otros canales).
    center = spec["center"]
    out[:, center] = _rms_match(restore_mono(audio[:, center]), audio[:, center])

    # LFE (spec["lfe"]) se deja intacto: out ya es una copia de audio, no se toca.

    rl, rr = spec["rear"]
    out[:, rl], out[:, rr] = _restore_pair_mid_side(audio[:, rl], audio[:, rr], restore_mono)

    if "side" in spec:
        sl, sr = spec["side"]
        out[:, sl], out[:, sr] = _restore_pair_mid_side(audio[:, sl], audio[:, sr], restore_mono)

    return out.astype(np.float32)
