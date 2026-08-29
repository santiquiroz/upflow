"""Pistas de practica "minus-one": la cancion sin UN instrumento.

La resta va sobre la MEZCLA DECODIFICADA (la misma que recibio el separador) y
no sobre la suma de stems estimados: cada estimacion trae su error, y sumarlas
acumularia el error de todas las pistas en el resultado en vez del de una sola.

`minus = mix - g*stem`, con `g = 1 - guide_percent/100`: guia 0 quita el
instrumento entero; guia 30 lo deja sonando al 30% como referencia de ensayo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from app.services.engines.stem_ensemble import align_lengths

# Tope de la guia: por encima del 30% ya no es una referencia de fondo, es
# dejar el instrumento en la mezcla.
GUIDE_PERCENT_MAX = 30


def guide_gain(guide_percent: int) -> float:
    return 1.0 - guide_percent / 100.0


def derive_minus_one(
    mix: np.ndarray, stem: np.ndarray, guide_percent: int = 0
) -> np.ndarray:
    # float64 por lo mismo que el ensemble: dos señales float32 cerca de escala
    # completa saturan la aritmetica intermedia.
    aligned_mix, aligned_stem = align_lengths(
        [mix.astype(np.float64), stem.astype(np.float64)]
    )
    minus = aligned_mix - guide_gain(guide_percent) * aligned_stem
    return _guard_clipping(minus)


def _guard_clipping(minus: np.ndarray) -> np.ndarray:
    """Escala hacia abajo SOLO si el pico salio de escala.

    No normaliza: la pista de ensayo tiene que sonar al nivel de la cancion
    original, y subirle el volumen a una resta silenciosa romperia justo eso.
    """
    peak = float(np.max(np.abs(minus))) if minus.size else 0.0
    if peak > 1.0:
        return minus / peak
    return minus


def derive_minus_one_file(
    mix_path: Path, stem_path: Path, destination: Path, guide_percent: int = 0
) -> None:
    mix, sample_rate = sf.read(mix_path, dtype="float32", always_2d=True)
    stem, _ = sf.read(stem_path, dtype="float32", always_2d=True)
    minus = derive_minus_one(mix, stem, guide_percent)
    # subtype explicito como en el ensemble: el default PCM_16 le cortaria bits
    # a un intermedio que todavia pasa por el encode final.
    sf.write(destination, minus.astype(np.float32), sample_rate, subtype="FLOAT")
