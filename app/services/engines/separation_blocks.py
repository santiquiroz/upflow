"""Troceado en bloques con crossfade, compartido por los motores de separacion.

El problema es el mismo en VR y en RoFormer aunque el DSP no se parezca en
nada: los dos acumulan buffers cuyo tamaño crece con la DURACION de la pista
(el espectrograma combinado 4band_v3 en VR, los acumuladores float64 del
overlap-add en RoFormer), asi que una pista larga entera no entra en memoria y
hay que cortarla.

Cortarla en seco NO sirve: cada motor arranca su propio contexto en el borde
del bloque (normalizacion global por espectrograma en VR, reflect-pad y rejilla
de chunks propia en RoFormer), asi que dos bloques contiguos calculan valores
apenas distintos donde se tocan y el corte se oye como un escalon. De ahi las
dos piezas de aca: `block_ranges` da a cada bloque un MARGEN que se procesa y
se tira (contexto), y `crossfade_window` cruza los tramos utiles con rampas
complementarias que suman 1 exacto.

Modulo puro (numpy y nada mas): son geometria y ventanas, sin nada del motor.
"""

from __future__ import annotations

import numpy as np


def block_ranges(
    total: int, block: int, margin: int, fade: int
) -> list[tuple[int, int, int, int]]:
    """Bloques (lo, hi, keep_lo, keep_hi) que cubren [0, total) sin huecos.

    `lo..hi` es lo que se le da al motor (incluye el margen de contexto que
    despues se descarta); `keep_lo..keep_hi` es el tramo del resultado que
    sobrevive. Cada bloque menos el primero ARRANCA `fade` muestras antes de su
    tramo propio, asi la rampa de subida de uno cae exactamente sobre la de
    bajada del anterior y las dos suman 1.
    """
    if total <= block:
        return [(0, total, 0, total)]
    ranges = []
    for start in range(0, total, block):
        keep_hi = min(start + block, total)
        if start >= keep_hi:
            break
        keep_lo = start - fade if start > 0 else 0
        ranges.append((max(0, start - margin), min(total, keep_hi + margin), keep_lo, keep_hi))
    return ranges


def crossfade_window(length: int, fade_in: int, fade_out: int) -> np.ndarray:
    """Rampas LINEALES complementarias: dos bloques contiguos suman 1 exacto."""
    window = np.ones(length, dtype=np.float32)
    if fade_in > 0:
        window[:fade_in] = np.linspace(0.0, 1.0, fade_in, endpoint=False)
    if fade_out > 0:
        window[length - fade_out :] = np.linspace(1.0, 0.0, fade_out, endpoint=False)
    return window


def overlap_add_blocks(
    destinations: tuple[np.ndarray, ...],
    sources: tuple[np.ndarray, ...],
    block: tuple[int, int, int, int],
    total: int,
    fade: int,
) -> None:
    """Suma el tramo util de un bloque sobre los acumuladores, ya ventaneado.

    `sources` son las salidas del motor para `lo..hi` (misma longitud que ese
    tramo) y `destinations` los acumuladores [C, total], emparejados por indice.
    """
    lo, _, keep_lo, keep_hi = block
    window = crossfade_window(
        keep_hi - keep_lo, fade if keep_lo > 0 else 0, fade if keep_hi < total else 0
    )
    offset = keep_lo - lo
    for source, destination in zip(sources, destinations):
        piece = source[:, offset : offset + window.size]
        destination[:, keep_lo:keep_hi] += piece * window
