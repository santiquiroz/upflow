"""Arma la pista de voz doblada a partir de los segmentos ya sintetizados.

El problema propio del doblaje no es traducir ni sintetizar — eso ya esta — sino
el TIEMPO: lo traducido casi nunca dura lo mismo que el original. Si cada linea
se suelta donde cae, a los treinta segundos la voz habla sobre otra escena.

La salida es sintetizar a la velocidad que hace falta para entrar en el hueco del
original, con un tope: pasado cierto punto la voz deja de entenderse, y apurarla
mas es peor que dejarla salir del hueco. Cuando eso pasa se cuenta y se avisa, en
vez de entregar un doblaje corrido sin decir nada.

Todo lo de aca es funcion pura de sus entradas: no toca modelos ni archivos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# El tope sale de que la voz siga siendo voz: mas rapido que 1,6x se vuelve
# atropellada, y una linea atropellada que ademas no se entiende es peor que una
# que se sale de su hueco.
SPEED_MAX = 1.6


def speed_for_slot(*, natural_seconds: float, slot_seconds: float) -> float:
    """Cuanto hay que acelerar la voz para que entre en el hueco del original.

    Solo se acelera. Una linea que ya entra se deja como esta: estirarla para
    llenar el hueco la deja arrastrada, y el silencio que queda es el mismo que
    habia en el original.
    """
    if natural_seconds <= 0:
        # No hay voz que acomodar; cambiar la velocidad no significa nada.
        return 1.0
    if slot_seconds <= 0:
        return SPEED_MAX
    if natural_seconds <= slot_seconds:
        return 1.0
    return min(SPEED_MAX, natural_seconds / slot_seconds)


@dataclass(slots=True)
class DubbedPiece:
    start: float
    audio: np.ndarray
    # Cuanto duraba el hueco original. Sirve para contar cuantas lineas no
    # entraron ni al maximo de velocidad.
    slot_seconds: float = field(default=0.0)


def _canvas_length(pieces: list[DubbedPiece], total_seconds: float, sample_rate: int) -> int:
    """La pista dura lo que el video, salvo que la ultima linea se pase.

    Cortarla para respetar la duracion nominal seria perder la ultima palabra.
    """
    minimo = int(round(total_seconds * sample_rate))
    for pieza in pieces:
        fin = int(round(pieza.start * sample_rate)) + len(pieza.audio)
        minimo = max(minimo, fin)
    return minimo


def assemble_track(
    pieces: list[DubbedPiece], *, total_seconds: float, sample_rate: int
) -> np.ndarray:
    """Pone cada linea en su segundo sobre una pista en silencio."""
    track = np.zeros(_canvas_length(pieces, total_seconds, sample_rate), dtype=np.float32)
    for pieza in pieces:
        inicio = max(0, int(round(pieza.start * sample_rate)))
        track[inicio : inicio + len(pieza.audio)] += pieza.audio
    return _without_clipping(track)


def _without_clipping(track: np.ndarray) -> np.ndarray:
    """Dos voces sumadas se pasan de 1.0 y el wav sale distorsionado."""
    pico = float(np.max(np.abs(track))) if track.size else 0.0
    if pico <= 1.0:
        return track
    return (track / pico).astype(np.float32)


def count_overflowing(pieces: list[DubbedPiece], *, sample_rate: int) -> int:
    """Cuantas lineas no entraron en su hueco ni al maximo de velocidad."""
    return sum(
        1
        for pieza in pieces
        if pieza.slot_seconds > 0 and len(pieza.audio) / sample_rate > pieza.slot_seconds
    )


# Kokoro nombra cada voz con la inicial del idioma y del genero: `ef_dora` es
# una voz femenina en espanol, `am_michael` una masculina en ingles americano.
VOICE_PREFIX_BY_LANGUAGE = {
    "en": ("a", "b"),
    "es": ("e",),
    "fr": ("f",),
    "it": ("i",),
    "pt": ("p",),
    "hi": ("h",),
    "ja": ("j",),
    "zh": ("z",),
}


def voice_for_language(available: list[str], language: str) -> str | None:
    """La voz instalada que habla ese idioma, o la primera que haya.

    Doblar al espanol con una voz inglesa suena a extranjero leyendo. No esta
    roto — el modelo pronuncia los fonemas igual — pero es peor de lo necesario
    cuando hay una voz del idioma al lado.
    """
    if not available:
        return None
    prefijos = VOICE_PREFIX_BY_LANGUAGE.get((language or "").strip().lower()[:2], ())
    for voz in available:
        if voz[:1] in prefijos:
            return voz
    return available[0]
