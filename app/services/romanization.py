"""Letra japonesa a romaji (Hepburn), conservando los tiempos.

Whisper transcribe japones en kanji/kana, y un karaoke que el usuario no puede
leer no se puede cantar. La conversion pasa SOBRE los segmentos ya cronometrados
—texto y palabras— para que todo lo que se derive de ellos (transcripcion, SRT,
ASS) salga parejo en romaji sin tocar ningun tiempo.

pykakasi se importa adentro y no arriba: el server tiene que poder arrancar
aunque la libreria falte, y fallar recien cuando alguien pida romaji.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any

from app.services.subtitles import TranscriptSegment

# Rangos Unicode que delatan japones: hiragana, katakana (+ extensiones
# foneticas), kanji y katakana de medio ancho. Lo que no cae aca (latin,
# numeros, puntuacion) ya se lee y se deja intacto.
_JAPANESE_RANGES = (
    (0x3040, 0x30FF),
    (0x31F0, 0x31FF),
    (0x4E00, 0x9FFF),
    (0xFF66, 0xFF9D),
)


def contains_japanese(text: str) -> bool:
    return any(
        low <= ord(caracter) <= high
        for caracter in text
        for low, high in _JAPANESE_RANGES
    )


@lru_cache(maxsize=1)
def _converter() -> Any:
    import pykakasi

    return pykakasi.kakasi()


def romanize_text(text: str) -> str:
    """El texto en romaji Hepburn; sin japones adentro, vuelve intacto.

    Se une con espacios porque el japones no los trae: sin separar, una linea
    entera seria una sola palabra ilegible en pantalla.
    """
    if not contains_japanese(text):
        return text
    piezas = (item["hepburn"].strip() for item in _converter().convert(text))
    romaji = " ".join(pieza for pieza in piezas if pieza)
    return romaji or text


def romanize_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    return [_romanize_segment(segmento) for segmento in segments]


def _romanize_segment(segment: TranscriptSegment) -> TranscriptSegment:
    palabras = tuple(
        replace(palabra, word=romanize_text(palabra.word))
        for palabra in segment.words
    )
    return replace(segment, text=romanize_text(segment.text), words=palabras)
