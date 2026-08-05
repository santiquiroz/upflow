"""Encadena las piezas que ya existen para producir una pista doblada.

    transcribir (ya estaba) -> traducir -> sintetizar -> acomodar en el tiempo

Lo unico propio del doblaje es el TIEMPO: cada linea traducida se sintetiza una
vez a velocidad normal para MEDIR cuanto dura, y si no entra en el hueco del
original se vuelve a sintetizar a la velocidad que la hace entrar. Estirar el
audio ya generado tambien serviria, pero re-sintetizar suena mejor: el modelo
ajusta la duracion de cada fonema en vez de reescalar una onda terminada.

Los motores entran como colaboradores explicitos, asi que esto se prueba sin
descargar un solo modelo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from app.services.dubbing import (
    DubbedPiece,
    assemble_track,
    count_overflowing,
    speed_for_slot,
    voice_for_language,
)
from app.services.translate import parse_pair

DEFAULT_VOICE = "af_heart"


class DubbingUnavailable(RuntimeError):
    pass


class SupportsTranslate(Protocol):
    def translate(self, texts: list[str], pair: Any) -> list[str]: ...


class SupportsSynthesis(Protocol):
    def synthesize(
        self, *, model_dir: Path, phonemes: str, voice: str, speed: float = 1.0
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class DubbedTrack:
    track: np.ndarray
    # Cuantas lineas no entraron en su hueco ni al maximo de velocidad. Se
    # cuenta para avisarlo: un doblaje corrido que no dice que se corrio es
    # peor que uno que lo admite.
    overflowing: int


class DubbingPipeline:
    def __init__(
        self,
        *,
        translation: SupportsTranslate,
        tts: SupportsSynthesis,
        tts_model_dir: Path,
        phonemize: Callable[[str, str | None], str],
        sample_rate: int,
        # Las voces que hay instaladas. Con esto se elige una del idioma al que
        # se dobla, en vez de leer espanol con voz inglesa.
        available_voices: list[str] | None = None,
    ) -> None:
        self.translation = translation
        self.tts = tts
        self.tts_model_dir = tts_model_dir
        self.phonemize = phonemize
        self.sample_rate = sample_rate
        self.available_voices = available_voices or []

    def build_track(
        self,
        segments: list[Any],
        *,
        source_language: str | None,
        target_language: str,
        total_seconds: float,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> DubbedTrack:
        pair = self._pair(source_language, target_language)
        voice = voice_for_language(self.available_voices, target_language) or DEFAULT_VOICE
        traducidos = self.translation.translate([s.text for s in segments], pair)

        piezas: list[DubbedPiece] = []
        for indice, (segmento, texto) in enumerate(zip(segments, traducidos), start=1):
            pieza = self._speak_into_slot(segmento, texto, target_language, voice)
            if pieza is not None:
                piezas.append(pieza)
            if progress_cb is not None:
                progress_cb(indice, len(segments))

        return DubbedTrack(
            track=assemble_track(
                piezas, total_seconds=total_seconds, sample_rate=self.sample_rate
            ),
            overflowing=count_overflowing(piezas, sample_rate=self.sample_rate),
        )

    def _pair(self, source_language: str | None, target_language: str) -> Any:
        if not target_language.strip():
            raise DubbingUnavailable("Hace falta el idioma al que doblar.")
        try:
            return parse_pair(source_language or "en", target_language)
        except Exception as exc:  # noqa: BLE001
            raise DubbingUnavailable(str(exc)) from exc

    def _speak_into_slot(
        self, segmento: Any, texto: str, language: str, voice: str
    ) -> DubbedPiece | None:
        if not texto.strip():
            return None
        phonemes = self.phonemize(texto, language)
        if not phonemes:
            # Una linea que el fonemizador no entiende no puede tirar abajo el
            # doblaje entero del video.
            return None

        slot = max(0.0, float(segmento.end) - float(segmento.start))
        audio = self._synthesize(phonemes, 1.0, voice)
        natural = len(audio) / self.sample_rate
        speed = speed_for_slot(natural_seconds=natural, slot_seconds=slot)
        if speed != 1.0:
            audio = self._synthesize(phonemes, speed, voice)
        return DubbedPiece(start=float(segmento.start), audio=audio, slot_seconds=slot)

    def _synthesize(self, phonemes: str, speed: float, voice: str) -> np.ndarray:
        audio = self.tts.synthesize(
            model_dir=self.tts_model_dir, phonemes=phonemes, voice=voice, speed=speed
        )
        return np.asarray(audio, dtype=np.float32).reshape(-1)
