"""Conversión de voz: dice lo mismo, con la voz de otra grabación.

Cadena verificada el 2026-08-05 (`scripts/spike_voice_conversion.py` y la quinta
entrada de docs/superpowers/specs/2026-08-05-voice-conversion-findings.md):

    muestra de referencia -> x-vector -> SpeechT5 VC -> audio con esa voz

Clonando una muestra masculina sobre una frase con voz femenina, la salida quedo
a 0,777 de la voz destino y 0,446 de la de origen, con el contenido intacto al
90%. Con un x-vector de otro espacio daba 0,609 y 0,673 — al reves, o sea peor
que no convertir. De ahi que el x-vector salga SI O SI de `xvector.py`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.services.xvector import XvectorEncoder, XvectorUnavailable

SAMPLE_RATE = 16000
VC_DIRNAME = "speecht5-vc"
VOCODER_DIRNAME = "speecht5-hifigan"
XVECTOR_FILENAME = "tdnn.onnx"

# Un audio mas largo que esto tarda demasiado para una request sincronica y
# conviene que el usuario lo parta.
MAX_SECONDS = 60


class VoiceConversionUnavailable(RuntimeError):
    pass


def assert_usable_audio(audio: Any) -> None:
    """Mismo guard que el TTS, por el mismo motivo: el fallo caro de estos
    modelos no es una excepcion sino audio NaN con la duracion correcta."""
    if audio.size == 0:
        raise VoiceConversionUnavailable("La conversion devolvio audio vacio.")
    if not np.all(np.isfinite(audio)):
        raise VoiceConversionUnavailable("La conversion devolvio audio invalido (NaN).")


def assert_convertible(audio: Any) -> None:
    if audio.size == 0:
        raise VoiceConversionUnavailable("No hay audio que convertir.")
    seconds = audio.size / SAMPLE_RATE
    if seconds > MAX_SECONDS:
        raise VoiceConversionUnavailable(
            f"El audio dura {seconds:.0f} s y el maximo es {MAX_SECONDS} s. "
            "Partilo en tramos mas cortos."
        )


class VoiceConversionEngine:
    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root
        self.xvector = XvectorEncoder(models_root / "xvector" / XVECTOR_FILENAME)
        self._loaded: tuple[Any, Any, Any] | None = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return (
            (self.models_root / VC_DIRNAME).is_dir()
            and (self.models_root / VOCODER_DIRNAME).is_dir()
            and self.xvector.available()
        )

    def convert(self, *, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        assert_convertible(source)
        try:
            speaker = self.xvector.encode(reference)
        except XvectorUnavailable as exc:
            raise VoiceConversionUnavailable(str(exc)) from exc

        import torch

        processor, model, vocoder = self._load()
        inputs = processor(audio=source, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate_speech(
                inputs["input_values"],
                torch.tensor(speaker).unsqueeze(0),
                vocoder=vocoder,
            )
        audio = generated.cpu().numpy()
        assert_usable_audio(audio)
        return audio

    def _load(self) -> tuple[Any, Any, Any]:
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            from transformers import (
                SpeechT5ForSpeechToSpeech,
                SpeechT5HifiGan,
                SpeechT5Processor,
            )

            vc_dir = self.models_root / VC_DIRNAME
            vocoder_dir = self.models_root / VOCODER_DIRNAME
            if not vc_dir.is_dir():
                raise VoiceConversionUnavailable(
                    f"Falta el modelo en {vc_dir}. Instalalo con scripts/download-voice-conversion.ps1"
                )
            self._loaded = (
                SpeechT5Processor.from_pretrained(str(vc_dir)),
                SpeechT5ForSpeechToSpeech.from_pretrained(str(vc_dir)),
                SpeechT5HifiGan.from_pretrained(str(vocoder_dir)),
            )
            return self._loaded
