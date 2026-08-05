"""Sintesis de voz con Kokoro (Apache 2.0) sobre ONNX Runtime.

Verificado el 2026-08-04 (`scripts/spike_kokoro_tts.py`): 0,16x tiempo real y
100% de coincidencia al transcribir el audio de vuelta con Whisper.

DOS TRAMPAS MEDIDAS en las variantes que publica el mismo repo de modelos, y
ninguna avisa bien:
  - `model_q8f16.onnx` revienta el proceso con segfault al crear la sesion.
  - `model_fp16.onnx` carga, corre, devuelve audio de la duracion CORRECTA, y
    ese audio es todo NaN.
Por eso se usa `model.onnx` (fp32) y ademas se revisa el audio antes de
entregarlo: un NaN que pasa callado es peor que un error.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.services.tts_tokens import build_input_ids, phonemes_to_tokens, style_for_length

SAMPLE_RATE = 24000
MODEL_FILENAME = "model.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
VOICES_DIRNAME = "voices"
STYLE_WIDTH = 256


class TtsUnavailable(RuntimeError):
    pass


def load_vocab(model_dir: Path) -> dict[str, int]:
    tokenizer = model_dir / TOKENIZER_FILENAME
    if not tokenizer.exists():
        raise TtsUnavailable(f"Falta {TOKENIZER_FILENAME} en {model_dir}")
    data = json.loads(tokenizer.read_text(encoding="utf-8"))
    return data["model"]["vocab"]


def available_voices(model_dir: Path) -> list[str]:
    voices_dir = model_dir / VOICES_DIRNAME
    if not voices_dir.is_dir():
        return []
    return sorted(path.stem for path in voices_dir.glob("*.bin"))


def load_voice(model_dir: Path, voice: str) -> Any:
    path = model_dir / VOICES_DIRNAME / f"{voice}.bin"
    if not path.exists():
        raise TtsUnavailable(f"No existe la voz {voice!r} en {path.parent}")
    return np.fromfile(path, dtype=np.float32).reshape(-1, STYLE_WIDTH)


def assert_usable_audio(audio: Any) -> None:
    """El fallo mas caro de este modelo no es una excepcion: es audio NaN con la
    duracion correcta. Se revisa antes de entregarlo."""
    if audio.size == 0:
        raise TtsUnavailable("La sintesis devolvio audio vacio.")
    if not np.all(np.isfinite(audio)):
        raise TtsUnavailable(
            "La sintesis devolvio audio invalido (NaN). "
            "Suele ser la variante fp16 del modelo: usar model.onnx."
        )


class KokoroTtsEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cache: OrderedDict[str, tuple[Any, dict[str, int]]] = OrderedDict()
        self._lock = threading.Lock()

    def available(self, model_dir: Path) -> bool:
        return (model_dir / MODEL_FILENAME).exists() and (model_dir / TOKENIZER_FILENAME).exists()

    def synthesize(
        self, *, model_dir: Path, phonemes: str, voice: str, speed: float = 1.0
    ) -> Any:
        if speed <= 0:
            raise ValueError(f"La velocidad tiene que ser mayor que cero, no {speed!r}.")
        session, vocab = self._session_for(model_dir)
        tokens = phonemes_to_tokens(phonemes, vocab)
        input_ids = build_input_ids(tokens)
        voices = load_voice(model_dir, voice)
        style = style_for_length(voices, len(tokens))

        raw = session.run(
            None,
            {
                "input_ids": input_ids,
                "style": style,
                "speed": np.array([speed], dtype=np.float32),
            },
        )[0]
        audio = np.asarray(raw).squeeze()
        assert_usable_audio(audio)
        return audio

    def _session_for(self, model_dir: Path) -> tuple[Any, dict[str, int]]:
        key = str(model_dir)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        import onnxruntime as ort

        model_path = model_dir / MODEL_FILENAME
        if not model_path.exists():
            raise TtsUnavailable(f"Falta {MODEL_FILENAME} en {model_dir}")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        pair = (session, load_vocab(model_dir))

        with self._lock:
            self._cache[key] = pair
            self._cache.move_to_end(key)
            # Una sola sesion viva: el modelo pesa y nadie sintetiza con dos a la vez.
            while len(self._cache) > 1:
                self._cache.popitem(last=False)
        return pair
