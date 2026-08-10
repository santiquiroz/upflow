"""Extrae el x-vector de una grabación, para clonar su voz.

SpeechT5 VC solo clona bien si el x-vector viene del espacio del encoder de
speechbrain (medido 2026-08-05: con embeddings de WavLM el resultado se parecia
MAS a la voz de origen que a la de destino, o sea peor que no convertir).

speechbrain no se puede instalar en el entorno de la app — 1.1.0 tiene un lazy
import roto y 1.0.2 choca con torchaudio 2.11 —, asi que el TDNN se exporto a
ONNX aparte (`scripts/export_xvector_onnx.py`) y se publico en
santiquiroz/port-xvector-onnx, de donde el pack de conversion de voz lo baja
como cualquier otro modelo. Lo que ese export NO pudo llevarse es el frontend de
features, que revienta al trazar el STFT. Eso es lo que se reproduce aca.

Los numeros salen de LEER el modelo cargado, no de la documentacion.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
N_MELS = 24
N_FFT = 400
WIN_LENGTH = 400  # 25 ms a 16 kHz
HOP_LENGTH = 160  # 10 ms a 16 kHz
F_MIN = 0.0
F_MAX = 8000.0
# speechbrain toma la MAGNITUD (power=0.5 sobre el espectro de potencia), no la
# potencia. Elevar al cuadrado aca cambia la escala del log y corre el embedding.
LOG_EPSILON = 1e-10


class XvectorUnavailable(RuntimeError):
    pass


def _mel_filterbank() -> np.ndarray:
    """Banco de filtros mel en escala Slaney, como el de speechbrain."""

    def hz_to_mel(hz: np.ndarray | float) -> Any:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    puntos_mel = np.linspace(hz_to_mel(F_MIN), hz_to_mel(F_MAX), N_MELS + 2)
    puntos_hz = mel_to_hz(puntos_mel)
    bins = np.floor((N_FFT + 1) * puntos_hz / SAMPLE_RATE).astype(int)

    banco = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        izq, centro, der = bins[m - 1], bins[m], bins[m + 1]
        for k in range(izq, centro):
            if centro > izq:
                banco[m - 1, k] = (k - izq) / (centro - izq)
        for k in range(centro, der):
            if der > centro:
                banco[m - 1, k] = (der - k) / (der - centro)
    return banco


_FILTERBANK = _mel_filterbank()


def compute_fbank(audio: np.ndarray) -> np.ndarray:
    """Log-mel de la señal, con la forma (cuadros, n_mels) que espera el TDNN."""
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    if len(audio) < WIN_LENGTH:
        # Sin al menos una ventana entera no hay nada que analizar, y seguir
        # revienta con un error de formas que no le dice nada a nadie.
        raise XvectorUnavailable(
            f"La grabacion es demasiado corta: hacen falta al menos "
            f"{WIN_LENGTH / SAMPLE_RATE:.2f} s de audio."
        )

    ventana = np.hamming(WIN_LENGTH).astype(np.float32)
    cuadros = 1 + (len(audio) - WIN_LENGTH) // HOP_LENGTH

    espectro = np.empty((cuadros, N_FFT // 2 + 1), dtype=np.float32)
    for i in range(cuadros):
        inicio = i * HOP_LENGTH
        trozo = audio[inicio : inicio + WIN_LENGTH] * ventana
        espectro[i] = np.abs(np.fft.rfft(trozo, n=N_FFT))

    mel = espectro @ _FILTERBANK.T
    return np.log(mel + LOG_EPSILON).astype(np.float32)


def normalize_sentence(feats: np.ndarray) -> np.ndarray:
    """Resta la media de la oracion. NO divide por el desvio.

    speechbrain usa `norm_type="sentence"` con `std_norm=False`. Dividir tambien
    por el desvio deja el embedding fuera del espacio y la conversion deja de
    clonar.
    """
    return feats - feats.mean(axis=0, keepdims=True)


class XvectorEncoder:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._session: Any | None = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return self.model_path.exists()

    def encode(self, audio: np.ndarray) -> np.ndarray:
        feats = normalize_sentence(compute_fbank(audio))
        salida = self._run(feats[None, :, :])
        vector = np.asarray(salida).reshape(-1)
        norma = np.linalg.norm(vector)
        if not np.isfinite(norma) or norma == 0:
            raise XvectorUnavailable("No se pudo sacar la voz de esa grabacion.")
        return (vector / norma).astype(np.float32)

    def _run(self, feats: np.ndarray) -> Any:
        with self._lock:
            if self._session is None:
                import onnxruntime as ort

                if not self.model_path.exists():
                    # El encoder ya viene en el pack de conversion de voz: la app
                    # lo baja sola. Antes este mensaje mandaba a generarlo a mano
                    # en un venv con torch+speechbrain.
                    from app.services.missing_pack import missing_pack_message

                    raise XvectorUnavailable(
                        missing_pack_message(
                            "voice-conversion",
                            detail=f"Se esperaba el encoder de voz en {self.model_path}.",
                        )
                    )
                self._session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
            session = self._session
        return session.run(["embedding"], {"feats": feats})[0]
