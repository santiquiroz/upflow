"""Conversion de timbre con OpenVoice V2, por ONNX.

Reemplazo del camino SpeechT5: el mismo trabajo —decir lo mismo con otra voz—
con un modelo mucho mas nuevo y, sobre todo, con pesos MIT. Los clonadores que
salen primero en cualquier busqueda (F5-TTS, E2-TTS, XTTS) publican los pesos con
CC-BY-NC, o sea no se pueden distribuir dentro de una aplicacion que otra gente
descarga.

El algoritmo, la paridad contra el modelo original y las trampas estan en el port
propio: https://github.com/santiquiroz/port-openvoice-onnx

Convive con el motor viejo en vez de borrarlo: quien ya tenia bajado el pack de
SpeechT5 no se queda sin la funcion porque salio otro modelo.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.services.engines.voice_convert import (
    VoiceConversionUnavailable,
    assert_convertible,
    assert_usable_audio,
)
from app.services.missing_pack import missing_pack_message

# OpenVoice trabaja a 22050, no a los 16000 de SpeechT5. La diferencia viaja como
# constante propia del motor porque el que llama tiene que remuestrear: entrarle
# audio a otra frecuencia no falla, cambia el tono de la voz.
SAMPLE_RATE = 22050

N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
# Canales del latente (`inter_channels`). El ruido tiene que tener esta forma o
# el grafo rechaza la entrada.
LATENT_CHANNELS = 192
# El default de OpenVoice: cuanto se aparta del original. Mas alto se aleja mas
# del timbre de origen, y tambien de su articulacion.
DEFAULT_TAU = 0.3

CONVERTER_FILENAME = "openvoice_converter.onnx"
SPEAKER_FILENAME = "openvoice_speaker.onnx"
OPENVOICE_DIRNAME = "openvoice"


def hann_window(size: int) -> np.ndarray:
    """La ventana de Hann PERIODICA, como la de torch.

    `np.hanning` devuelve la simetrica, que difiere en una muestra. Esa muestra
    alcanza para que la voz salga apenas corrida o metalica, y nada avisa: las
    dos son "una ventana de Hann".
    """
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(size) / size)


def spectrogram(audio: np.ndarray) -> np.ndarray:
    """La magnitud de la STFT en la forma [1, bins, cuadros] que come el grafo.

    El relleno es REFLEJADO y no con ceros: los ceros meten un borde duro que el
    modelo lee como un chasquido al principio y al final de cada conversion.
    """
    if audio.ndim != 1:
        raise ValueError(f"Se espera audio mono de una dimension, llego {audio.shape}")
    ancho = int((N_FFT - HOP_LENGTH) / 2)
    acolchado = np.pad(audio.astype(np.float64), ancho, mode="reflect")
    cuadros = 1 + (len(acolchado) - N_FFT) // HOP_LENGTH
    if cuadros <= 0:
        raise VoiceConversionUnavailable(
            f"El audio es mas corto que {N_FFT / SAMPLE_RATE:.2f} s y no alcanza "
            "para un solo cuadro de analisis."
        )
    ventana = hann_window(WIN_LENGTH)
    recortes = np.stack(
        [acolchado[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] * ventana for i in range(cuadros)]
    )
    espectro = np.fft.rfft(recortes, n=N_FFT, axis=-1)
    # El +1e-6 viene del modelo original: sin el, un bin exactamente en cero da
    # gradiente infinito. Se conserva para que el numero sea EL MISMO.
    magnitud = np.sqrt(np.real(espectro) ** 2 + np.imag(espectro) ** 2 + 1e-6)
    return magnitud.T.astype(np.float32)[np.newaxis, ...]


class OpenVoiceConversionEngine:
    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root
        self._sessions: tuple[Any, Any] | None = None
        self._lock = threading.Lock()

    def required_paths(self) -> tuple[Path, ...]:
        base = self.models_root / OPENVOICE_DIRNAME
        return (base / CONVERTER_FILENAME, base / SPEAKER_FILENAME)

    def available(self) -> bool:
        return all(path.exists() for path in self.required_paths())

    def convert(
        self,
        *,
        source: np.ndarray,
        reference: np.ndarray,
        tau: float = DEFAULT_TAU,
        seed: int | None = None,
    ) -> np.ndarray:
        """El audio de `source`, dicho con la voz de `reference`.

        `seed` existe porque el modelo es estocastico: sin fijarla, dos corridas
        del mismo pedido dan audio distinto y no se puede volver a una toma que
        gusto.
        """
        assert_convertible(source, sample_rate=SAMPLE_RATE)
        convertidor, hablante = self._load()

        spec_origen = spectrogram(source)
        g_src = hablante.run(None, {"spec": spec_origen})[0]
        g_tgt = hablante.run(None, {"spec": spectrogram(reference)})[0]

        cuadros = spec_origen.shape[-1]
        ruido = np.random.default_rng(seed).standard_normal(
            (1, LATENT_CHANNELS, cuadros)
        ).astype(np.float32)
        salida = convertidor.run(
            None,
            {
                "spec": spec_origen,
                "spec_lengths": np.array([cuadros], dtype=np.int64),
                "g_src": g_src,
                "g_tgt": g_tgt,
                "noise": ruido,
                "tau": np.array(tau, dtype=np.float32),
            },
        )[0]
        audio = salida[0, 0]
        assert_usable_audio(audio)
        return audio

    def _load(self) -> tuple[Any, Any]:
        with self._lock:
            if self._sessions is not None:
                return self._sessions
            faltantes = [p for p in self.required_paths() if not p.exists()]
            if faltantes:
                raise VoiceConversionUnavailable(
                    missing_pack_message(
                        "openvoice",
                        detail=f"Falta {faltantes[0].name} en {faltantes[0].parent}.",
                    )
                )
            import onnxruntime as ort

            convertidor, hablante = (
                ort.InferenceSession(str(ruta), providers=["CPUExecutionProvider"])
                for ruta in self.required_paths()
            )
            self._sessions = (convertidor, hablante)
            return self._sessions
