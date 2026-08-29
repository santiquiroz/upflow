"""Huella de voz por linea de letra, para saber quien canta cual.

El protocolo existe por la decision #6 del spec F2a (2026-08-28): el encoder
tiene que ser intercambiable. Hoy es el x-vector TDNN que ya viaja en el pack
de conversion de voz — reusarlo evita un pack nuevo de 30 MB —; si la calidad
no alcanza se cambia por WeSpeaker sin tocar el clustering ni el manager.

Las ventanas se cortan por los tiempos de la transcripcion (que se hizo sobre
la MEZCLA, decision #7) pero el audio sale del STEM VOCAL: ahi la voz esta
sola y el embedding no arrastra a la banda.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from app.services.xvector import XvectorEncoder, XvectorUnavailable

# El que espera el x-vector; cualquier encoder que se enchufe acomoda su
# frontend a esta frecuencia, no al reves.
EMBEDDING_SAMPLE_RATE = 16000

# Por debajo de esto la ventana no da para una huella confiable: la linea
# queda sin embedding y hereda el cantante del vecino en tiempo (contrato F2a).
MIN_LINE_SECONDS = 0.4


class SpeakerEmbedder(Protocol):
    def available(self) -> bool: ...

    def encode(self, audio: np.ndarray) -> np.ndarray: ...


def xvector_speaker_embedder(model_path: Path) -> SpeakerEmbedder:
    return XvectorEncoder(model_path)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def resample_for_embedding(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == EMBEDDING_SAMPLE_RATE:
        return audio.astype(np.float32)
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)
    muestras = int(round(len(audio) / source_rate * EMBEDDING_SAMPLE_RATE))
    posiciones = np.linspace(0.0, len(audio) - 1, num=muestras, dtype=np.float64)
    return np.interp(posiciones, np.arange(len(audio)), audio).astype(np.float32)


def line_embeddings(
    embedder: SpeakerEmbedder,
    audio: np.ndarray,
    sample_rate: int,
    spans: list[tuple[float, float]],
) -> list[np.ndarray | None]:
    """Un embedding por linea, o None donde no hay huella que sacar.

    None cubre dos casos con el mismo destino (heredar por tiempo): la linea
    demasiado corta y la que el encoder no puede leer (una linea "cantada"
    que en el stem vocal es puro silencio da norma cero).
    """
    remuestreado = resample_for_embedding(to_mono(audio), sample_rate)
    minimo = int(MIN_LINE_SECONDS * EMBEDDING_SAMPLE_RATE)
    resultados: list[np.ndarray | None] = []
    for inicio, fin in spans:
        desde = max(int(round(inicio * EMBEDDING_SAMPLE_RATE)), 0)
        hasta = max(int(round(fin * EMBEDDING_SAMPLE_RATE)), 0)
        ventana = remuestreado[desde:hasta]
        if len(ventana) < minimo:
            resultados.append(None)
            continue
        try:
            resultados.append(embedder.encode(ventana))
        except XvectorUnavailable:
            resultados.append(None)
    return resultados
