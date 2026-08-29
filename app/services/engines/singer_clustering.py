"""Agrupa las lineas de la letra por cantante y mutea sus tramos.

Clustering jerarquico por distancia coseno (scipy, que ya es dependencia;
sklearn quedo prohibido en el contrato F2a) sobre los embeddings por linea.
Sin supervision: el caso banda son cantantes que se turnan lineas, y con 2-4
clusters la jerarquia coseno alcanza.

Las etiquetas son ESTABLES por definicion: s1 es el primero que canta, no el
cluster 1 de scipy. Sin esa regla el mismo audio saldria a veces con los ids
intercambiados y el color o el mute elegidos en review le pegarian al otro.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

# Limites del contrato F2a: 2-4 cantantes, default 2.
SINGER_COUNT_MIN = 2
SINGER_COUNT_DEFAULT = 2
SINGER_COUNT_MAX = 4

# El mute entra y sale con una rampa de ~50 ms (contrato F2a): un corte seco
# en medio de la pista de voces es un click audible en cada linea.
CROSSFADE_SECONDS = 0.05


def singer_label(index: int) -> str:
    return f"s{index + 1}"


def cluster_singers(
    embeddings: list[np.ndarray | None],
    spans: list[tuple[float, float]],
    singer_count: int,
) -> list[str]:
    """La etiqueta de cantante de cada linea, en el orden de las lineas.

    `None` en `embeddings` marca una linea sin huella (demasiado corta o sin
    voz): hereda la etiqueta del vecino valido mas cercano en tiempo.
    """
    if len(embeddings) != len(spans):
        raise ValueError(
            f"Hay {len(embeddings)} embeddings para {len(spans)} lineas: "
            "tienen que venir apareados por indice."
        )
    if not embeddings:
        return []
    validos = [i for i, huella in enumerate(embeddings) if huella is not None]
    if not validos:
        return [singer_label(0)] * len(embeddings)
    crudos = _cluster_raw([embeddings[i] for i in validos], singer_count)
    etiquetados = dict(zip(validos, _relabel_by_first_appearance(crudos)))
    return _inherit_nearest_in_time(etiquetados, spans)


def _cluster_raw(vectors: list[np.ndarray], singer_count: int) -> list[int]:
    if len(vectors) == 1:
        return [0]
    jerarquia = linkage(np.stack(vectors), method="average", metric="cosine")
    grupos = fcluster(jerarquia, t=min(singer_count, len(vectors)), criterion="maxclust")
    return [int(grupo) for grupo in grupos]


def _relabel_by_first_appearance(raw_clusters: list[int]) -> list[str]:
    # Las lineas llegan en orden de cancion, asi que "primera aparicion" es
    # literalmente quien canta primero.
    orden: dict[int, str] = {}
    for crudo in raw_clusters:
        if crudo not in orden:
            orden[crudo] = singer_label(len(orden))
    return [orden[crudo] for crudo in raw_clusters]


def _inherit_nearest_in_time(
    labeled: dict[int, str], spans: list[tuple[float, float]]
) -> list[str]:
    centros = [(inicio + fin) / 2 for inicio, fin in spans]
    etiquetas: list[str] = []
    for indice in range(len(spans)):
        if indice in labeled:
            etiquetas.append(labeled[indice])
            continue
        vecino = min(labeled, key=lambda j: (abs(centros[j] - centros[indice]), j))
        etiquetas.append(labeled[vecino])
    return etiquetas


def mute_time_spans(
    audio: np.ndarray,
    sample_rate: int,
    spans: list[tuple[float, float]],
    crossfade_seconds: float = CROSSFADE_SECONDS,
) -> np.ndarray:
    """Una copia del audio con esos tramos en silencio y rampas afuera.

    El span pedido queda mudo ENTERO: las rampas comen hasta `crossfade` del
    audio vecino en vez de dejar sonando el arranque de la linea muteada.
    Mono `(n,)` o multicanal `(n, ch)`; los tramos se combinan por minimo, asi
    que solaparse no re-enciende nada.
    """
    total = audio.shape[0]
    fade = int(round(crossfade_seconds * sample_rate))
    envolvente = np.ones(total, dtype=np.float64)
    for inicio, fin in spans:
        desde = min(max(int(round(inicio * sample_rate)), 0), total)
        hasta = min(max(int(round(fin * sample_rate)), 0), total)
        if hasta <= desde:
            continue
        envolvente = np.minimum(envolvente, _span_envelope(total, desde, hasta, fade))
    if audio.ndim == 2:
        return (audio.astype(np.float64) * envolvente[:, None]).astype(audio.dtype)
    return (audio.astype(np.float64) * envolvente).astype(audio.dtype)


def _span_envelope(total: int, start: int, end: int, fade: int) -> np.ndarray:
    envolvente = np.ones(total, dtype=np.float64)
    envolvente[start:end] = 0.0
    bajada_desde = max(0, start - fade)
    if start > bajada_desde:
        envolvente[bajada_desde:start] = _ramp(start - bajada_desde)[::-1]
    subida_hasta = min(total, end + fade)
    if subida_hasta > end:
        envolvente[end:subida_hasta] = _ramp(subida_hasta - end)
    return envolvente


def _ramp(samples: int) -> np.ndarray:
    # Puntos INTERIORES de 0 a 1: el paso queda en 1/(n+1), estrictamente menor
    # que 1/n, para que el redondeo a float32 no convierta el paso teorico
    # exacto en un escalon apenas mayor que el prometido.
    return np.linspace(0.0, 1.0, samples + 2)[1:-1]
