from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Emparejar el grano del sensor en la zona editada.
#
# El difusor devuelve un relleno limpio mientras el resto de la foto conserva
# su ruido de alta frecuencia, y al imprimir el parche se ve plástico. Se mide
# cuánto ruido le falta a lo editado y se le suma ruido nuevo con ese nivel,
# solo donde la máscara lo pide.
# ---------------------------------------------------------------------------

# Ancho del blur que separa la textura fina (grano) del contenido.
_HF_SIGMA = 1.2
# Ancho del blur del ruido que se suma: apenas correlacionado entre vecinos.
_NOISE_SIGMA = 0.7


def _high_freq(channel: np.ndarray) -> np.ndarray:
    import cv2

    plano = np.asarray(channel, dtype=np.float32)
    return plano - cv2.GaussianBlur(plano, (0, 0), _HF_SIGMA)


def _grain_reference(mask: np.ndarray) -> np.ndarray:
    import cv2

    marked = (mask > 0).astype(np.uint8)
    _, _, width, height = cv2.boundingRect(marked)
    radius = max(16, round(0.05 * max(width, height)))
    diameter = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    outside = mask == 0
    ring = (cv2.dilate(marked, kernel) > 0) & outside
    return ring if bool(ring.any()) else outside


def match_grain(
    edited: np.ndarray, original: np.ndarray, mask: np.ndarray, seed: int = 0
) -> np.ndarray:
    """Suma a `edited` el grano que le falta respecto de `original`.

    RGB uint8 HxWx3; máscara uint8 HxW 0-255 con degradado. El ruido se pesa
    por el alpha de la máscara: fuera de la marca no se toca un solo píxel.
    """
    fuera = mask == 0
    dentro = mask >= 128
    if not bool(fuera.any()) or not bool(dentro.any()):
        return edited
    import cv2

    referencia = _grain_reference(mask)
    rng = np.random.default_rng(seed)
    alpha = (mask.astype(np.float32) / 255.0)[..., None]
    salida = edited.astype(np.float32)
    for canal in range(3):
        want = float(_high_freq(original[:, :, canal])[referencia].std())
        have = float(_high_freq(edited[:, :, canal])[dentro].std())
        agrega = math.sqrt(max(0.0, want * want - have * have))
        if agrega <= 0:
            continue
        ruido = rng.standard_normal(mask.shape).astype(np.float32)
        ruido = cv2.GaussianBlur(ruido, (0, 0), _NOISE_SIGMA)
        nivel = float(_high_freq(ruido)[dentro].std())
        if nivel <= 0:
            continue
        # rint, NO truncar: truncar corre el color medio 0.5 hacia abajo.
        salida[:, :, canal] += (ruido * (agrega / nivel)) * alpha[:, :, 0]
    return np.clip(np.rint(salida), 0, 255).astype(np.uint8)
