"""Compara luminosidad y textura con el anillo para penalizar bultos y patrones inventados.

La textura se mide a 1/4 para ignorar el ruido del sensor; menor puntaje es mejor.
"""

import cv2
import numpy as np


def _ring(mask: np.ndarray, protect: np.ndarray, frac: float = 0.10) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    r = max(24, int(frac * max(np.ptp(ys), np.ptp(xs))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    return (cv2.dilate(mask.astype(np.uint8), k) > 0) & ~mask & ~protect


def _texture(gray4: np.ndarray, sel4: np.ndarray) -> float:
    lap = np.abs(cv2.Laplacian(gray4, cv2.CV_32F, ksize=3))
    return float(lap[sel4].mean()) if sel4.any() else 0.0


def fill_score(rgb: np.ndarray, mask: np.ndarray, protect: np.ndarray) -> dict:
    ring = _ring(mask, protect)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    d_light = abs(float(lab[..., 0][mask].mean() - lab[..., 0][ring].mean())) * 100 / 255
    small = (rgb.shape[1] // 4, rgb.shape[0] // 4)
    g4 = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), small, interpolation=cv2.INTER_AREA)
    m4 = cv2.resize(mask.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST) > 0
    r4 = cv2.resize(ring.astype(np.uint8), small, interpolation=cv2.INTER_NEAREST) > 0
    tex = _texture(g4, m4) / max(_texture(g4, r4), 1e-3)
    score = d_light / 15 + max(0.0, tex - 1.2)
    return {"score": round(score, 3), "d_light": round(d_light, 1), "tex_ratio": round(tex, 2)}
