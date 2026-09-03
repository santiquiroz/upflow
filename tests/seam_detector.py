"""Detector de costuras de tiling para las pruebas de regresion.

En cada borde de tile (columnas/filas multiplo de `period`) mide el gradiente
medio y lo divide por el de sus vecinas inmediatas. Una salida limpia da ~1.0;
una rejilla de tiles visible da > 5 (la salida rota de ncnn `-s 2` dio 14-37).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

NEIGHBOR_RADIUS = 6
NEIGHBOR_GAP = 2


@dataclass(frozen=True, slots=True)
class SeamReport:
    period: int
    ratio_x: float
    ratio_y: float
    boundaries_x: int
    boundaries_y: int

    @property
    def worst_ratio(self) -> float:
        return max(self.ratio_x, self.ratio_y)


def load_gray(image: Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return np.asarray(Image.fromarray(image).convert("L"), dtype=np.float32)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.float32)
    with Image.open(image) as opened:
        return np.asarray(opened.convert("L"), dtype=np.float32)


def column_gradients(gray: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(gray, axis=1)).mean(axis=0)


def boundary_ratio(gradients: np.ndarray, index: int) -> float | None:
    low, high = index - NEIGHBOR_RADIUS, index + NEIGHBOR_RADIUS
    if low < 0 or high >= len(gradients):
        return None
    neighbors = np.concatenate(
        [gradients[low : index - NEIGHBOR_GAP + 1], gradients[index + NEIGHBOR_GAP : high + 1]]
    )
    baseline = float(neighbors.mean())
    if baseline <= 1e-6:
        return None
    return float(gradients[index]) / baseline


def boundary_ratios(gradients: np.ndarray, period: int) -> list[float]:
    # El borde entre la columna period-1 y period es el diff de indice period-1.
    ratios = []
    index = period - 1
    while index < len(gradients) - NEIGHBOR_RADIUS:
        ratio = boundary_ratio(gradients, index)
        if ratio is not None:
            ratios.append(ratio)
        index += period
    return ratios


def _median_or_one(values: list[float]) -> float:
    return float(np.median(values)) if values else 1.0


def measure_seams(image: Path | Image.Image | np.ndarray, period: int) -> SeamReport:
    gray = load_gray(image)
    ratios_x = boundary_ratios(column_gradients(gray), period)
    ratios_y = boundary_ratios(column_gradients(gray.T), period)
    if not ratios_x and not ratios_y:
        raise ValueError(f"image {gray.shape[1]}x{gray.shape[0]} has no tile boundary at period {period}")
    return SeamReport(
        period=period,
        ratio_x=_median_or_one(ratios_x),
        ratio_y=_median_or_one(ratios_y),
        boundaries_x=len(ratios_x),
        boundaries_y=len(ratios_y),
    )
