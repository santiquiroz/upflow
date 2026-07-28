from __future__ import annotations

# Factores EXTRAPOLADOS, no medidos por modelo (ver el spec de 2026-07-28,
# seccion vram_estimate). El margen sobre los pesos cubre activaciones y
# buffers intermedios, que crecen con la resolucion mientras los pesos no.
# Revisar con mediciones reales; los tests fijan monotonia, no estos numeros.
_STEPS: tuple[tuple[int, float], ...] = (
    (512 * 512, 1.25),
    (768 * 768, 1.45),
    (1024 * 1024, 1.70),
)


def _factor(pixels: int) -> float:
    if pixels <= _STEPS[0][0]:
        return _STEPS[0][1]
    if pixels >= _STEPS[-1][0]:
        return _STEPS[-1][1]
    for (low_px, low_f), (high_px, high_f) in zip(_STEPS, _STEPS[1:]):
        if low_px <= pixels <= high_px:
            ratio = (pixels - low_px) / (high_px - low_px)
            return low_f + ratio * (high_f - low_f)
    return _STEPS[-1][1]


def estimate_peak_bytes(weight_bytes: int, width: int, height: int) -> int:
    return int(weight_bytes * _factor(max(width, 0) * max(height, 0)))
