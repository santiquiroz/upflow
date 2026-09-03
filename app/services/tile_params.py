from __future__ import annotations

from typing import Any

from app.config import Settings
from app.services.engines.onnx_common import TILE_OVERLAP_PX

# Semantica de `tile_size` en el job de imagen, comun a los dos motores:
#   None  -> auto: ncnn deja elegir al binario (`-t 0`, tile por heap de VRAM);
#            onnx usa ONNX_TILE_SIZE.
#   0     -> sin tiling: ncnn elige el mayor tile que entra en la VRAM libre
#            (con tope en el lado mayor de la imagen); onnx infiere en un solo pase.
#   N>=32 -> tile explicito.
# `tile_overlap` solo aplica a onnx (mezcla con feather); el binario ncnn usa un
# prepadding fijo de 10 px que no expone por linea de comandos.
NCNN_AUTO_TILE = 0
NCNN_MIN_TILE = 32
NCNN_PREPADDING_PX = 10
NCNN_TILE_STEP = 32
# Medido en RX 7800 XT con realesrgan-x4plus a escala nativa (2026-09-02): pico de
# VRAM ~ 600 MB + 0.0215 MB por pixel de tile (t200 -> 1.5 GB, t400 -> 4.1 GB,
# t900x576 -> 11.7 GB, t1000x576 -> vkAllocateMemory failed). Tiles mas grandes no
# acortan el tiempo (todos ~6-7 s) ni cambian la calidad a escala nativa.
NCNN_VRAM_BASE_MB = 600.0
NCNN_VRAM_MB_PER_TILE_PIXEL = 0.0215
NCNN_VRAM_SAFETY = 0.8
MIN_OVERLAP_MULTIPLE = 2


def validate_tile_params(tile_size: int | None, tile_overlap: int | None) -> None:
    if tile_size is not None and tile_size != 0 and tile_size < NCNN_MIN_TILE:
        raise ValueError(f"tile_size must be 0 (no tiling), or at least {NCNN_MIN_TILE}")
    if tile_overlap is not None and tile_overlap < 0:
        raise ValueError("tile_overlap must be 0 or greater")
    if tile_size and tile_overlap is not None and tile_size < tile_overlap * MIN_OVERLAP_MULTIPLE:
        raise ValueError(f"tile_size must be at least {MIN_OVERLAP_MULTIPLE}x tile_overlap")


def onnx_tile_size(job: Any, settings: Settings) -> int:
    requested = getattr(job, "tile_size", None)
    return settings.onnx_tile_size if requested is None else requested


def onnx_tile_overlap(job: Any) -> int:
    requested = getattr(job, "tile_overlap", None)
    return TILE_OVERLAP_PX if requested is None else requested


def choose_ncnn_tile(width: int, height: int, free_vram_mb: int | None) -> int:
    """Mayor tile que entra en la VRAM libre; el lado mayor de la imagen = sin tiling."""
    if free_vram_mb is None:
        return NCNN_AUTO_TILE
    no_tiling = max(width, height)
    budget_px = (free_vram_mb * NCNN_VRAM_SAFETY - NCNN_VRAM_BASE_MB) / NCNN_VRAM_MB_PER_TILE_PIXEL
    if budget_px < NCNN_MIN_TILE**2:
        return NCNN_MIN_TILE
    side = int(budget_px**0.5)
    if side >= no_tiling:
        return no_tiling
    return max(NCNN_MIN_TILE, side - side % NCNN_TILE_STEP)
