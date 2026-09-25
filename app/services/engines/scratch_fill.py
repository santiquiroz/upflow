from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from app.services.engines.migan_eraser import build_migan_inputs
from app.services.engines.onnx_common import blend_tiles_float, tile_starts
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.engines.tiled_restore_runner import BlendTile, RestoreCancelled, TileInfer, TileProgress
from app.services.inpaint_mask import feather_mask
from app.services.photo_diagnosis import LARGE_HOLE_PX
from app.services.restore_models import MIGAN_MODEL_ID

FaceBox = tuple[float, float, float, float]

FAST_ENGINE = "fast"
CLASSIC_ENGINE = "classic"
FILL_ENGINES = (FAST_ENGINE, CLASSIC_ENGINE)
# A resolucion nativa: con la foto entera el grafo reduce el bbox de toda la mascara a 512 (§3.4.2).
MIGAN_WINDOW = 512
MIGAN_STRIDE = 384
MIGAN_PRECISION = "fp32"
FEATHER_PX = 2
TELEA_RADIUS = 3
BIT_DEPTH_PEAKS = {8: 255.0, 16: 65535.0}
U8_PEAK = 255.0
RGB_CHANNELS = 3


@dataclass(frozen=True, slots=True)
class FillRequest:
    engine: str = FAST_ENGINE
    leave_large_holes: bool = False
    bit_depth: int = 8
    face_boxes: tuple[FaceBox, ...] = ()


@dataclass(frozen=True, slots=True)
class FillResult:
    image: np.ndarray
    engine: str
    windows: int
    coverage: float
    large_holes: int
    large_holes_left_unfilled: bool
    touches_faces: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "engine": self.engine,
            "windows": self.windows,
            "finalCoverage": self.coverage,
            "largeHoles": self.large_holes,
            "largeHolesLeftUnfilled": self.large_holes_left_unfilled,
            "touchesFaces": self.touches_faces,
        }


def fill_damage(
    image: np.ndarray,
    mask: np.ndarray,
    request: FillRequest,
    *,
    migan: TileInfer | None = None,
    cancel_event: threading.Event | None = None,
    on_progress: TileProgress | None = None,
) -> FillResult:
    _validate(image, mask, request)
    holes = mask.astype(bool, copy=False)
    large, large_count = _large_holes(holes)
    final = holes & ~large if request.leave_large_holes else holes
    engine = effective_engine(request.engine, migan)
    if engine == FAST_ENGINE:
        filled, windows = migan_fill(image, final, migan, cancel_event=cancel_event, on_progress=on_progress)
    else:
        filled, windows = telea_fill(image, final, request.bit_depth), 0
    return FillResult(
        image=filled,
        engine=engine,
        windows=windows,
        coverage=float(final.mean()),
        large_holes=large_count,
        large_holes_left_unfilled=request.leave_large_holes and large_count > 0,
        touches_faces=fill_touches_faces(final, request.face_boxes),
    )


def effective_engine(requested: str, migan: TileInfer | None) -> str:
    # Sin el pack migan, "Fast" cae a Telea (§3.4.2).
    return CLASSIC_ENGINE if requested == FAST_ENGINE and migan is None else requested


def migan_infer(engine: PhotoRestoreEngine, device: str) -> TileInfer:
    engine.begin_phase(device)
    return engine.tile_infer(MIGAN_MODEL_ID, device, MIGAN_PRECISION, infer_for=migan_window_infer, clamp=False)


def migan_window_infer(session: Any) -> TileInfer:
    def infer(window: np.ndarray) -> np.ndarray:
        rgb = Image.fromarray(np.ascontiguousarray(window[:, :, :RGB_CHANNELS]))
        hole = Image.fromarray(np.ascontiguousarray(window[:, :, RGB_CHANNELS]))
        image_input, known_input = build_migan_inputs(rgb, hole)
        result = session.run(None, {"image": image_input, "mask": known_input})[0]
        return np.transpose(np.asarray(result[0], dtype=np.uint8), (1, 2, 0))

    return infer


def window_origins(height: int, width: int) -> list[tuple[int, int]]:
    overlap = MIGAN_WINDOW - MIGAN_STRIDE
    rows = tile_starts(height, MIGAN_WINDOW, overlap)
    columns = tile_starts(width, MIGAN_WINDOW, overlap)
    return [(y0, x0) for y0 in rows for x0 in columns]


def masked_windows(holes: np.ndarray) -> list[tuple[int, int]]:
    return [
        (y0, x0)
        for y0, x0 in window_origins(*holes.shape)
        if holes[y0 : y0 + MIGAN_WINDOW, x0 : x0 + MIGAN_WINDOW].any()
    ]


def migan_fill(
    image: np.ndarray,
    holes: np.ndarray,
    migan: TileInfer,
    *,
    cancel_event: threading.Event | None = None,
    on_progress: TileProgress | None = None,
) -> tuple[np.ndarray, int]:
    origins = masked_windows(holes)
    if not origins:
        return image, 0
    hole_u8 = _to_u8_mask(holes)
    stacked = np.dstack([_quantize(_as_rgb(image), U8_PEAK, np.uint8), hole_u8])
    tiles = _run_windows(migan, stacked, origins, cancel_event, on_progress)
    height, width = holes.shape
    painted = blend_tiles_float(tiles, height, width, RGB_CHANNELS, 1, feather=MIGAN_WINDOW - MIGAN_STRIDE)
    # MI-GAN sangra fuera del hueco: solo se pega lo enmascarado, con 2 px de feather.
    alpha = feather_mask(hole_u8, FEATHER_PX).astype(np.float32) / U8_PEAK * _window_coverage(origins, holes.shape)
    return composite(image, _like(painted / U8_PEAK, image), alpha), len(origins)


def telea_fill(image: np.ndarray, holes: np.ndarray, bit_depth: int) -> np.ndarray:
    if not holes.any():
        return image
    peak = BIT_DEPTH_PEAKS[bit_depth]
    quantized = _quantize(image, peak, np.uint8 if bit_depth == 8 else np.uint16)
    hole_u8 = _to_u8_mask(holes)
    # Telea en 3 canales solo acepta 8 bits: en 16 va canal por canal.
    inpainted = _inpaint(quantized, hole_u8) if bit_depth == 8 else _inpaint_per_channel(quantized, hole_u8)
    filled = inpainted.reshape(image.shape).astype(np.float32) / np.float32(peak)
    return composite(image, filled, holes.astype(np.float32))


def composite(image: np.ndarray, filled: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    weight = alpha if image.ndim == 2 else alpha[:, :, np.newaxis]
    blended = image + weight * (filled - image)
    return np.where(weight > 0.0, blended, image).astype(np.float32, copy=False)


def hole_widths(mask: np.ndarray) -> np.ndarray:
    return _labeled_widths(mask)[1]


def large_hole_mask(mask: np.ndarray) -> np.ndarray:
    return _large_holes(mask)[0]


def fill_touches_faces(mask: np.ndarray, face_boxes: Sequence[FaceBox]) -> bool:
    # Cualquier relleno dentro de la caja de una cara, aunque sea chico, inventa rasgos (§3.4.2).
    return any(_box_region(mask, box).any() for box in face_boxes)


def _validate(image: np.ndarray, mask: np.ndarray, request: FillRequest) -> None:
    if request.engine not in FILL_ENGINES:
        raise ValueError(f"Unknown fill engine {request.engine!r}; expected one of {FILL_ENGINES}")
    if request.bit_depth not in BIT_DEPTH_PEAKS:
        raise ValueError(f"Bit depth must be one of {tuple(BIT_DEPTH_PEAKS)}, got {request.bit_depth}")
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Mask {mask.shape} does not match the photo {image.shape[:2]}")


def _labeled_widths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, count = ndimage.label(mask)
    if count == 0:
        return labels, np.zeros(0, dtype=np.float32)
    padded = np.pad(mask, 1).astype(np.uint8)
    inside = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    # Ancho de cada componente = el doble de su distancia maxima al borde (§3.4.2).
    widths = 2.0 * np.asarray(ndimage.maximum(inside, labels, np.arange(1, count + 1)), dtype=np.float32)
    return labels, widths


def _large_holes(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, widths = _labeled_widths(mask)
    large_labels = np.flatnonzero(widths > LARGE_HOLE_PX) + 1
    return np.isin(labels, large_labels), int(large_labels.size)


def _box_region(mask: np.ndarray, box: FaceBox) -> np.ndarray:
    left, top, right, bottom = box
    height, width = mask.shape
    rows = slice(max(0, math.floor(top)), min(height, math.ceil(bottom)))
    columns = slice(max(0, math.floor(left)), min(width, math.ceil(right)))
    return mask[rows, columns]


def _run_windows(
    migan: TileInfer,
    stacked: np.ndarray,
    origins: list[tuple[int, int]],
    cancel_event: threading.Event | None,
    on_progress: TileProgress | None,
) -> list[BlendTile]:
    tiles: list[BlendTile] = []
    for done, (y0, x0) in enumerate(origins, start=1):
        _raise_if_cancelled(cancel_event)
        window = stacked[y0 : y0 + MIGAN_WINDOW, x0 : x0 + MIGAN_WINDOW]
        tiles.append((y0, x0, window.shape[0], window.shape[1], migan(window)))
        if on_progress is not None:
            on_progress(done, len(origins))
    return tiles


def _window_coverage(origins: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    covered = np.zeros(shape, dtype=np.float32)
    for y0, x0 in origins:
        covered[y0 : y0 + MIGAN_WINDOW, x0 : x0 + MIGAN_WINDOW] = 1.0
    return covered


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RestoreCancelled("Restoration cancelled")


def _as_rgb(image: np.ndarray) -> np.ndarray:
    hwc = image[:, :, np.newaxis] if image.ndim == 2 else image
    return np.repeat(hwc, RGB_CHANNELS, axis=2) if hwc.shape[2] == 1 else hwc


def _like(rgb: np.ndarray, image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return rgb.mean(axis=2, dtype=np.float32)
    if image.shape[2] == 1:
        return rgb.mean(axis=2, keepdims=True, dtype=np.float32)
    return rgb.astype(np.float32, copy=False)


def _quantize(image: np.ndarray, peak: float, dtype: type[np.generic]) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * peak).astype(dtype)


def _to_u8_mask(holes: np.ndarray) -> np.ndarray:
    return np.where(holes, np.uint8(255), np.uint8(0))


def _inpaint(plane: np.ndarray, hole_u8: np.ndarray) -> np.ndarray:
    return cv2.inpaint(np.ascontiguousarray(plane), hole_u8, TELEA_RADIUS, cv2.INPAINT_TELEA)


def _inpaint_per_channel(quantized: np.ndarray, hole_u8: np.ndarray) -> np.ndarray:
    if quantized.ndim == 2:
        return _inpaint(quantized, hole_u8)
    return np.stack([_inpaint(quantized[:, :, index], hole_u8) for index in range(quantized.shape[2])], axis=2)
