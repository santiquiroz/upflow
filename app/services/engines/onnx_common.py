from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np

from app.services.dml_device import try_parse_dml_device_id

# ---------------------------------------------------------------------------
# Shared ONNX Runtime infrastructure for every in-process ONNX engine
# (OnnxUpscaler, OnnxVideoUpscaler, GmfssEngine, model_installer validation,
# generation, the CPU-fallback probe): provider lists, error translation, the
# LRU session-cache pattern, and the tile geometry + feathered blending math.
#
# Extracted from engines/onnx_upscaler.py (consolidation roadmap #3): 8+
# modules were importing its `_`-private helpers, so the shared surface now
# lives here with public names. Engine-specific I/O stays with each engine --
# PIL/fp32-NCHW in onnx_upscaler vs cv2/uint8-NHWC-IOBinding in
# onnx_video_upscaler is a measured, documented decision, not duplication.
# ---------------------------------------------------------------------------

SESSION_CACHE_SIZE = 2
TILE_OVERLAP_PX = 16
CPU_PROVIDER = "CPUExecutionProvider"
DML_PROVIDER = "DmlExecutionProvider"


def build_providers(device: str) -> list[str | tuple[str, dict[str, int]]]:
    if device == "cpu":
        return [CPU_PROVIDER]
    device_id = try_parse_dml_device_id(device)
    if device_id is not None:
        return [(DML_PROVIDER, {"device_id": device_id}), CPU_PROVIDER]
    raise RuntimeError(f"Unsupported device for ONNX inference: {device!r}")


def wrap_onnx_error(context: str, exc: Exception) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()
    if any(token in lowered for token in ("memory", "alloc", "oom")):
        return RuntimeError(f"{context}: insufficient GPU/VRAM memory ({message})")
    return RuntimeError(f"{context}: {message}")


def get_cached_session(
    cache: OrderedDict[Any, Any],
    lock: threading.Lock,
    cache_key: Any,
    build_session: Callable[[], Any],
    error_context: str,
    cache_size: int = SESSION_CACHE_SIZE,
) -> Any:
    """LRU lookup-or-build shared by every engine's `_get_session`.

    Session creation (expensive I/O + graph load) happens outside the lock so
    it never blocks other threads' cache lookups; a rare race where two
    threads miss the same key concurrently just builds the session twice
    (last insert wins), which is wasteful but not corrupting -- preferred
    over holding the lock across a slow load. Errors are translated here (not
    inside each engine's `_create_session` seam) so the clear-error guarantee
    also covers test doubles that replace the seam outright.
    """
    with lock:
        cached = cache.get(cache_key)
        if cached is not None:
            cache.move_to_end(cache_key)
            return cached

    try:
        session = build_session()
    except Exception as exc:  # onnxruntime raises its own native exception types
        raise wrap_onnx_error(error_context, exc) from exc

    with lock:
        cache[cache_key] = session
        cache.move_to_end(cache_key)
        if len(cache) > cache_size:
            cache.popitem(last=False)
    return session


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile <= 0 or length <= tile:
        return [0]
    # Guards against a pathological ONNX_TILE_SIZE smaller than the overlap
    # (never happens with the real default of 256 vs. 16px overlap, but a
    # negative/zero step would otherwise infinite-loop `range`).
    step = max(1, tile - overlap)
    starts = list(range(0, length - tile + 1, step))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def detect_scale(tile_h: int, tile_w: int, output_tile: np.ndarray) -> int:
    out_h, out_w = output_tile.shape[0], output_tile.shape[1]
    if tile_h <= 0 or tile_w <= 0 or out_h % tile_h != 0 or out_w % tile_w != 0:
        raise RuntimeError(
            f"Could not detect an integer upscale ratio from ONNX output shape "
            f"(input {tile_h}x{tile_w} -> output {out_h}x{out_w})"
        )
    scale_h, scale_w = out_h // tile_h, out_w // tile_w
    if scale_h != scale_w:
        raise RuntimeError(f"ONNX model produced a non-uniform scale: {scale_h}x vs {scale_w}x")
    return scale_h


def _axis_weights(length: int, feather: int, is_start_edge: bool, is_end_edge: bool) -> np.ndarray:
    weights = np.ones(length, dtype=np.float32)
    feather = min(feather, length // 2) if length > 0 else 0
    if feather <= 0:
        return weights
    ramp = np.arange(1, feather + 1, dtype=np.float32) / (feather + 1)
    if not is_start_edge:
        weights[:feather] = ramp
    if not is_end_edge:
        weights[-feather:] = ramp[::-1]
    return weights


def tile_weights(
    out_h: int, out_w: int, feather: int, is_top: bool, is_bottom: bool, is_left: bool, is_right: bool
) -> np.ndarray:
    vertical = _axis_weights(out_h, feather, is_top, is_bottom)
    horizontal = _axis_weights(out_w, feather, is_left, is_right)
    return (vertical[:, None] * horizontal[None, :])[:, :, None]


def to_nchw_float(tile_rgb: np.ndarray) -> np.ndarray:
    normalized = tile_rgb.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]


def from_nchw_float(output: np.ndarray) -> np.ndarray:
    array = np.transpose(output[0], (1, 2, 0))
    return np.clip(array * 255.0, 0.0, 255.0)


def finalize_uint8(array: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(array, 0, 255)).astype(np.uint8)


def blend_tiles(
    tiles: list[tuple[int, int, int, int, np.ndarray]],
    height: int,
    width: int,
    channels: int,
    scale: int,
) -> np.ndarray:
    """Stitch inferred tiles (y0, x0, tile_h, tile_w, output_tile) back into a
    single HWC uint8 canvas with a linear-feather weighted blend across the
    TILE_OVERLAP_PX overlap band, so seams don't show up as hard edges.

    One algorithm for both engines: onnx_upscaler feeds float32 tiles,
    onnx_video_upscaler feeds uint8 tiles -- the accumulator is float32 either
    way.
    """
    canvas_h, canvas_w = height * scale, width * scale
    accumulator = np.zeros((canvas_h, canvas_w, channels), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
    feather = scale * TILE_OVERLAP_PX
    for y0, x0, tile_h, tile_w, output_tile in tiles:
        out_h, out_w = tile_h * scale, tile_w * scale
        weights = tile_weights(
            out_h,
            out_w,
            feather,
            is_top=(y0 == 0),
            is_bottom=(y0 + tile_h == height),
            is_left=(x0 == 0),
            is_right=(x0 + tile_w == width),
        )
        oy, ox = y0 * scale, x0 * scale
        accumulator[oy : oy + out_h, ox : ox + out_w] += (
            output_tile.astype(np.float32, copy=False) * weights
        )
        weight_sum[oy : oy + out_h, ox : ox + out_w] += weights
    blended = accumulator / np.clip(weight_sum, 1e-6, None)
    return finalize_uint8(blended)
