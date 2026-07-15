from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import Settings
from app.models import UpscaleJob
from app.services.devices_service import DevicesService
from app.services.engines.base import UpscaleEngine
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus

# ---------------------------------------------------------------------------
# ONNX Runtime DirectML upscaling engine (in-process, no subprocess).
#
# Session caching: onnxruntime InferenceSession objects are expensive to
# build (they load + optimize the whole graph), so sessions are cached by
# (model_id, device) with a small LRU(2) -- large enough to keep the
# currently-selected model warm across consecutive jobs on the same device
# without unbounded VRAM growth from every model the user has ever tried.
#
# GPU semaphore: this engine intentionally does NOT accept or manage its own
# asyncio.Semaphore. JobManager/VideoJobManager already wrap every
# `await self.engine.run(job)` call in `async with self.gpu_semaphore:` --
# that gating is applied uniformly to whichever engine is plugged in, so once
# Task 7 wires OnnxUpscaler as `self.engine`, concurrency is gated for free.
# Adding a second semaphore here would be redundant and risks a deadlock if
# the two semaphores ever had different capacities.
#
# Tiling: ONNX_TILE_SIZE (default 256, 0 disables tiling) with a fixed 16px
# overlap. Each tile is inferred independently and stitched back with a
# linear-feather weighted blend across the overlap band, so seams don't show
# up as hard edges for models with real receptive-field context. Upscale
# ratio is *not* read from static ONNX metadata (input/output shapes are
# frequently dynamic/symbolic there) -- it is derived from the concrete
# output array of the first inferred tile instead.
# ---------------------------------------------------------------------------

SESSION_CACHE_SIZE = 2
TILE_OVERLAP_PX = 16
CPU_PROVIDER = "CPUExecutionProvider"
DML_PROVIDER = "DmlExecutionProvider"


def _parse_dml_device_id(device: str) -> int:
    _, _, suffix = device.partition(":")
    try:
        return int(suffix)
    except ValueError as exc:
        raise RuntimeError(f"Unsupported device for ONNX inference: {device!r}") from exc


def _build_providers(device: str) -> list[str | tuple[str, dict[str, int]]]:
    if device == "cpu":
        return [CPU_PROVIDER]
    if device.startswith("dml:"):
        device_id = _parse_dml_device_id(device)
        return [(DML_PROVIDER, {"device_id": device_id}), CPU_PROVIDER]
    raise RuntimeError(f"Unsupported device for ONNX inference: {device!r}")


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
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


def _detect_scale(tile_h: int, tile_w: int, output_tile: np.ndarray) -> int:
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


def _tile_weights(
    out_h: int, out_w: int, feather: int, is_top: bool, is_bottom: bool, is_left: bool, is_right: bool
) -> np.ndarray:
    vertical = _axis_weights(out_h, feather, is_top, is_bottom)
    horizontal = _axis_weights(out_w, feather, is_left, is_right)
    return (vertical[:, None] * horizontal[None, :])[:, :, None]


def _load_rgb_array(source_path: Path) -> np.ndarray:
    with Image.open(source_path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def _save_rgb_array(array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(output_path)


def _to_nchw_float(tile_rgb: np.ndarray) -> np.ndarray:
    normalized = tile_rgb.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]


def _from_nchw_float(output: np.ndarray) -> np.ndarray:
    array = np.transpose(output[0], (1, 2, 0))
    return np.clip(array * 255.0, 0.0, 255.0)


def _finalize_uint8(array: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(array, 0, 255)).astype(np.uint8)


def _wrap_onnx_error(context: str, exc: Exception) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()
    if any(token in lowered for token in ("memory", "alloc", "oom")):
        return RuntimeError(f"{context}: insufficient GPU/VRAM memory ({message})")
    return RuntimeError(f"{context}: {message}")


class OnnxUpscaler(UpscaleEngine):
    def __init__(self, settings: Settings, registry: ModelRegistry, devices: DevicesService) -> None:
        self.settings = settings
        self.registry = registry
        self.devices = devices
        self._session_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._session_lock = threading.Lock()

    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except (ImportError, OSError):
            return False
        return True

    async def run(self, job: UpscaleJob) -> Path:
        # Only the in-memory registry lookup happens synchronously here.
        # available()/devices.validate() touch native libraries (onnxruntime
        # import, real DXGI adapter enumeration) and are deferred into the
        # same asyncio.to_thread call as the actual inference, so nothing
        # that can block on hardware ever runs on the event loop thread.
        entry = self._resolve_installed_entry(job.model_id)
        output_path = self._output_path(job)

        await asyncio.to_thread(self._run_and_save, job.source_path, entry, job.device, output_path)

        if not self._is_non_empty_file(output_path):
            raise RuntimeError("ONNX upscaling completed but no output file was produced")
        return output_path

    def _output_path(self, job: UpscaleJob) -> Path:
        return self.settings.outputs_path / f"{job.id}.{job.output_format.lower()}"

    def _resolve_installed_entry(self, model_id: str) -> ModelEntry:
        entry = self.registry.get(model_id)
        if entry is None:
            raise RuntimeError(f"Unknown ONNX model id: {model_id!r}")
        if entry.kind != ModelKind.onnx:
            raise RuntimeError(f"Model {model_id!r} is not an ONNX model (kind={entry.kind.value})")
        if entry.status != ModelStatus.installed or entry.file_path is None:
            raise RuntimeError(f"Model {model_id!r} is not ready for inference (status={entry.status.value})")
        return entry

    def _run_and_save(self, source_path: Path, entry: ModelEntry, device: str, output_path: Path) -> None:
        if not self.available():
            raise RuntimeError("ONNX engine is not available: onnxruntime is not installed")
        self.devices.validate(device)
        image = _load_rgb_array(source_path)
        session = self._get_session(entry.id, device, entry)
        upscaled = self._upscale_array(session, image, self.settings.onnx_tile_size)
        _save_rgb_array(upscaled, output_path)

    def _get_session(self, model_id: str, device: str, entry: ModelEntry) -> Any:
        cache_key = (model_id, device)
        with self._session_lock:
            cached = self._session_cache.get(cache_key)
            if cached is not None:
                self._session_cache.move_to_end(cache_key)
                return cached

        # Session creation (expensive I/O + graph load) happens outside the
        # lock so it never blocks other threads' cache lookups; a rare race
        # where two threads miss the same key concurrently just builds the
        # session twice (last insert wins), which is wasteful but not
        # corrupting -- preferred over holding the lock across a slow load.
        # Errors are translated here (not inside `_create_session`) so the
        # clear-error guarantee also covers test doubles that replace the
        # seam outright.
        try:
            session = self._create_session(model_id, device, entry)
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise _wrap_onnx_error(f"Failed to load ONNX model {model_id!r} on device {device!r}", exc) from exc

        with self._session_lock:
            self._session_cache[cache_key] = session
            self._session_cache.move_to_end(cache_key)
            if len(self._session_cache) > SESSION_CACHE_SIZE:
                self._session_cache.popitem(last=False)
        return session

    def _create_session(self, model_id: str, device: str, entry: ModelEntry) -> Any:
        # Monkeypatchable seam: unit tests override this to inject a fake
        # numpy-based session and never touch real onnxruntime. Errors raised
        # here (including a missing onnxruntime import) are translated to a
        # clear RuntimeError by the caller, `_get_session`.
        import onnxruntime as ort

        providers = _build_providers(device)
        model_path = self.settings.models_path / entry.file_path  # type: ignore[operator]
        return ort.InferenceSession(str(model_path), providers=providers)

    def _upscale_array(self, session: Any, image: np.ndarray, tile_size: int) -> np.ndarray:
        height, width, _ = image.shape
        if tile_size <= 0 or (height <= tile_size and width <= tile_size):
            return _finalize_uint8(self._infer_tile(session, image))
        return self._upscale_tiled(session, image, tile_size)

    def _upscale_tiled(self, session: Any, image: np.ndarray, tile_size: int) -> np.ndarray:
        height, width, channels = image.shape
        starts_y = _tile_starts(height, tile_size, TILE_OVERLAP_PX)
        starts_x = _tile_starts(width, tile_size, TILE_OVERLAP_PX)

        tiles: list[tuple[int, int, int, int, np.ndarray]] = []
        for y0 in starts_y:
            for x0 in starts_x:
                tile_h = min(tile_size, height - y0)
                tile_w = min(tile_size, width - x0)
                source_tile = image[y0 : y0 + tile_h, x0 : x0 + tile_w]
                output_tile = self._infer_tile(session, source_tile)
                tiles.append((y0, x0, tile_h, tile_w, output_tile))

        _, _, first_h, first_w, first_out = tiles[0]
        scale = _detect_scale(first_h, first_w, first_out)
        canvas_h, canvas_w = height * scale, width * scale
        accumulator = np.zeros((canvas_h, canvas_w, channels), dtype=np.float32)
        weight_sum = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
        feather = scale * TILE_OVERLAP_PX

        for y0, x0, tile_h, tile_w, output_tile in tiles:
            out_h, out_w = tile_h * scale, tile_w * scale
            weights = _tile_weights(
                out_h,
                out_w,
                feather,
                is_top=(y0 == 0),
                is_bottom=(y0 + tile_h == height),
                is_left=(x0 == 0),
                is_right=(x0 + tile_w == width),
            )
            oy, ox = y0 * scale, x0 * scale
            accumulator[oy : oy + out_h, ox : ox + out_w] += output_tile * weights
            weight_sum[oy : oy + out_h, ox : ox + out_w] += weights

        blended = accumulator / np.clip(weight_sum, 1e-6, None)
        return _finalize_uint8(blended)

    def _infer_tile(self, session: Any, tile_rgb: np.ndarray) -> np.ndarray:
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        batch = _to_nchw_float(tile_rgb)
        try:
            result = session.run([output_info.name], {input_info.name: batch})[0]
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise _wrap_onnx_error("ONNX inference failed", exc) from exc
        return _from_nchw_float(result)

    @staticmethod
    def _is_non_empty_file(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0
