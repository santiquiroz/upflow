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
from app.services.engines.onnx_common import (
    TILE_OVERLAP_PX,
    blend_tiles,
    detect_scale,
    finalize_uint8,
    from_nchw_float,
    get_cached_session,
    tile_starts,
    to_nchw_float,
    wrap_onnx_error,
)
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.process_runner import is_non_empty_file
from app.services.progress import apply_image_tile_progress

# ---------------------------------------------------------------------------
# ONNX Runtime DirectML upscaling engine (in-process, no subprocess).
#
# Session caching: onnxruntime InferenceSession objects are expensive to
# build (they load + optimize the whole graph), so sessions are cached by
# (model_id, device) with a small LRU(2) -- large enough to keep the
# currently-selected model warm across consecutive jobs on the same device
# without unbounded VRAM growth from every model the user has ever tried.
# The LRU lookup/build mechanics live in onnx_common.get_cached_session.
#
# GPU semaphore: this engine intentionally does NOT accept or manage its own
# asyncio.Semaphore. JobManager/VideoJobManager already wrap every
# `await self.engine.run(job)` call in
# `async with self.device_semaphores.acquire(job.device):` -- that gating is
# applied uniformly to whichever engine is plugged in, keyed by the job's own
# device, so once Task 7 wires OnnxUpscaler as `self.engine`, concurrency is
# gated for free. Adding a second semaphore here would be redundant and risks
# a deadlock if the two semaphores ever had different capacities.
#
# Tiling: ONNX_TILE_SIZE (default 256, 0 disables tiling) with a fixed 16px
# overlap. Each tile is inferred independently and stitched back with a
# linear-feather weighted blend across the overlap band (onnx_common's
# tile_starts/blend_tiles), so seams don't show up as hard edges for models
# with real receptive-field context. Upscale ratio is *not* read from static
# ONNX metadata (input/output shapes are frequently dynamic/symbolic there)
# -- it is derived from the concrete output array of the first inferred tile
# instead.
#
# Frame I/O is PIL + fp32 NCHW on purpose (arbitrary HF-installed graphs are
# fp32 NCHW); the video engine's cv2/uint8-NHWC fast path lives in
# onnx_video_upscaler -- measured decision, not duplication.
#
# Progress (SP5 Task 4): the tiled path reports tilesDone/tilesTotal into
# job.metadata (framesDone/framesTotal, reusing the video job progress
# fields the frontend already renders) between tile inferences -- this runs
# inside the worker thread from asyncio.to_thread, never on the event loop.
# The single-pass path (image fits in one tile) intentionally does NOT
# report tile progress: a tilesTotal=1 would be a fake sub-progress, so it
# stays on the coarse validating/upscaling stages instead.
# ---------------------------------------------------------------------------


def _load_rgb_array(source_path: Path) -> np.ndarray:
    with Image.open(source_path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def _save_rgb_array(array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(output_path)


class OnnxUpscaler(UpscaleEngine):
    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        devices: DevicesService,
        gpu_coordinator: GpuSessionCoordinator,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.devices = devices
        self.gpu_coordinator = gpu_coordinator
        self._session_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._session_lock = threading.Lock()

    def available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401
        except (ImportError, OSError):
            return False
        return True

    def release_device(self, device: str) -> None:
        # Cache is keyed by (model_id, device) -- a single device can hold
        # several model entries, so every key whose device matches must be
        # evicted, not just one.
        with self._session_lock:
            keys_to_remove = [key for key in self._session_cache if key[1] == device]
            for key in keys_to_remove:
                del self._session_cache[key]

    async def run(self, job: UpscaleJob) -> Path:
        # Only the in-memory registry lookup happens synchronously here.
        # available()/devices.validate() touch native libraries (onnxruntime
        # import, real DXGI adapter enumeration) and are deferred into the
        # same asyncio.to_thread call as the actual inference, so nothing
        # that can block on hardware ever runs on the event loop thread.
        entry = self._resolve_installed_entry(job.model_id)
        output_path = self._output_path(job)

        await asyncio.to_thread(self._run_and_save, job, entry, output_path)

        if not is_non_empty_file(output_path):
            raise RuntimeError("ONNX upscaling completed but no output file was produced")
        return output_path

    async def run_frames(self, frames_in: Path, frames_out: Path, model_id: str, device: str) -> Path:
        # Video pipeline contract (mirrors RifeNcnnEngine.run): frames_in/out
        # are directories of "%08d.png" frames. Every frame is upscaled
        # independently through the same cached session -- no cross-frame
        # state -- and the output frame count is validated against the input
        # count the same way RIFE validates its target frame count.
        entry = self._resolve_installed_entry(model_id)
        frames_out.mkdir(parents=True, exist_ok=True)
        source_frame_count = self._count_frame_files(frames_in)

        await asyncio.to_thread(self._run_frames_and_save, frames_in, frames_out, entry, device)

        self._validate_frame_output_count(frames_out, source_frame_count)
        return frames_out

    def _run_frames_and_save(self, frames_in: Path, frames_out: Path, entry: ModelEntry, device: str) -> None:
        if not self.available():
            raise RuntimeError("ONNX engine is not available: onnxruntime is not installed")
        self.devices.validate(device)
        session = self._get_session(entry.id, device, entry)
        for frame_path in sorted(frames_in.glob("*.png")):
            image = _load_rgb_array(frame_path)
            upscaled = self._upscale_array(session, image, self.settings.onnx_tile_size)
            _save_rgb_array(upscaled, frames_out / frame_path.name)

    @staticmethod
    def _count_frame_files(directory: Path) -> int:
        return sum(1 for _ in directory.glob("*.png"))

    def _validate_frame_output_count(self, frames_out: Path, expected_count: int) -> None:
        actual_count = self._count_frame_files(frames_out)
        if actual_count == 0:
            raise RuntimeError("ONNX frame upscaling completed but no output frames were produced")
        if actual_count != expected_count:
            raise RuntimeError(
                f"ONNX frame upscaling completed with {actual_count} frames, expected {expected_count}"
            )

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

    def _run_and_save(self, job: UpscaleJob, entry: ModelEntry, output_path: Path) -> None:
        if not self.available():
            raise RuntimeError("ONNX engine is not available: onnxruntime is not installed")
        self.devices.validate(job.device)
        image = _load_rgb_array(job.source_path)
        session = self._get_session(entry.id, job.device, entry)
        upscaled = self._upscale_array(session, image, self.settings.onnx_tile_size, job=job)
        _save_rgb_array(upscaled, output_path)

    def _get_session(self, model_id: str, device: str, entry: ModelEntry) -> Any:
        self.gpu_coordinator.acquire(device, self)
        return get_cached_session(
            self._session_cache,
            self._session_lock,
            (model_id, device),
            lambda: self._create_session(model_id, device, entry),
            f"Failed to load ONNX model {model_id!r} on device {device!r}",
        )

    def _create_session(self, model_id: str, device: str, entry: ModelEntry) -> Any:
        # Monkeypatchable seam: unit tests override this to inject a fake
        # numpy-based session and never touch real onnxruntime. Errors raised
        # here (including a missing onnxruntime import) are translated to a
        # clear RuntimeError by the caller, `_get_session`.
        from app.services import ep_registry

        model_path = self.settings.models_path / entry.file_path  # type: ignore[operator]
        return ep_registry.create_session(str(model_path), device, self.settings)

    def _upscale_array(
        self, session: Any, image: np.ndarray, tile_size: int, job: UpscaleJob | None = None
    ) -> np.ndarray:
        height, width, _ = image.shape
        if tile_size <= 0 or (height <= tile_size and width <= tile_size):
            # Single pass: no honest sub-progress to report (tilesTotal=1 would
            # be a fake ETA), so job is intentionally not threaded through here.
            return finalize_uint8(self._infer_tile(session, image))
        return self._upscale_tiled(session, image, tile_size, job)

    def _upscale_tiled(
        self, session: Any, image: np.ndarray, tile_size: int, job: UpscaleJob | None = None
    ) -> np.ndarray:
        height, width, channels = image.shape
        starts_y = tile_starts(height, tile_size, TILE_OVERLAP_PX)
        starts_x = tile_starts(width, tile_size, TILE_OVERLAP_PX)
        tiles_total = len(starts_y) * len(starts_x)

        tiles: list[tuple[int, int, int, int, np.ndarray]] = []
        for y0 in starts_y:
            for x0 in starts_x:
                tile_h = min(tile_size, height - y0)
                tile_w = min(tile_size, width - x0)
                source_tile = image[y0 : y0 + tile_h, x0 : x0 + tile_w]
                output_tile = self._infer_tile(session, source_tile)
                tiles.append((y0, x0, tile_h, tile_w, output_tile))
                self._report_tile_progress(job, len(tiles), tiles_total)

        _, _, first_h, first_w, first_out = tiles[0]
        scale = detect_scale(first_h, first_w, first_out)
        return blend_tiles(tiles, height, width, channels, scale)

    @staticmethod
    def _report_tile_progress(job: UpscaleJob | None, tiles_done: int, tiles_total: int) -> None:
        # job is None for algorithm-level callers (tests, and the video
        # per-frame ONNX path in run_frames) that don't want tile-level
        # progress mixed into a job's metadata.
        if job is None:
            return
        apply_image_tile_progress(job, tiles_done, tiles_total)

    def _infer_tile(self, session: Any, tile_rgb: np.ndarray) -> np.ndarray:
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        batch = to_nchw_float(tile_rgb)
        try:
            result = session.run([output_info.name], {input_info.name: batch})[0]
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise wrap_onnx_error("ONNX inference failed", exc) from exc
        return from_nchw_float(result)
