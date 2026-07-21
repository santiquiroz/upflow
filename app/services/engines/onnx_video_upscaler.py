from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Blocking queue put/join poll interval: short enough that a cancel is observed
# almost immediately, long enough not to busy-spin.
_QUEUE_POLL_SECONDS = 0.2
_THREAD_JOIN_TIMEOUT_SECONDS = 30.0


def _put_until_cancelled(
    q: queue.Queue, item: Any, cancel_event: threading.Event, timeout: float = _QUEUE_POLL_SECONDS
) -> bool:
    """Enqueue `item`, re-checking `cancel_event` while the queue is full.

    A plain blocking `queue.put()` cannot be interrupted, so on cancel a loader
    blocked on a full queue would hang forever (its only consumer stops draining)
    and leak the worker thread. Returns True once enqueued, False if cancelled
    before a slot frees.
    """
    while not cancel_event.is_set():
        try:
            q.put(item, timeout=timeout)
            return True
        except queue.Full:
            continue
    return False


def _drain_queue(q: queue.Queue) -> None:
    """Discard everything currently queued so a producer blocked on a full queue
    can complete its put and exit. Used during pipeline teardown."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


_OOM_SIGNATURES = (
    "out of memory",
    "outofmemory",
    "failed to allocate",
    "insufficient",
    "d3d12",  # DirectML allocation failures surface with D3D12 device-removed/hung text
    "device removed",
    "device hung",
    "cudamalloc",
)


def _is_oom_error(exc: BaseException) -> bool:
    """True if an inference exception looks like a GPU memory / allocation failure,
    the case where retrying the same frame tiled (smaller allocations) can succeed."""
    text = str(exc).lower()
    return any(sig in text for sig in _OOM_SIGNATURES)

from app.config import Settings
from app.services.backend_registry import get_builtin_onnx_model
from app.services.devices_service import DevicesService
from app.services.engines.onnx_upscaler import (
    SESSION_CACHE_SIZE,
    TILE_OVERLAP_PX,
    _build_providers,
    _detect_scale,
    _finalize_uint8,
    _parse_dml_device_id,
    _tile_starts,
    _tile_weights,
    _wrap_onnx_error,
)
from app.services.gpu_session_coordinator import GpuSessionCoordinator

# ---------------------------------------------------------------------------
# SP11 - optimized ONNX Runtime video frame engine.
#
# The speed comes from four things the prototype benchmark proved (RX 7800 XT,
# animevideov3-x4, 720p 4x: NCNN 5.4 fps -> 11.49 fps, 2.1x):
#   1. uint8-in/out ONNX graph: /255, NCHW, *255, clamp/round baked INTO the
#      graph, so a frame is a raw uint8 NHWC array in and out -- no per-frame
#      numpy pre/post (147ms) and no fp32 readback (177MB/frame).
#   2. Whole-frame inference (NO tiling) when the frame fits VRAM -- tiling on
#      DirectML was catastrophic (1.26 fps). Tiling is only a fallback for
#      huge frames (heuristic on input pixels).
#   3. IO binding: input/output OrtValue bound on the device (dml).
#   4. A threaded load(N+1)/infer(N)/save(N-1) pipeline with OpenCV PNG I/O
#      (cv2 is ~50x faster than PIL for large frames), sustaining ~2x NCNN.
#
# This engine runs BUILTIN Real-ESRGAN models from their vendored uint8 ONNX
# exports. HF-installed arbitrary ONNX models keep using OnnxUpscaler (their
# graphs are fp32 NCHW, not uint8 NHWC). GPU concurrency is gated by the
# caller's DeviceSemaphores (same as OnnxUpscaler) -- this engine holds no
# semaphore of its own.
# ---------------------------------------------------------------------------

DML_DEVICE_PREFIX = "dml:"
GPU_EXECUTION_PROVIDERS = frozenset(
    {"DmlExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"}
)
FALLBACK_TILE_SIZE = 512


def should_tile_frame(input_pixels: int, max_whole_frame_pixels: int) -> bool:
    """Whole-frame is the fast path; tile only when the frame is too big.

    `max_whole_frame_pixels <= 0` disables tiling entirely (always whole-frame),
    matching ONNX_TILE_SIZE=0's "no tiling" convention.
    """
    if max_whole_frame_pixels <= 0:
        return False
    return input_pixels > max_whole_frame_pixels


def _load_frame(source_path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read frame {source_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)[np.newaxis, ...]  # NHWC uint8 [1,H,W,3]


def _save_frame(frame_nhwc: np.ndarray, output_path: Path, png_compression: int) -> None:
    import cv2

    bgr = cv2.cvtColor(frame_nhwc[0], cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(output_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)])
    if not ok:
        raise RuntimeError(f"Failed to write frame {output_path}")


class OnnxVideoUpscaler:
    def __init__(
        self,
        settings: Settings,
        registry: Any,
        devices: DevicesService,
        gpu_coordinator: GpuSessionCoordinator,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.devices = devices
        self.gpu_coordinator = gpu_coordinator
        self._session_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._session_lock = threading.Lock()
        self._gpu_ep_cache: bool | None = None
        self._iobinding_warned = False

    # --- capability probes -------------------------------------------------

    def available(self) -> bool:
        return self._onnxruntime_available() and self._opencv_available()

    def release_device(self, device: str) -> None:
        # Cache is keyed by (model_path, device) -- a single device can hold
        # several model entries, so every key whose device matches must be
        # evicted, not just one.
        with self._session_lock:
            keys_to_remove = [key for key in self._session_cache if key[1] == device]
            for key in keys_to_remove:
                del self._session_cache[key]

    @staticmethod
    def _onnxruntime_available() -> bool:
        try:
            import onnxruntime  # noqa: F401
        except (ImportError, OSError):
            return False
        return True

    @staticmethod
    def _opencv_available() -> bool:
        try:
            import cv2  # noqa: F401
        except (ImportError, OSError):
            return False
        return True

    def has_gpu_execution_provider(self) -> bool:
        if self._gpu_ep_cache is None:
            self._gpu_ep_cache = self._probe_gpu_execution_provider()
        return self._gpu_ep_cache

    @staticmethod
    def _probe_gpu_execution_provider() -> bool:
        try:
            import onnxruntime as ort
        except (ImportError, OSError):
            return False
        return bool(set(ort.get_available_providers()) & GPU_EXECUTION_PROVIDERS)

    def builtin_onnx_available(self, engine_model_name: str) -> bool:
        model = get_builtin_onnx_model(engine_model_name)
        if model is None:
            return False
        return (self.settings.builtin_onnx_path / model.filename).exists()

    def _select_model_file(self, model: Any, device: str) -> Path:
        """Prefer the fp16 export on GPU when enabled and present; the fp32 file is
        the fallback (and the only sane choice on the CPU EP, where fp16 is
        emulated). A missing fp16 sibling silently uses fp32."""
        onnx_dir = self.settings.builtin_onnx_path
        if self.settings.onnx_prefer_fp16 and not device.startswith("cpu"):
            fp16_path = onnx_dir / model.fp16_filename
            if fp16_path.exists():
                return fp16_path
        return onnx_dir / model.filename

    # --- public video entry point ------------------------------------------

    async def run_frames_builtin(
        self, frames_in: Path, frames_out: Path, engine_model_name: str, device: str
    ) -> Path:
        # Mirrors OnnxUpscaler.run_frames/RifeNcnnEngine.run: frames_in/out are
        # "%08d.png" directories; the whole run happens off the event loop in a
        # worker thread. A threading.Event makes the (otherwise un-cancellable)
        # to_thread pipeline cooperatively stoppable so per-job cancel and the
        # stall watchdog can actually kill it.
        model = get_builtin_onnx_model(engine_model_name)
        if model is None:
            raise RuntimeError(f"No ONNX export configured for builtin model {engine_model_name!r}")
        onnx_path = self._select_model_file(model, device)
        frames_out.mkdir(parents=True, exist_ok=True)
        source_frame_count = self._count_frame_files(frames_in)

        cancel_event = threading.Event()
        # Shield the worker so a cancel doesn't abandon it mid-flight: on cancel
        # we signal cancel_event and then WAIT for the worker to fully unwind
        # (all pipeline threads joined, no more cv2 I/O against frames_in/out)
        # before propagating. Otherwise the caller's work_dir cleanup would race
        # live file writes -- the ncnn path already waits for its subprocess to
        # die (process_runner), and this gives the onnx path the same guarantee.
        worker = asyncio.ensure_future(
            asyncio.to_thread(
                self._run_frames_blocking, frames_in, frames_out, onnx_path, device, cancel_event, model.scale
            )
        )
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancel_event.set()
            with contextlib.suppress(BaseException):
                await worker
            raise

        self._validate_frame_output_count(frames_out, source_frame_count)
        return frames_out

    # --- streaming (raw-pipe) entry point ----------------------------------

    async def run_frames_streaming(
        self,
        frames_in: Path,
        engine_model_name: str,
        device: str,
        write_frame: "Callable[[np.ndarray], None]",
    ) -> int:
        """Upscale frames and hand each RGB HWC uint8 frame, IN ORDER, to
        write_frame -- no PNG round-trip to disk. Returns the frame count.

        Cancel-safe with the same shield+await pattern as run_frames_builtin: on
        cancel we signal, then wait for the worker to unwind before propagating so
        the caller can tear down its ffmpeg process without racing a live write.
        """
        model = get_builtin_onnx_model(engine_model_name)
        if model is None:
            raise RuntimeError(f"No ONNX export configured for builtin model {engine_model_name!r}")
        onnx_path = self._select_model_file(model, device)

        cancel_event = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(
                self._run_streaming_blocking, frames_in, onnx_path, device, write_frame, cancel_event
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancel_event.set()
            with contextlib.suppress(BaseException):
                await worker
            raise

    def _run_streaming_blocking(
        self,
        frames_in: Path,
        onnx_path: Path,
        device: str,
        write_frame: "Callable[[np.ndarray], None]",
        cancel_event: threading.Event,
    ) -> int:
        if not self.available():
            raise RuntimeError("ONNX video engine is not available: onnxruntime and opencv are required")
        if not onnx_path.exists():
            raise RuntimeError(f"ONNX model file not found: {onnx_path}")
        self.devices.validate(device)
        session = self._get_session(str(onnx_path), device)
        frame_paths = sorted(frames_in.glob("*.png"))
        self._run_streaming_pipeline(session, frame_paths, device, write_frame, cancel_event)
        return len(frame_paths)

    def _run_streaming_pipeline(
        self,
        session: Any,
        frame_paths: list[Path],
        device: str,
        write_frame: "Callable[[np.ndarray], None]",
        cancel_event: threading.Event,
    ) -> None:
        # Exactly one loader so frames load + infer in strict index order and the
        # writer never has to hold more than the in-flight frame (the reorder
        # buffer below is a safety net, not the normal path). Infer is the limiter
        # (~116ms/frame), so a single ~30ms loader is not a bottleneck.
        load_q: queue.Queue[tuple[str, np.ndarray]] = queue.Queue(maxsize=4)
        save_q: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue(maxsize=4)
        todo: queue.Queue[Path] = queue.Queue()
        for path in frame_paths:
            todo.put(path)
        errors: list[Exception] = []

        loader = threading.Thread(
            target=self._loader_loop, args=(todo, load_q, errors, cancel_event), daemon=True
        )
        writer = threading.Thread(
            target=self._ordered_writer_loop,
            args=(save_q, write_frame, len(frame_paths), errors, cancel_event),
            daemon=True,
        )
        loader.start()
        writer.start()
        try:
            self._infer_loop(session, load_q, save_q, device, len(frame_paths), [loader], errors, cancel_event)
        finally:
            _drain_queue(load_q)
            save_q.put(None)  # wake the writer if infer stopped early (error/cancel)
            for thread in (writer, loader):
                thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    logger.error("onnx streaming thread did not stop within timeout: %s", thread.name)
        if errors:
            raise errors[0]

    @staticmethod
    def _ordered_writer_loop(
        save_q: queue.Queue[tuple[str, np.ndarray] | None],
        write_frame: "Callable[[np.ndarray], None]",
        total: int,
        errors: list[Exception],
        cancel_event: threading.Event,
    ) -> None:
        # Emit frames strictly in index order (00000001, 00000002, ...). With a
        # single loader they already arrive in order; the pending map only ever
        # holds the current frame, but it makes the contract robust to reordering.
        pending: dict[int, np.ndarray] = {}
        next_index = 1
        written = 0
        while written < total:
            if cancel_event.is_set():
                return
            try:
                item = save_q.get(timeout=0.2)
            except queue.Empty:
                if cancel_event.is_set() or errors:
                    return
                continue
            if item is None:
                return  # sentinel: infer stopped early
            name, frame = item
            try:
                pending[int(Path(name).stem)] = frame
            except ValueError as exc:  # a frame name that isn't an index
                errors.append(exc)
                cancel_event.set()
                return
            while next_index in pending:
                try:
                    write_frame(pending.pop(next_index)[0])  # strip batch -> HWC RGB uint8
                except Exception as exc:  # noqa: BLE001 -- ffmpeg died / broken pipe
                    errors.append(exc)
                    cancel_event.set()
                    return
                next_index += 1
                written += 1

    def _run_frames_blocking(
        self,
        frames_in: Path,
        frames_out: Path,
        onnx_path: Path,
        device: str,
        cancel_event: threading.Event,
        scale: int,
    ) -> None:
        if not self.available():
            raise RuntimeError("ONNX video engine is not available: onnxruntime and opencv are required")
        if not onnx_path.exists():
            raise RuntimeError(f"ONNX model file not found: {onnx_path}")
        self.devices.validate(device)
        session = self._get_session(str(onnx_path), device)
        frame_paths = sorted(frames_in.glob("*.png"))
        self._run_pipeline(session, frame_paths, frames_out, device, cancel_event, scale)

    # --- threaded load/infer/save pipeline ---------------------------------

    def _save_queue_maxsize(self, frame_paths: list[Path], scale: int, n_save: int) -> int:
        """Bound the save queue by a RAM budget instead of by thread count.

        Each queued item is a full 4x output frame (~44MB @ 5120x2880). Sizing by
        n_save*2 alone lets the queue hold ~1GB with no relation to memory, so we
        derive maxsize from ONNX_VIDEO_MAX_PIPELINE_MB and the real output frame
        size, with a floor of n_save so savers never starve.
        """
        default = n_save * 2
        if not frame_paths or scale < 1:
            return default
        out_bytes = self._output_frame_bytes(frame_paths[0], scale)
        if out_bytes <= 0:
            return default
        budget_bytes = max(1, self.settings.onnx_video_max_pipeline_mb) * 1024 * 1024
        return max(n_save, min(default, budget_bytes // out_bytes))

    @staticmethod
    def _output_frame_bytes(frame_path: Path, scale: int) -> int:
        # One extra decode of a single frame (negligible) to learn input dims;
        # the output frame is scale x larger per axis, 3 bytes/px (uint8 BGR).
        image = _load_frame(frame_path)
        _, height, width, channels = image.shape
        return height * scale * width * scale * channels

    def _run_pipeline(
        self,
        session: Any,
        frame_paths: list[Path],
        frames_out: Path,
        device: str,
        cancel_event: threading.Event,
        scale: int = 1,
    ) -> None:
        n_load = max(1, self.settings.onnx_video_load_threads)
        n_save = max(1, self.settings.onnx_video_save_threads)
        png_level = self.settings.onnx_video_png_compression
        save_maxsize = self._save_queue_maxsize(frame_paths, scale, n_save)
        load_q: queue.Queue[tuple[str, np.ndarray]] = queue.Queue(maxsize=n_load * 2)
        save_q: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue(maxsize=save_maxsize)
        todo: queue.Queue[Path] = queue.Queue()
        for path in frame_paths:
            todo.put(path)
        errors: list[Exception] = []

        loaders = [
            threading.Thread(target=self._loader_loop, args=(todo, load_q, errors, cancel_event), daemon=True)
            for _ in range(n_load)
        ]
        savers = [
            threading.Thread(
                target=self._saver_loop, args=(save_q, frames_out, png_level, errors, cancel_event), daemon=True
            )
            for _ in range(n_save)
        ]
        for thread in loaders + savers:
            thread.start()

        try:
            self._infer_loop(session, load_q, save_q, device, len(frame_paths), loaders, errors, cancel_event)
        finally:
            # Drain any frames still queued so a loader blocked on a full load_q
            # (no longer consumed once infer returns) can finish its put and exit
            # instead of hanging the join below forever.
            _drain_queue(load_q)
            for _ in savers:
                save_q.put(None)  # sentinel: savers are draining, so this never blocks for long
            for thread in savers + loaders:
                thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    logger.error("onnx video pipeline thread did not stop within timeout: %s", thread.name)

        if errors:
            raise errors[0]

    @staticmethod
    def _loader_loop(
        todo: queue.Queue[Path],
        load_q: queue.Queue[tuple[str, np.ndarray]],
        errors: list[Exception],
        cancel_event: threading.Event,
    ) -> None:
        while not cancel_event.is_set():
            try:
                path = todo.get_nowait()
            except queue.Empty:
                return
            try:
                item = (path.name, _load_frame(path))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                cancel_event.set()
                return
            if not _put_until_cancelled(load_q, item, cancel_event):
                return

    @staticmethod
    def _saver_loop(
        save_q: queue.Queue[tuple[str, np.ndarray] | None],
        frames_out: Path,
        png_level: int,
        errors: list[Exception],
        cancel_event: threading.Event,
    ) -> None:
        while True:
            item = save_q.get()
            if item is None:
                save_q.task_done()
                return
            name, frame = item
            try:
                _save_frame(frame, frames_out / name, png_level)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                cancel_event.set()
            finally:
                save_q.task_done()

    def _infer_loop(
        self,
        session: Any,
        load_q: queue.Queue[tuple[str, np.ndarray]],
        save_q: queue.Queue[tuple[str, np.ndarray] | None],
        device: str,
        total: int,
        loaders: list[threading.Thread],
        errors: list[Exception],
        cancel_event: threading.Event,
    ) -> None:
        processed = 0
        force_tiled = False  # sticky per-run: once a whole-frame OOM forces tiling, stay tiled
        while processed < total:
            if cancel_event.is_set():
                return
            try:
                name, frame = load_q.get(timeout=0.2)
            except queue.Empty:
                # No frame ready: bail if we've been cancelled/errored, or if
                # every loader finished and nothing more is coming (a loader
                # died mid-run) -- prevents a deadlock waiting on a frame that
                # will never arrive.
                if cancel_event.is_set() or errors:
                    return
                if load_q.empty() and all(not thread.is_alive() for thread in loaders):
                    return
                continue
            try:
                upscaled, force_tiled = self._upscale_one(session, frame, device, force_tiled)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                cancel_event.set()
                return
            if not _put_until_cancelled(save_q, (name, upscaled), cancel_event):
                return
            processed += 1

    # --- inference ---------------------------------------------------------

    def _upscale_one(
        self, session: Any, frame_nhwc: np.ndarray, device: str, force_tiled: bool = False
    ) -> tuple[np.ndarray, bool]:
        """Upscale one frame; returns (frame, force_tiled_for_rest_of_job).

        A whole-frame OOM aborts the whole job today. Instead, catch an OOM-like
        error, retry THIS frame tiled, and stick to tiling for the rest of the run
        (memoized via the returned flag) so a low-VRAM job finishes slow instead
        of crashing.
        """
        _, height, width, _ = frame_nhwc.shape
        if force_tiled or should_tile_frame(height * width, self.settings.onnx_whole_frame_max_pixels):
            return self._infer_tiled(session, frame_nhwc, device), force_tiled
        try:
            return self._infer_frame(session, frame_nhwc, device), False
        except Exception as exc:  # noqa: BLE001
            if not _is_oom_error(exc):
                raise
            logger.warning(
                "whole-frame ONNX inference hit an OOM-like error on %s; falling back to tiling "
                "for the rest of this job",
                device,
                exc_info=True,
            )
            return self._infer_tiled(session, frame_nhwc, device), True

    def _infer_frame(self, session: Any, frame_nhwc: np.ndarray, device: str) -> np.ndarray:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        if device.startswith(DML_DEVICE_PREFIX):
            bound = self._infer_iobinding(session, frame_nhwc, input_name, output_name, device)
            if bound is not None:
                return bound
        try:
            return session.run([output_name], {input_name: frame_nhwc})[0]
        except Exception as exc:  # onnxruntime raises native exception types
            raise _wrap_onnx_error("ONNX inference failed", exc) from exc

    def _infer_iobinding(
        self, session: Any, frame_nhwc: np.ndarray, input_name: str, output_name: str, device: str
    ) -> np.ndarray | None:
        # Best-effort: bind input/output on the DirectML device to skip the
        # host<->device copies. Any failure (older ort, EP quirk) returns None
        # so the caller falls back to a plain run rather than failing the job.
        try:
            import onnxruntime as ort

            device_id = _parse_dml_device_id(device)
            io_binding = session.io_binding()
            input_value = ort.OrtValue.ortvalue_from_numpy(frame_nhwc, "dml", device_id)
            io_binding.bind_ortvalue_input(input_name, input_value)
            io_binding.bind_output(output_name, "dml")
            session.run_with_iobinding(io_binding)
            return io_binding.copy_outputs_to_cpu()[0]
        except Exception:  # noqa: BLE001
            # Log once: a persistent failure silently downgrades EVERY frame to
            # the slower plain-run path, defeating the whole speedup, so surface
            # it instead of leaving only "the encode was slow" as the symptom.
            if not self._iobinding_warned:
                self._iobinding_warned = True
                logger.warning(
                    "ONNX IO binding failed on %s; falling back to the slower plain-run path", device,
                    exc_info=True,
                )
            return None

    def _infer_tiled(self, session: Any, frame_nhwc: np.ndarray, device: str) -> np.ndarray:
        tile_size = self.settings.onnx_tile_size if self.settings.onnx_tile_size > 0 else FALLBACK_TILE_SIZE
        image = frame_nhwc[0]
        height, width, channels = image.shape
        starts_y = _tile_starts(height, tile_size, TILE_OVERLAP_PX)
        starts_x = _tile_starts(width, tile_size, TILE_OVERLAP_PX)

        tiles: list[tuple[int, int, int, int, np.ndarray]] = []
        for y0 in starts_y:
            for x0 in starts_x:
                tile_h = min(tile_size, height - y0)
                tile_w = min(tile_size, width - x0)
                source_tile = image[y0 : y0 + tile_h, x0 : x0 + tile_w][np.newaxis, ...]
                output_tile = self._infer_frame(session, source_tile, device)[0]
                tiles.append((y0, x0, tile_h, tile_w, output_tile))

        _, _, first_h, first_w, first_out = tiles[0]
        scale = _detect_scale(first_h, first_w, first_out)
        blended = self._blend_tiles(tiles, height, width, channels, scale)
        return blended[np.newaxis, ...]

    @staticmethod
    def _blend_tiles(
        tiles: list[tuple[int, int, int, int, np.ndarray]],
        height: int,
        width: int,
        channels: int,
        scale: int,
    ) -> np.ndarray:
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
            accumulator[oy : oy + out_h, ox : ox + out_w] += output_tile.astype(np.float32) * weights
            weight_sum[oy : oy + out_h, ox : ox + out_w] += weights
        blended = accumulator / np.clip(weight_sum, 1e-6, None)
        return _finalize_uint8(blended)

    # --- session cache -----------------------------------------------------

    def _get_session(self, model_path: str, device: str) -> Any:
        self.gpu_coordinator.acquire(device, self)
        cache_key = (model_path, device)
        with self._session_lock:
            cached = self._session_cache.get(cache_key)
            if cached is not None:
                self._session_cache.move_to_end(cache_key)
                return cached
        try:
            session = self._create_session(model_path, device)
        except Exception as exc:  # onnxruntime raises native exception types
            raise _wrap_onnx_error(
                f"Failed to load ONNX model {model_path!r} on device {device!r}", exc
            ) from exc
        with self._session_lock:
            self._session_cache[cache_key] = session
            self._session_cache.move_to_end(cache_key)
            if len(self._session_cache) > SESSION_CACHE_SIZE:
                self._session_cache.popitem(last=False)
        return session

    def _create_session(self, model_path: str, device: str) -> Any:
        # Monkeypatchable seam: unit tests replace this with a numpy fake so no
        # real onnxruntime session (or GPU) is needed.
        import onnxruntime as ort

        return ort.InferenceSession(model_path, providers=_build_providers(device))

    # --- frame counting ----------------------------------------------------

    @staticmethod
    def _count_frame_files(directory: Path) -> int:
        return sum(1 for _ in directory.glob("*.png"))

    def _validate_frame_output_count(self, frames_out: Path, expected_count: int) -> None:
        actual_count = self._count_frame_files(frames_out)
        if actual_count == 0:
            raise RuntimeError("ONNX video frame upscaling completed but no output frames were produced")
        if actual_count != expected_count:
            raise RuntimeError(
                f"ONNX video frame upscaling completed with {actual_count} frames, expected {expected_count}"
            )
