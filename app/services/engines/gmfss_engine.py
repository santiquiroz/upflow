from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.services.engines.gmfss import softsplat_cl
from app.services.engines.gmfss.assets import GRAPH_NAMES, GmfssAssets
from app.services.engines.gmfss.pipeline import GmfssDriver, resize_bilinear
from app.services.engines.onnx_upscaler import _build_providers, _wrap_onnx_error
from app.services.engines.onnx_video_upscaler import (
    _THREAD_JOIN_TIMEOUT_SECONDS,
    _drain_queue,
    _put_until_cancelled,
)
from app.services.gpu_session_coordinator import GpuSessionCoordinator

logger = logging.getLogger(__name__)

# Real measured config (port project Task 3.2, RX 7800 XT): DirectML + fp16
# fusionnet + OpenCL splat = 0.72-0.73fps @1080p 2x, vs ~0.2fps at fp32
# fusionnet + CPU splat (~3.5x). fp16 only wins on a GPU EP -- CPU-EP fp16 is
# emulated (slower than fp32), same rule ONNX_PREFER_FP16 already applies to
# the builtin upscale models -- so CPU always keeps fp32 fusionnet regardless
# of whether the fp16 file is present. Only fusionnet has an fp16 variant:
# featurenet/gmflow/metricnet hit a reproducible onnxconverter-common bug and
# were never converted (see the port repo's manifest.json fp16_variants note).
# GPU splat (driver.softsplat_cl) is always passed as splat_fn regardless of
# device or fp16 availability -- it has its own one-time-warning CPU fallback
# when pyopencl/an OpenCL GPU isn't actually available, so there is no
# separate device gate to duplicate here.
FP16_FUSIONNET_FILENAME = "fusionnet_fp16.onnx"

# ---------------------------------------------------------------------------
# GMFSS interpolation engine (ONNX, in-process). Second frame-interpolation
# engine next to RifeNcnnEngine: much higher quality (softmax-splatting-based,
# anime-tuned GMFSS_Fortuna) but 10x or more slower -- own port
# santiquiroz/port-gmfss-onnx (see app/services/engines/gmfss/ for the
# vendored driver). A short-clip smoke test measured ~20.8x full-job wall
# time, but the clip was short enough that GMFSS's cold-start ONNX session
# load dominated the ratio -- treat 10x as a floor, not a stable number.
# `run()`'s signature is IDENTICAL to RifeNcnnEngine.run's so it drops into
# video_upscaler._maybe_interpolate unchanged (Task 4.2 wires the actual
# engine selector; this task only builds the engine).
#
# Unlike RIFE (an external binary that distributes frames across pairs
# internally given `-n <count>`), GMFSS has no such binary -- this engine
# does the frame-pair -> timestep -> output-frame arithmetic itself. See
# _build_interpolation_plan.
#
# Session cache holds the 4 graphs of ONE device (LRU 1), same shape as
# AudioSrRestorer -- a full set is small (<80MB of weights) but there is no
# reason to keep more than the currently-selected device warm.
#
# MetricNet DirectML gotcha (Task 1.1 of the port project): the default ORT
# graph-fusion optimizer built a fused DML kernel for MetricNet that
# reproducibly hung the GPU (DXGI_ERROR_DEVICE_HUNG) on the validation
# hardware. The fix is graph_optimization_level = ORT_DISABLE_ALL -- applied
# here to EVERY session (all 4 graphs), not just MetricNet's, so the same
# class of fusion bug showing up on a different graph/driver is covered too.
# ---------------------------------------------------------------------------


class GmfssEngine:
    def __init__(self, settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._session_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._session_lock = threading.Lock()

    def available(self) -> bool:
        return self.settings.gmfss_available()

    def release_device(self, device: str) -> None:
        with self._session_lock:
            self._session_cache.pop(device, None)

    async def run(
        self,
        frames_in: Path,
        frames_out: Path,
        source_frame_count: int,
        multiplier: int = 1,
        *,
        target_frame_count: int | None = None,
        device: str | None = None,
    ) -> Path:
        if not self.available():
            raise RuntimeError(
                "GMFSS interpolation engine is not available. Enable ENABLE_GMFSS and install the "
                "models (scripts/download-gmfss-onnx.ps1)."
            )

        resolved_target_frame_count = self._resolve_target_frame_count(
            source_frame_count, multiplier, target_frame_count
        )
        frames_out.mkdir(parents=True, exist_ok=True)
        # GMFSS calls onnxruntime directly (unlike RIFE's Vulkan binary), so it
        # needs a concrete "cpu"/"dml:N" id to build providers -- same
        # None -> settings.default_device fallback video_upscaler already uses
        # for the audio-restore engines.
        resolved_device = device or self.settings.default_device

        cancel_event = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(
                self._run_blocking,
                frames_in,
                frames_out,
                source_frame_count,
                resolved_target_frame_count,
                resolved_device,
                cancel_event,
            )
        )
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Shield+await pattern (AudioSrRestorer/OnnxVideoUpscaler): the
            # pipeline threads can't be interrupted mid-write, so wait for them
            # to actually stop before letting the caller's frames_out cleanup
            # race a straggler write.
            cancel_event.set()
            with contextlib.suppress(BaseException):
                await worker
            raise

        self._validate_output_frame_count(frames_out, resolved_target_frame_count)
        return frames_out

    def run_frames_fused(
        self,
        frames_in: Path,
        source_frame_count: int,
        multiplier: int = 1,
        *,
        target_frame_count: int | None = None,
        device: str | None = None,
        upscale_frame: Callable[[np.ndarray], np.ndarray],
    ) -> Iterator[np.ndarray]:
        """Yield each output frame ALREADY interpolated + upscaled, in order,
        with no intermediate PNG round-trip -- the fused counterpart of run().

        Unlike run() (async, threaded PNG save pipeline, cancel_event + shield),
        this is a plain pull-based generator: the caller (Task 8) drives it and
        owns threading/cancellation. Abandoning it unwinds the generator with a
        GeneratorExit at the current yield -- no background thread outlives it,
        so none of run()'s threaded-teardown machinery is needed or duplicated
        here. Each yielded frame is NHWC uint8 RGB ([1,H,W,3]) at the source
        resolution -- the format OnnxVideoUpscaler consumes.

        CRITICAL: Each next() call blocks on ONNX inference; when called from
        async code, iterate this generator from inside a worker thread
        (e.g., asyncio.to_thread), never directly on the event loop.
        """
        if not self.available():
            raise RuntimeError(
                "GMFSS interpolation engine is not available. Enable ENABLE_GMFSS and install the "
                "models (scripts/download-gmfss-onnx.ps1)."
            )
        resolved_target_frame_count = self._resolve_target_frame_count(
            source_frame_count, multiplier, target_frame_count
        )
        resolved_device = device or self.settings.default_device
        for output_frame in self._iter_interpolated_frames(
            frames_in, source_frame_count, resolved_target_frame_count, resolved_device
        ):
            yield upscale_frame(output_frame)

    @staticmethod
    def _resolve_target_frame_count(
        source_frame_count: int, multiplier: int, target_frame_count: int | None
    ) -> int:
        if target_frame_count is not None:
            return target_frame_count
        return source_frame_count * multiplier

    def _run_blocking(
        self,
        frames_in: Path,
        frames_out: Path,
        source_frame_count: int,
        target_frame_count: int,
        device: str,
        cancel_event: threading.Event,
    ) -> None:
        driver, padded_hw, frame_paths, plan = self._prepare_pipeline(
            frames_in, source_frame_count, target_frame_count, device
        )
        self._run_pair_pipeline(driver, padded_hw, frame_paths, plan, frames_out, cancel_event)

    def _prepare_pipeline(
        self,
        frames_in: Path,
        source_frame_count: int,
        target_frame_count: int,
        device: str,
    ) -> tuple[GmfssDriver, tuple[int, int], list[Path], list[list[float]]]:
        # Setup shared by run() (threaded PNG pipeline) and run_frames_fused()
        # (pull-based generator): frame glob + count check, interpolation plan,
        # session load, driver build. Fail fast on a bad frame-count request
        # before paying for session load.
        frame_paths = sorted(frames_in.glob("*.png"))
        if len(frame_paths) != source_frame_count:
            raise RuntimeError(
                f"GMFSS expected {source_frame_count} source frames in {frames_in}, "
                f"found {len(frame_paths)}"
            )
        plan = _build_interpolation_plan(source_frame_count, target_frame_count)

        sessions = self._get_sessions(device)
        assets = GmfssAssets.load(self.settings.gmfss_model_dir_path)
        driver = GmfssDriver(assets, _graph_runner(sessions), splat_fn=softsplat_cl.splat_softmax)
        return driver, assets.padded_hw, frame_paths, plan

    def _iter_interpolated_frames(
        self,
        frames_in: Path,
        source_frame_count: int,
        target_frame_count: int,
        device: str,
    ) -> Iterator[np.ndarray]:
        # Same emission order and pair/timestep arithmetic as _compute_loop
        # (source[0], interp(pair0)..., source[1], ..., source[N-1]), but
        # synchronous and pull-based instead of pushed onto a save queue. Only
        # one pair (prev/next) is held at a time -- next of pair i is reused as
        # prev of pair i+1, so each source frame is decoded exactly once.
        driver, padded_hw, frame_paths, plan = self._prepare_pipeline(
            frames_in, source_frame_count, target_frame_count, device
        )

        prev_source, prev_chw, prev_hw = _load_source_frame(frame_paths[0], padded_hw)
        yield prev_source  # source[0] verbatim (t=0): raw pixels, no resize round-trip

        for pair_index in range(len(frame_paths) - 1):
            next_source, next_chw, next_hw = _load_source_frame(frame_paths[pair_index + 1], padded_hw)

            timesteps = plan[pair_index]
            if timesteps:  # a 0-extra pair skips reuse()+forward passes entirely
                for output_chw in driver.interpolate_pair(prev_chw, next_chw, timesteps):
                    yield _chw_float_to_nhwc_uint8(output_chw, prev_hw)

            yield next_source  # source[i+1] verbatim (t=1)
            prev_source, prev_chw, prev_hw = next_source, next_chw, next_hw

    # --- session cache -------------------------------------------------

    def _get_sessions(self, device: str) -> dict[str, Any]:
        self.gpu_coordinator.acquire(device, self)
        with self._session_lock:
            cached = self._session_cache.get(device)
            if cached is not None:
                self._session_cache.move_to_end(device)
                return cached

        try:
            sessions = self._create_sessions(device)
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise _wrap_onnx_error(f"Failed to load GMFSS models on device {device!r}", exc) from exc

        with self._session_lock:
            self._session_cache[device] = sessions
            self._session_cache.move_to_end(device)
            if len(self._session_cache) > 1:
                self._session_cache.popitem(last=False)
        return sessions

    def _create_sessions(self, device: str) -> dict[str, Any]:
        # Monkeypatchable seam: unit tests override this to inject fake numpy
        # sessions and never touch real onnxruntime.
        import onnxruntime as ort

        providers = _build_providers(device)
        model_dir = self.settings.gmfss_model_dir_path
        sessions: dict[str, Any] = {}
        for name in GRAPH_NAMES:
            sess_options = ort.SessionOptions()
            # MUST be disabled on every graph, not just MetricNet's -- see the
            # module docstring for the DXGI_ERROR_DEVICE_HUNG history.
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            _tune_session_options_for_device(sess_options, device)
            graph_path = self._resolve_graph_path(model_dir, name, device)
            sessions[name] = ort.InferenceSession(
                str(graph_path), sess_options=sess_options, providers=providers
            )
        return sessions

    @staticmethod
    def _resolve_graph_path(model_dir: Path, name: str, device: str) -> Path:
        if name == "fusionnet" and _should_use_fp16_fusionnet(model_dir, device):
            return model_dir / FP16_FUSIONNET_FILENAME
        return model_dir / f"{name}.onnx"

    # --- threaded load/compute/save pipeline ----------------------------

    def _run_pair_pipeline(
        self,
        driver: GmfssDriver,
        padded_hw: tuple[int, int],
        frame_paths: list[Path],
        plan: list[list[float]],
        frames_out: Path,
        cancel_event: threading.Event,
    ) -> None:
        n_load = max(1, self.settings.onnx_video_load_threads)
        n_save = max(1, self.settings.onnx_video_save_threads)
        png_level = self.settings.onnx_video_png_compression

        todo: queue.Queue[tuple[int, Path]] = queue.Queue()
        for index, path in enumerate(frame_paths):
            todo.put((index, path))
        load_q: queue.Queue[tuple[int, np.ndarray, tuple[int, int]]] = queue.Queue(maxsize=n_load * 2)
        save_q: queue.Queue[tuple | None] = queue.Queue(maxsize=n_save * 2)
        errors: list[Exception] = []

        loaders = [
            threading.Thread(
                target=_loader_loop, args=(todo, load_q, padded_hw, errors, cancel_event), daemon=True
            )
            for _ in range(n_load)
        ]
        savers = [
            threading.Thread(target=_saver_loop, args=(save_q, png_level, errors, cancel_event), daemon=True)
            for _ in range(n_save)
        ]
        for thread in loaders + savers:
            thread.start()

        try:
            self._compute_loop(
                driver, frame_paths, plan, load_q, save_q, frames_out, loaders, errors, cancel_event
            )
        finally:
            _drain_queue(load_q)
            for _ in savers:
                save_q.put(None)  # sentinel: savers are draining, so this never blocks for long
            for thread in savers + loaders:
                thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    logger.error("gmfss pipeline thread did not stop within timeout: %s", thread.name)

        if errors:
            raise errors[0]

    @staticmethod
    def _next_loaded_frame(
        load_q: queue.Queue,
        pending: dict[int, tuple[np.ndarray, tuple[int, int]]],
        index: int,
        loaders: list[threading.Thread],
        errors: list[Exception],
        cancel_event: threading.Event,
    ) -> tuple[np.ndarray, tuple[int, int]] | None:
        while index not in pending:
            if cancel_event.is_set() or errors:
                return None
            try:
                got_index, frame, original_hw = load_q.get(timeout=0.2)
            except queue.Empty:
                # No frame ready: bail if every loader finished and nothing more
                # is coming (a loader died mid-run), same guard as
                # OnnxVideoUpscaler._infer_loop -- otherwise this polls forever
                # waiting on a frame that will never arrive.
                if load_q.empty() and all(not thread.is_alive() for thread in loaders) and index not in pending:
                    return None
                continue
            pending[got_index] = (frame, original_hw)
        return pending.pop(index)

    @staticmethod
    def _compute_loop(
        driver: GmfssDriver,
        frame_paths: list[Path],
        plan: list[list[float]],
        load_q: queue.Queue,
        save_q: queue.Queue,
        frames_out: Path,
        loaders: list[threading.Thread],
        errors: list[Exception],
        cancel_event: threading.Event,
    ) -> None:
        # Emits: source[0], interp(pair0)..., source[1], interp(pair1)..., ...,
        # source[N-1]. Only ONE pair's worth of frames (prev/next) is ever held
        # in memory -- frame[i+1] of pair i is reused as frame[i] of pair i+1,
        # so each source frame is loaded exactly once regardless of N.
        pending: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}
        total = len(frame_paths)
        out_index = 1

        first = GmfssEngine._next_loaded_frame(load_q, pending, 0, loaders, errors, cancel_event)
        if first is None:
            return
        prev_frame, prev_hw = first
        if not _enqueue_copy(save_q, frame_paths[0], frames_out, out_index, cancel_event):
            return
        out_index += 1

        for pair_index in range(total - 1):
            # Cancel is checked once per PAIR (not more granularly): reuse() +
            # its multi-timestep forward passes are GMFSS's natural checkpoint
            # boundary, same granularity RifeNcnnEngine/OnnxVideoUpscaler use.
            if cancel_event.is_set() or errors:
                return

            nxt = GmfssEngine._next_loaded_frame(load_q, pending, pair_index + 1, loaders, errors, cancel_event)
            if nxt is None:
                return
            next_frame, next_hw = nxt

            timesteps = plan[pair_index]
            if timesteps:
                try:
                    outputs = driver.interpolate_pair(prev_frame, next_frame, timesteps)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    cancel_event.set()
                    return
                for output in outputs:
                    if not _enqueue_frame(save_q, output, prev_hw, frames_out, out_index, cancel_event):
                        return
                    out_index += 1

            if not _enqueue_copy(save_q, frame_paths[pair_index + 1], frames_out, out_index, cancel_event):
                return
            out_index += 1

            prev_frame, prev_hw = next_frame, next_hw

    # --- output validation -----------------------------------------------

    def _validate_output_frame_count(self, frames_out: Path, target_frame_count: int) -> None:
        actual_frame_count = self._count_output_frames(frames_out)

        if actual_frame_count == 0:
            raise RuntimeError("GMFSS interpolation completed but no output frames were produced")

        if actual_frame_count != target_frame_count:
            raise RuntimeError(
                "GMFSS interpolation completed with "
                f"{actual_frame_count} frames, expected {target_frame_count}"
            )

    @staticmethod
    def _count_output_frames(frames_out: Path) -> int:
        return sum(1 for _ in frames_out.glob("*.png"))


# ---------------------------------------------------------------------------
# Frame-pair -> timestep -> output-frame arithmetic (pure, no I/O -- see
# tests/test_gmfss_engine.py for exactness coverage across multiplier and
# target_frame_count modes).
# ---------------------------------------------------------------------------


def _distribute_extra_frames(pair_count: int, extra_frames: int) -> list[int]:
    """Bresenham-style distribution of `extra_frames` interpolated frames
    across `pair_count` consecutive source-frame pairs, as evenly as possible.

    Exact: sum(result) == extra_frames and len(result) == pair_count. The
    accumulator spreads the remainder across the whole run (not bunched at
    the front like a plain divmod split), so every entry is within 1 of the
    mean extra_frames/pair_count -- this is what makes multiplier mode (extra
    = source_frame_count * (multiplier - 1)) come out to "multiplier - 1 per
    pair" almost everywhere, with the same exact algorithm handling arbitrary
    (non-integer-ratio) target_frame_count requests too.
    """
    if pair_count <= 0:
        if extra_frames != 0:
            raise ValueError("Cannot distribute extra frames across zero pairs")
        return []
    counts = []
    accumulator = 0
    for _ in range(pair_count):
        accumulator += extra_frames
        count = accumulator // pair_count
        accumulator -= count * pair_count
        counts.append(count)
    return counts


def _pair_timesteps(extra_count: int) -> list[float]:
    """Evenly spaced timesteps strictly inside (0, 1) for `extra_count` frames
    interpolated between two consecutive source frames -- the source frames
    themselves are t=0/t=1 and are emitted verbatim (copied), never
    re-synthesized through the network."""
    if extra_count <= 0:
        return []
    return [(index + 1) / (extra_count + 1) for index in range(extra_count)]


def _build_interpolation_plan(source_frame_count: int, target_frame_count: int) -> list[list[float]]:
    """For each of the (source_frame_count - 1) consecutive source-frame
    pairs, the list of timesteps to interpolate at.

    Exact by construction: source_frame_count + sum(len(p) for p in plan) ==
    target_frame_count, for any target_frame_count >= source_frame_count
    (raises RuntimeError otherwise -- interpolation only adds frames).
    """
    pair_count = source_frame_count - 1
    extra_frames = target_frame_count - source_frame_count
    if extra_frames < 0:
        raise RuntimeError(
            f"GMFSS target_frame_count ({target_frame_count}) is smaller than "
            f"source_frame_count ({source_frame_count}); interpolation cannot remove frames"
        )
    if extra_frames > 0 and pair_count <= 0:
        raise RuntimeError(
            f"Cannot interpolate {source_frame_count} source frame(s) into "
            f"{target_frame_count} frames: at least 2 source frames are required"
        )
    per_pair_extra = _distribute_extra_frames(pair_count, extra_frames)
    return [_pair_timesteps(count) for count in per_pair_extra]


# ---------------------------------------------------------------------------
# Frame I/O: uint8 RGB PNG <-> normalized [1,3,H,W] float32 [0,1], resized
# (stretch bilinear, not letterbox-padded) to/from the driver's fixed padded
# resolution -- same resize_bilinear call driver/pipeline.py itself uses
# internally, applied here at the engine boundary in both directions.
# ---------------------------------------------------------------------------


def _decode_rgb(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read frame {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, (rgb.shape[0], rgb.shape[1])


def _rgb_to_padded_chw(
    rgb: np.ndarray, original_hw: tuple[int, int], padded_hw: tuple[int, int]
) -> np.ndarray:
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32)[np.newaxis, ...] / 255.0
    if original_hw != padded_hw:
        chw = resize_bilinear(chw, padded_hw[0], padded_hw[1])
    return np.ascontiguousarray(chw, dtype=np.float32)


def _load_padded_frame(path: Path, padded_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    rgb, original_hw = _decode_rgb(path)
    return _rgb_to_padded_chw(rgb, original_hw, padded_hw), original_hw


def _load_source_frame(
    path: Path, padded_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Decode a source frame once into both representations run_frames_fused
    needs: the raw NHWC uint8 RGB frame ([1,H,W,3], yielded verbatim for the
    t=0/t=1 boundary frames -- pixel-identical, no resize round-trip) and the
    padded [1,3,pH,pW] float tensor fed to the driver for interpolation."""
    rgb, original_hw = _decode_rgb(path)
    source_nhwc = np.ascontiguousarray(rgb)[np.newaxis, ...]
    padded_chw = _rgb_to_padded_chw(rgb, original_hw, padded_hw)
    return source_nhwc, padded_chw, original_hw


def _chw_float_to_hwc_uint8(frame_chw: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    """[1,3,H,W] float32 [0,1] (driver padded res) -> [H,W,3] uint8 RGB resized
    back to original_hw. The exact numpy conversion _save_frame applies before
    the RGB->BGR + imwrite; run_frames_fused reuses it (NHWC-batched) so the
    fused output matches what run()'s PNGs would have held, minus the disk hop."""
    current_hw = (frame_chw.shape[2], frame_chw.shape[3])
    if current_hw != original_hw:
        frame_chw = resize_bilinear(frame_chw, original_hw[0], original_hw[1])
    hwc = np.transpose(np.clip(frame_chw[0], 0.0, 1.0), (1, 2, 0))
    return np.rint(hwc * 255.0).astype(np.uint8)


def _chw_float_to_nhwc_uint8(frame_chw: np.ndarray, original_hw: tuple[int, int]) -> np.ndarray:
    return _chw_float_to_hwc_uint8(frame_chw, original_hw)[np.newaxis, ...]


def _save_frame(frame_chw: np.ndarray, original_hw: tuple[int, int], path: Path, png_compression: int) -> None:
    import cv2

    rgb_u8 = _chw_float_to_hwc_uint8(frame_chw, original_hw)
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)])
    if not ok:
        raise RuntimeError(f"Failed to write frame {path}")


def _output_path(frames_out: Path, index: int) -> Path:
    return frames_out / f"{index:08d}.png"


def _enqueue_copy(
    save_q: queue.Queue, source_path: Path, frames_out: Path, index: int, cancel_event: threading.Event
) -> bool:
    dest_path = _output_path(frames_out, index)
    return _put_until_cancelled(save_q, ("copy", source_path, dest_path), cancel_event)


def _enqueue_frame(
    save_q: queue.Queue,
    frame_chw: np.ndarray,
    original_hw: tuple[int, int],
    frames_out: Path,
    index: int,
    cancel_event: threading.Event,
) -> bool:
    dest_path = _output_path(frames_out, index)
    return _put_until_cancelled(save_q, ("frame", frame_chw, original_hw, dest_path), cancel_event)


def _loader_loop(
    todo: queue.Queue[tuple[int, Path]],
    load_q: queue.Queue[tuple[int, np.ndarray, tuple[int, int]]],
    padded_hw: tuple[int, int],
    errors: list[Exception],
    cancel_event: threading.Event,
) -> None:
    while not cancel_event.is_set():
        try:
            index, path = todo.get_nowait()
        except queue.Empty:
            return
        try:
            frame, original_hw = _load_padded_frame(path, padded_hw)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            cancel_event.set()
            return
        if not _put_until_cancelled(load_q, (index, frame, original_hw), cancel_event):
            return


def _saver_loop(
    save_q: queue.Queue,
    png_level: int,
    errors: list[Exception],
    cancel_event: threading.Event,
) -> None:
    while True:
        item = save_q.get()
        if item is None:
            save_q.task_done()
            return
        try:
            _execute_save_task(item, png_level)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            cancel_event.set()
        finally:
            save_q.task_done()


def _execute_save_task(item: tuple, png_level: int) -> None:
    if item[0] == "copy":
        _, source_path, dest_path = item
        shutil.copyfile(source_path, dest_path)
        return
    _, frame_chw, original_hw, dest_path = item
    _save_frame(frame_chw, original_hw, dest_path, png_level)


def _should_use_fp16_fusionnet(model_dir: Path, device: str) -> bool:
    if _is_cpu_device(device):
        return False
    return (model_dir / FP16_FUSIONNET_FILENAME).exists()


def _is_cpu_device(device: str) -> bool:
    return device.strip().lower() == "cpu"


def _tune_session_options_for_device(sess_options: Any, device: str) -> None:
    """DirectML execution-provider guidance (onnxruntime docs): the DML EP
    manages its own GPU allocator, so ORT's memory-pattern reuse optimization
    is not just useless there -- it's a known contributor to the same class of
    driver instability as the MetricNet DXGI_ERROR_DEVICE_HUNG bug documented
    in the module docstring above, so disable it whenever compute actually
    runs on DML. Single intra-op thread avoids spinning up CPU worker threads
    for graphs this tiny, whose real cost is GPU dispatch overhead rather than
    CPU-side op parallelism; the CPU device path is left at ORT's defaults
    since CPU *is* the compute backend there and benefits from its own
    multi-threading."""
    if _is_cpu_device(device):
        return
    sess_options.enable_mem_pattern = False
    sess_options.intra_op_num_threads = 1


def _graph_runner(sessions: dict[str, Any]):
    def run_graph(name: str, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        session = sessions[name]
        feeds = {key: np.ascontiguousarray(value) for key, value in feeds.items()}
        try:
            return session.run(None, feeds)
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise _wrap_onnx_error(f"GMFSS {name} inference failed", exc) from exc

    return run_graph
