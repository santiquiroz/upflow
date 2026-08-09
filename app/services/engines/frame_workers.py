from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Shared infrastructure for the threaded frame pipelines (OnnxVideoUpscaler,
# GmfssEngine, FramePipeline, video_upscaler): cancel-aware bounded queues,
# RAM-budget queue sizing, the preallocated readback ring, cv2 NHWC frame
# I/O, and the generic loader/saver worker loops.
#
# Extracted from engines/onnx_video_upscaler.py (consolidation roadmap #3):
# frame_pipeline/gmfss_engine/video_upscaler were importing its `_`-private
# helpers, and the gmfss + onnx_video loader/saver loops were structurally
# identical (None sentinel, errors.append + cancel_event.set) modulo the
# per-item transform -- which is now a callable parameter.
# ---------------------------------------------------------------------------

# Blocking queue put/join poll interval: short enough that a cancel is observed
# almost immediately, long enough not to busy-spin.
QUEUE_POLL_SECONDS = 0.2
THREAD_JOIN_TIMEOUT_SECONDS = 30.0


def put_until_cancelled(
    q: queue.Queue, item: Any, cancel_event: threading.Event, timeout: float = QUEUE_POLL_SECONDS
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


def drain_queue(q: queue.Queue) -> None:
    """Discard everything currently queued so a producer blocked on a full queue
    can complete its put and exit. Used during pipeline teardown."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def derive_queue_maxsize(frame_bytes: int, budget_bytes: int, floor: int, ceiling: int) -> int:
    """Cuántos frames caben en cola bajo un presupuesto de RAM en bytes.

    Extraído de _save_queue_maxsize para compartirlo con las colas del stream
    pipeline (spec 2026-07-25-stream-frame-pipeline-design.md): mismo criterio,
    piso para no matar throughput y techo para no acumular de más.
    """
    if frame_bytes <= 0:
        return ceiling
    return max(floor, min(ceiling, budget_bytes // frame_bytes))


MIN_READBACK_RING_CAPACITY = 2


class FrameReadbackRing:
    """Anillo de K buffers CPU preasignados para el readback GPU→CPU de frames.

    Alocar un array de salida nuevo por frame costaba ~11% del tiempo de frame
    (54.3→48.4 ms medido); el anillo preasigna los buffers una vez y rota. El
    output de ORT se COPIA al buffer (np.copyto): lo que se evita es la
    allocación por frame, no la copia. K=1 está prohibido porque reusar el
    único buffer mientras el frame anterior sigue en la cola downstream
    corrompe frames (medido) — de ahí el mínimo del constructor.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < MIN_READBACK_RING_CAPACITY:
            raise ValueError(
                f"readback ring capacity must be >= {MIN_READBACK_RING_CAPACITY}, got {capacity}"
            )
        self.capacity = capacity
        self._buffers: list[np.ndarray] = []
        self._index = 0
        self._shape: tuple[int, ...] | None = None
        self._dtype: Any = None

    def copy_in(self, frame: np.ndarray) -> np.ndarray:
        """Copia `frame` al siguiente buffer del anillo y devuelve ese buffer."""
        if frame.shape != self._shape or frame.dtype != self._dtype:
            self._reallocate(frame.shape, frame.dtype)
        buffer = self._buffers[self._index]
        self._index = (self._index + 1) % self.capacity
        np.copyto(buffer, frame)
        return buffer

    def _reallocate(self, shape: tuple[int, ...], dtype: Any) -> None:
        # Cambio de resolución mid-run: los buffers viejos tienen otro shape,
        # así que el anillo completo se re-aloca para el shape nuevo.
        self._buffers = [np.empty(shape, dtype=dtype) for _ in range(self.capacity)]
        self._index = 0
        self._shape = shape
        self._dtype = dtype


def derive_readback_ring_capacity(downstream_slots: int, n_consumers: int) -> int:
    """K > frames en vuelo aguas abajo del infer.

    En vuelo: 1 recién producido (aún no encolado) + `downstream_slots` que la
    cola puede retener + `n_consumers` en manos de los threads consumidores.
    El +1 extra es el margen mínimo (K = en_vuelo + 1) para no reusar jamás un
    buffer que la cola downstream todavía referencia — eso corrompe frames.
    """
    return downstream_slots + n_consumers + 2


def load_frame(source_path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read frame {source_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb)[np.newaxis, ...]  # NHWC uint8 [1,H,W,3]


def save_frame(frame_nhwc: np.ndarray, output_path: Path, png_compression: int) -> None:
    import cv2

    bgr = cv2.cvtColor(frame_nhwc[0], cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(output_path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)])
    if not ok:
        raise RuntimeError(f"Failed to write frame {output_path}")


def loader_loop(
    todo: queue.Queue,
    load_q: queue.Queue,
    load_item: Callable[[Any], Any],
    errors: list[Exception],
    cancel_event: threading.Event,
) -> None:
    """Generic loader worker: drains `todo`, transforms each work unit with
    `load_item` (the engine-specific decode -- gmfss closes over its padded
    resolution here) and enqueues the result with cancel-aware backpressure.
    First error cancels the whole pipeline."""
    while not cancel_event.is_set():
        try:
            work = todo.get_nowait()
        except queue.Empty:
            return
        try:
            item = load_item(work)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            cancel_event.set()
            return
        if not put_until_cancelled(load_q, item, cancel_event):
            return


def saver_loop(
    save_q: queue.Queue,
    save_item: Callable[[Any], None],
    errors: list[Exception],
    cancel_event: threading.Event,
) -> None:
    """Generic saver worker: consumes `save_q` until the None sentinel,
    persisting each item with the engine-specific `save_item`. A failed save
    cancels the pipeline but still keeps draining (task_done) so producers
    blocked on the queue unwind cleanly."""
    while True:
        item = save_q.get()
        if item is None:
            save_q.task_done()
            return
        try:
            save_item(item)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
            cancel_event.set()
        finally:
            save_q.task_done()
