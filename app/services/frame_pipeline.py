from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Protocol

import numpy as np

from app.services.engines.onnx_video_upscaler import (
    _QUEUE_POLL_SECONDS,
    _THREAD_JOIN_TIMEOUT_SECONDS,
    _drain_queue,
    _put_until_cancelled,
    derive_queue_maxsize,
)

logger = logging.getLogger(__name__)

# Piso/techo por cola del stream pipeline: el piso evita matar el overlap con
# presupuestos chicos; el techo evita que frames diminutos acumulen cientos de
# entradas sin beneficio (el productor solo necesita ir un puñado adelante).
STREAM_QUEUE_FLOOR = 2
STREAM_QUEUE_CEILING = 16


class FrameStage(Protocol):
    """Etapa del pipeline: recibe un frame NHWC uint8 RGB [1,H,W,3] y emite
    0..N frames en orden (upscaler 1→1; interpolador 1→1+extras con ventana de
    2). flush() emite lo retenido al agotarse la fuente."""

    def process(self, frame: np.ndarray) -> list[np.ndarray]: ...

    def flush(self) -> list[np.ndarray]: ...


class MapStage:
    """Etapa 1→1 sobre un callable puro por frame (p.ej. el closure de upscale
    de OnnxVideoUpscaler.build_frame_upscaler)."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._fn = fn

    def process(self, frame: np.ndarray) -> list[np.ndarray]:
        return [self._fn(frame)]

    def flush(self) -> list[np.ndarray]:
        return []


def derive_stream_queue_maxsizes(
    input_frame_bytes: int, output_frame_bytes: int, n_stages: int, budget_bytes: int
) -> list[int]:
    """maxsize por cola bajo presupuesto GLOBAL (spec: repartido entre TODAS las
    colas del pipeline, no por cola). n_stages etapas => n_stages+1 colas; la
    ÚLTIMA transporta frames de salida (scale² más grandes), el resto de entrada.
    """
    n_queues = n_stages + 1
    per_queue_budget = max(1, budget_bytes // n_queues)
    sizes = [
        derive_queue_maxsize(input_frame_bytes, per_queue_budget, STREAM_QUEUE_FLOOR, STREAM_QUEUE_CEILING)
        for _ in range(n_queues - 1)
    ]
    sizes.append(
        derive_queue_maxsize(output_frame_bytes, per_queue_budget, STREAM_QUEUE_FLOOR, STREAM_QUEUE_CEILING)
    )
    return sizes


def drain_stream(stream, sink: list[bytes]) -> None:
    # ffmpeg llena su pipe de stderr; si nadie lo drena, el pipe de datos se
    # bloquea cuando el buffer se llena. Se conserva solo la cola para errores.
    # (Cuerpo idéntico al VideoUpscaler._drain_stream que Task 5 elimina.)
    try:
        for chunk in iter(lambda: stream.read(8192), b""):
            sink.append(chunk)
            if len(sink) > 64:
                del sink[:-64]
    except Exception:  # noqa: BLE001 - stream cerrado en un kill
        pass


class FramePipeline:
    """Etapas conectadas por colas acotadas, cada una en su propio thread.

    Backpressure: cola llena ⇒ el productor bloquea (vía _put_until_cancelled,
    que observa cancel_event — nunca un put ciego). Orden garantizado: un solo
    thread por etapa + colas FIFO. Fin de stream: sentinel None que cada etapa
    reenvía tras emitir su flush(). El sink corre en el thread llamador.
    """

    def __init__(
        self,
        source: Iterator[np.ndarray],
        stages: list[FrameStage],
        sink: Callable[[np.ndarray], None],
        queue_maxsizes: list[int],
    ) -> None:
        if len(queue_maxsizes) != len(stages) + 1:
            raise ValueError("queue_maxsizes debe tener len(stages) + 1 entradas")
        self._source = source
        self._stages = list(stages)
        self._sink = sink
        self._queues: list[queue.Queue] = [queue.Queue(maxsize=size) for size in queue_maxsizes]
        self._errors: list[Exception] = []

    def run(self, cancel_event: threading.Event) -> int:
        threads = [
            threading.Thread(
                target=self._source_loop, args=(cancel_event,), daemon=True, name="frame-pipeline-source"
            )
        ]
        for index, stage in enumerate(self._stages):
            threads.append(
                threading.Thread(
                    target=self._stage_loop,
                    args=(stage, self._queues[index], self._queues[index + 1], cancel_event),
                    daemon=True,
                    name=f"frame-pipeline-stage-{index}",
                )
            )
        for thread in threads:
            thread.start()
        try:
            delivered = self._sink_loop(cancel_event)
        finally:
            # Mismo teardown que OnnxVideoUpscaler._run_pipeline: drenar para
            # desbloquear productores y esperar los threads SIEMPRE, también en
            # el camino de error/cancel, para no filtrar threads zombie.
            if self._errors:
                cancel_event.set()
            for pending_queue in self._queues:
                _drain_queue(pending_queue)
            for thread in threads:
                thread.join(timeout=_THREAD_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    logger.error("frame pipeline thread did not stop within timeout: %s", thread.name)
        if self._errors:
            raise self._errors[0]
        return delivered

    def _source_loop(self, cancel_event: threading.Event) -> None:
        out_q = self._queues[0]
        try:
            for source_frame in self._source:
                if cancel_event.is_set():
                    return
                if not _put_until_cancelled(out_q, source_frame, cancel_event):
                    return
        except Exception as exc:  # noqa: BLE001 - un decode roto es error de pipeline, no crash
            self._fail(exc, cancel_event)
            return
        _put_until_cancelled(out_q, None, cancel_event)

    def _stage_loop(
        self,
        stage: FrameStage,
        in_q: queue.Queue,
        out_q: queue.Queue,
        cancel_event: threading.Event,
    ) -> None:
        while True:
            if cancel_event.is_set() or self._errors:
                return
            try:
                item = in_q.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            try:
                outputs = stage.flush() if item is None else stage.process(item)
            except Exception as exc:  # noqa: BLE001
                self._fail(exc, cancel_event)
                return
            for output_frame in outputs:
                if not _put_until_cancelled(out_q, output_frame, cancel_event):
                    return
            if item is None:
                _put_until_cancelled(out_q, None, cancel_event)
                return

    def _sink_loop(self, cancel_event: threading.Event) -> int:
        delivered = 0
        last_q = self._queues[-1]
        while True:
            if cancel_event.is_set() or self._errors:
                return delivered
            try:
                item = last_q.get(timeout=_QUEUE_POLL_SECONDS)
            except queue.Empty:
                continue
            if item is None:
                return delivered
            try:
                self._sink(item)
            except Exception as exc:  # noqa: BLE001 - ffmpeg muerto / broken pipe
                self._fail(exc, cancel_event)
                return delivered
            delivered += 1

    def _fail(self, exc: Exception, cancel_event: threading.Event) -> None:
        self._errors.append(exc)
        cancel_event.set()
