from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from app.services.engines.frame_workers import (
    QUEUE_POLL_SECONDS,
    THREAD_JOIN_TIMEOUT_SECONDS,
    derive_queue_maxsize,
    drain_queue,
    load_frame,
    put_until_cancelled,
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
    2). flush() emite lo retenido al agotarse la fuente.

    El retorno es un ITERABLE, no necesariamente una lista: una etapa 1→N puede
    devolver un generador para que sus frames lleguen a la cola de a uno y el
    presupuesto acotado siga valiendo (una lista completa esquivaría el
    backpressure). Las listas siguen siendo válidas — son iterables.
    """

    def process(self, frame: np.ndarray) -> Iterable[np.ndarray]: ...

    def flush(self) -> Iterable[np.ndarray]: ...


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


def iter_png_frames(frames_dir: Path, cancel_event: threading.Event) -> Iterator[np.ndarray]:
    """Source del tramo híbrido: los PNGs %08d.png del directorio, en orden,
    como frames NHWC uint8. UN solo hilo los decodea (~30ms/frame vs ~116ms de
    infer: no es cuello — mismo racional que el loader único del raw-pipe)."""
    for path in sorted(frames_dir.glob("*.png")):
        if cancel_event.is_set():
            return
        yield load_frame(path)


class FramePipeline:
    """Etapas conectadas por colas acotadas, cada una en su propio thread.

    Backpressure: cola llena ⇒ el productor bloquea (vía put_until_cancelled,
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
        if any(size < 1 for size in queue_maxsizes):
            raise ValueError("todos los valores de queue_maxsizes deben ser >= 1")
        self._source = source
        self._stages = list(stages)
        self._sink = sink
        self._queues: list[queue.Queue] = [queue.Queue(maxsize=size) for size in queue_maxsizes]
        self._errors: list[BaseException] = []

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
                drain_queue(pending_queue)
            for thread in threads:
                thread.join(timeout=THREAD_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():
                    logger.error("frame pipeline thread did not stop within timeout: %s", thread.name)
            self._close_source()
        if self._errors:
            raise self._errors[0]
        return delivered

    def _source_loop(self, cancel_event: threading.Event) -> None:
        out_q = self._queues[0]
        try:
            for source_frame in self._source:
                if cancel_event.is_set():
                    return
                if not put_until_cancelled(out_q, source_frame, cancel_event):
                    return
            put_until_cancelled(out_q, None, cancel_event)
        except BaseException as exc:  # noqa: BLE001 - todo fallo debe cancelar el pipeline
            self._fail(exc, cancel_event)

    def _stage_loop(
        self,
        stage: FrameStage,
        in_q: queue.Queue,
        out_q: queue.Queue,
        cancel_event: threading.Event,
    ) -> None:
        try:
            while True:
                if cancel_event.is_set() or self._errors:
                    return
                try:
                    item = in_q.get(timeout=QUEUE_POLL_SECONDS)
                except queue.Empty:
                    continue
                outputs = stage.flush() if item is None else stage.process(item)
                for output_frame in outputs:
                    if not put_until_cancelled(out_q, output_frame, cancel_event):
                        return
                if item is None:
                    put_until_cancelled(out_q, None, cancel_event)
                    return
        except BaseException as exc:  # noqa: BLE001 - todo fallo debe cancelar el pipeline
            self._fail(exc, cancel_event)

    def _sink_loop(self, cancel_event: threading.Event) -> int:
        delivered = 0
        last_q = self._queues[-1]
        while True:
            if cancel_event.is_set() or self._errors:
                return delivered
            try:
                item = last_q.get(timeout=QUEUE_POLL_SECONDS)
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

    def _fail(self, exc: BaseException, cancel_event: threading.Event) -> None:
        self._errors.append(exc)
        cancel_event.set()

    def _close_source(self) -> None:
        try:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()
        except BaseException:  # noqa: BLE001 - el cierre no debe tapar el error original
            logger.exception("no se pudo cerrar el iterador fuente del frame pipeline")
