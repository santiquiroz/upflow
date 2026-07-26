from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np
import pytest

from app.services.frame_pipeline import (
    STREAM_QUEUE_CEILING,
    STREAM_QUEUE_FLOOR,
    FramePipeline,
    MapStage,
    derive_stream_queue_maxsizes,
    drain_stream,
    iter_png_frames,
)


def frame(value: int) -> np.ndarray:
    return np.full((1, 2, 3, 3), value % 256, dtype=np.uint8)


class DuplicateStage:
    """Etapa 1→2 (simula interpolación): emite el frame y una copia +100."""

    def process(self, f: np.ndarray) -> list[np.ndarray]:
        return [f, ((f.astype(np.int32) + 100) % 256).astype(np.uint8)]

    def flush(self) -> list[np.ndarray]:
        return []


class FlushStage:
    """Etapa que emite un frame extra recién en el flush (ventana tipo GMFSS)."""

    def process(self, f: np.ndarray) -> list[np.ndarray]:
        return [f]

    def flush(self) -> list[np.ndarray]:
        return [frame(99)]


class FailingStage:
    def __init__(self) -> None:
        self.seen = 0

    def process(self, f: np.ndarray) -> list[np.ndarray]:
        self.seen += 1
        if self.seen == 3:
            raise RuntimeError("boom en frame 3")
        return [f]

    def flush(self) -> list[np.ndarray]:
        return []


class NonIterableStage:
    def process(self, f: np.ndarray) -> None:
        return None

    def flush(self) -> list[np.ndarray]:
        return []


def run_pipeline(source_frames, stages, maxsizes=None):
    received: list[np.ndarray] = []
    pipeline = FramePipeline(
        iter(source_frames), stages, received.append, maxsizes or [2] * (len(stages) + 1)
    )
    delivered = pipeline.run(threading.Event())
    return delivered, received


def first_pixels(frames) -> list[int]:
    return [int(f[0, 0, 0, 0]) for f in frames]


def test_map_stage_preserves_count_and_order() -> None:
    delivered, received = run_pipeline([frame(i) for i in range(10)], [MapStage(lambda f: f * 2)])
    assert delivered == 10
    assert first_pixels(received) == [(i * 2) % 256 for i in range(10)]


def test_expanding_stage_emits_one_to_two_in_order() -> None:
    delivered, received = run_pipeline(
        [frame(i) for i in range(4)], [DuplicateStage(), MapStage(lambda f: f)]
    )
    assert delivered == 8
    assert first_pixels(received) == [0, 100, 1, 101, 2, 102, 3, 103]


def test_backpressure_with_maxsize_one_still_delivers_everything() -> None:
    delivered, _ = run_pipeline(
        [frame(i) for i in range(25)], [MapStage(lambda f: f)], maxsizes=[1, 1]
    )
    assert delivered == 25


def test_flush_frames_are_emitted_after_last_input() -> None:
    delivered, received = run_pipeline([frame(1)], [FlushStage()])
    assert delivered == 2
    assert first_pixels(received) == [1, 99]


def test_stage_error_propagates_and_joins_all_threads() -> None:
    threads_before = set(threading.enumerate())
    with pytest.raises(RuntimeError, match="boom en frame 3"):
        run_pipeline([frame(i) for i in range(10)], [FailingStage()])
    assert set(threading.enumerate()) <= threads_before, "quedó un thread vivo tras el error"


def test_non_iterable_stage_output_raises_without_hanging() -> None:
    cancel = threading.Event()
    pipeline = FramePipeline(iter([frame(1)]), [NonIterableStage()], lambda f: None, [1, 1])
    errors: list[BaseException] = []

    def run() -> None:
        try:
            pipeline.run(cancel)
        except BaseException as exc:  # noqa: BLE001 - el test captura el error del thread llamador
            errors.append(exc)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    runner.join(timeout=2)
    finished_without_cancel = not runner.is_alive()
    if not finished_without_cancel:
        cancel.set()
        runner.join(timeout=2)

    assert finished_without_cancel, "run() quedó colgado con una salida de etapa no iterable"
    assert len(errors) == 1
    assert isinstance(errors[0], TypeError)


def test_source_error_propagates() -> None:
    def broken_source():
        yield frame(1)
        raise RuntimeError("decode roto")

    threads_before = set(threading.enumerate())
    pipeline = FramePipeline(broken_source(), [MapStage(lambda f: f)], lambda f: None, [2, 2])
    with pytest.raises(RuntimeError, match="decode roto"):
        pipeline.run(threading.Event())
    assert set(threading.enumerate()) <= threads_before, "quedó un thread vivo tras el error"


def test_sink_error_propagates() -> None:
    def bad_sink(f: np.ndarray) -> None:
        raise ValueError("sink roto")

    threads_before = set(threading.enumerate())
    pipeline = FramePipeline(iter([frame(1)]), [MapStage(lambda f: f)], bad_sink, [2, 2])
    with pytest.raises(ValueError, match="sink roto"):
        pipeline.run(threading.Event())
    assert set(threading.enumerate()) <= threads_before, "quedó un thread vivo tras el error"


def test_preset_cancel_delivers_nothing_and_leaks_no_threads() -> None:
    cancel = threading.Event()
    cancel.set()
    received: list[np.ndarray] = []
    threads_before = set(threading.enumerate())
    pipeline = FramePipeline(
        iter([frame(i) for i in range(5)]), [MapStage(lambda f: f)], received.append, [2, 2]
    )
    delivered = pipeline.run(cancel)
    assert delivered == 0
    assert received == []
    assert set(threading.enumerate()) <= threads_before


def test_cancel_releases_producer_blocked_on_full_queue() -> None:
    cancel = threading.Event()
    sink_started = threading.Event()
    third_frame_requested = threading.Event()
    release_sink = threading.Event()
    delivered: list[int] = []
    errors: list[BaseException] = []
    threads_before = set(threading.enumerate())

    def contended_source():
        for value in range(4):
            if value == 2:
                third_frame_requested.set()
            yield frame(value)

    def slow_sink(f: np.ndarray) -> None:
        sink_started.set()
        release_sink.wait(timeout=5)

    pipeline = FramePipeline(contended_source(), [], slow_sink, [1])

    def run() -> None:
        try:
            delivered.append(pipeline.run(cancel))
        except BaseException as exc:  # noqa: BLE001 - evidencia cualquier fallo del pipeline
            errors.append(exc)

    runner = threading.Thread(target=run, daemon=True)
    runner.start()
    assert sink_started.wait(timeout=2), "el sink no recibió el primer frame"
    assert third_frame_requested.wait(timeout=2), "el productor no llegó a la cola llena"
    assert pipeline._queues[0].full(), "la cola debía estar llena antes de cancelar"

    cancel.set()
    release_sink.set()
    runner.join(timeout=3)

    assert not runner.is_alive(), "run() no retornó después de cancelar bajo contención"
    assert errors == []
    assert delivered == [1]
    assert set(threading.enumerate()) <= threads_before, "quedó un thread vivo tras cancelar"


def test_source_iterator_closes_deterministically_after_error() -> None:
    closed = threading.Event()

    def source():
        try:
            value = 0
            while True:
                yield frame(value)
                value += 1
        finally:
            closed.set()

    def bad_sink(f: np.ndarray) -> None:
        raise RuntimeError("sink roto")

    source_iterator = source()
    pipeline = FramePipeline(source_iterator, [], bad_sink, [1])

    with pytest.raises(RuntimeError, match="sink roto"):
        pipeline.run(threading.Event())

    assert closed.is_set(), "el iterador fuente no se cerró durante el teardown"


def test_queue_maxsizes_must_match_stage_count() -> None:
    with pytest.raises(ValueError, match="queue_maxsizes"):
        FramePipeline(iter([]), [MapStage(lambda f: f)], lambda f: None, [2])


@pytest.mark.parametrize("invalid_maxsize", [0, -1])
def test_queue_maxsizes_must_be_positive(invalid_maxsize: int) -> None:
    with pytest.raises(ValueError, match=r">= 1"):
        FramePipeline(
            iter([]),
            [MapStage(lambda f: f)],
            lambda f: None,
            [invalid_maxsize, 2],
        )


def test_derive_stream_queue_maxsizes_splits_budget_globally() -> None:
    # Entrada 720p (1280x720x3 ≈ 2.6MB), salida 4x (16x px ≈ 42.2MB), 1 etapa
    # => 2 colas, presupuesto 256MB => 128MB por cola: la de entrada satura el
    # techo (16), la de salida da 128MB // 42.2MB = 3.
    input_bytes = 1280 * 720 * 3
    sizes = derive_stream_queue_maxsizes(input_bytes, input_bytes * 16, 1, 256 * 1024 * 1024)
    assert sizes == [STREAM_QUEUE_CEILING, 3]


def test_derive_stream_queue_maxsizes_floors_tiny_budget() -> None:
    sizes = derive_stream_queue_maxsizes(10_000_000, 160_000_000, 2, 1024)
    assert sizes == [STREAM_QUEUE_FLOOR] * 3


def test_drain_stream_accumulates_chunks() -> None:
    sink: list[bytes] = []

    drain_stream(io.BytesIO(b"a" * 8192 + b"tail"), sink)

    assert sink == [b"a" * 8192, b"tail"]


def test_drain_stream_keeps_only_bounded_tail() -> None:
    sink: list[bytes] = []
    stream = io.BytesIO(b"".join(bytes([value]) * 8192 for value in range(70)))

    drain_stream(stream, sink)

    assert len(sink) == 64
    assert [chunk[0] for chunk in sink] == list(range(6, 70))


def test_drain_stream_swallows_read_errors() -> None:
    class BrokenStream:
        def __init__(self) -> None:
            self._reads = 0

        def read(self, size: int) -> bytes:
            self._reads += 1
            if self._reads == 1:
                return b"antes del error"
            raise OSError("pipe cerrado")

    sink: list[bytes] = []

    drain_stream(BrokenStream(), sink)

    assert sink == [b"antes del error"]


def write_png(path: Path, value: int) -> None:
    import cv2

    frame_bgr = np.full((2, 3, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), frame_bgr)


def test_iter_png_frames_yields_in_name_order(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index in range(1, 4):
        write_png(frames_dir / f"{index:08d}.png", index * 10)

    frames = list(iter_png_frames(frames_dir, threading.Event()))

    assert [f.shape for f in frames] == [(1, 2, 3, 3)] * 3
    assert [f.dtype for f in frames] == [np.uint8] * 3
    # Valor uniforme por canal: BGR->RGB no cambia el primer píxel.
    assert [int(f[0, 0, 0, 0]) for f in frames] == [10, 20, 30]


def test_iter_png_frames_stops_on_preset_cancel(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    write_png(frames_dir / "00000001.png", 10)
    cancel = threading.Event()
    cancel.set()

    assert list(iter_png_frames(frames_dir, cancel)) == []
