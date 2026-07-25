from __future__ import annotations

import threading

import numpy as np
import pytest

from app.services.frame_pipeline import (
    STREAM_QUEUE_CEILING,
    STREAM_QUEUE_FLOOR,
    FramePipeline,
    MapStage,
    derive_stream_queue_maxsizes,
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


def test_source_error_propagates() -> None:
    def broken_source():
        yield frame(1)
        raise RuntimeError("decode roto")

    pipeline = FramePipeline(broken_source(), [MapStage(lambda f: f)], lambda f: None, [2, 2])
    with pytest.raises(RuntimeError, match="decode roto"):
        pipeline.run(threading.Event())


def test_sink_error_propagates() -> None:
    def bad_sink(f: np.ndarray) -> None:
        raise ValueError("sink roto")

    pipeline = FramePipeline(iter([frame(1)]), [MapStage(lambda f: f)], bad_sink, [2, 2])
    with pytest.raises(ValueError, match="sink roto"):
        pipeline.run(threading.Event())


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


def test_queue_maxsizes_must_match_stage_count() -> None:
    with pytest.raises(ValueError, match="queue_maxsizes"):
        FramePipeline(iter([]), [MapStage(lambda f: f)], lambda f: None, [2])


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
