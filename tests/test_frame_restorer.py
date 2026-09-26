from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from app.config import Settings
from app.services.engines.frame_model_runner import VIDEO_DEBLOCK_MODEL_ID, level_for_strength
from app.services.engines.frame_restorer import (
    OSD_EDGE_ALPHA,
    ComposedStage,
    FrameRestorer,
    build_composed_stage,
    osd_edge_mask,
    output_factor,
)
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.frame_pipeline import FramePipeline
from tests.test_frame_model_runner import RecordingCoordinator, video_spec, write_u8_plus_strength_graph

CPU = "cpu"


def frame(height: int = 24, width: int = 32, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 200, (1, height, width, 3), dtype=np.uint8)


class Spy:
    def __init__(self, transform, log: list[str], name: str, tracker: ConcurrencyTracker | None = None) -> None:
        self.transform = transform
        self.log = log
        self.name = name
        self.tracker = tracker
        self.calls = 0

    def __call__(self, frame_nhwc: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.log.append(self.name)
        if self.tracker is None:
            return self.transform(frame_nhwc)
        with self.tracker.running():
            return self.transform(frame_nhwc)


class ConcurrencyTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.threads: set[int] = set()

    def running(self):
        tracker = self

        class _Running:
            def __enter__(self) -> None:
                with tracker._lock:
                    tracker.active += 1
                    tracker.peak = max(tracker.peak, tracker.active)
                    tracker.threads.add(threading.get_ident())
                # Una pausa corta deja ver un solapamiento si dos run corrieran a la vez.
                time.sleep(0.002)

            def __exit__(self, *exc) -> None:
                with tracker._lock:
                    tracker.active -= 1

        return _Running()


def plus(amount: int):
    return lambda batch: np.clip(batch.astype(np.int16) + amount, 0, 255).astype(np.uint8)


def enlarge(factor: int):
    return lambda batch: np.repeat(np.repeat(batch, factor, axis=1), factor, axis=2)


def run_pipeline(stage: ComposedStage, frames: list[np.ndarray]) -> list[np.ndarray]:
    delivered: list[np.ndarray] = []
    pipeline = FramePipeline(iter(frames), [stage], delivered.append, [2, 2])
    pipeline.run(threading.Event())
    return delivered


# ---------------------------------------------------------------- composicion


def test_restore_then_upscale_run_in_that_order_on_each_frame() -> None:
    log: list[str] = []
    stage = build_composed_stage(Spy(plus(10), log, "restore"), Spy(enlarge(2), log, "upscale"))
    source = frame()

    output = stage.process(source)

    np.testing.assert_array_equal(output[0], enlarge(2)(plus(10)(source)))
    assert log == ["restore", "upscale"]


def test_without_upscale_the_stage_returns_the_restored_frame() -> None:
    stage = build_composed_stage(plus(5))
    source = frame()

    [output] = stage.process(source)

    np.testing.assert_array_equal(output, plus(5)(source))


def test_flush_emits_nothing_because_the_stage_is_one_to_one() -> None:
    assert list(build_composed_stage(plus(1)).flush()) == []


def test_restore_and_upscale_share_one_pipeline_stage_and_never_run_concurrently() -> None:
    tracker = ConcurrencyTracker()
    log: list[str] = []
    stage = build_composed_stage(
        Spy(plus(1), log, "restore", tracker), Spy(enlarge(2), log, "upscale", tracker)
    )
    frames = [frame(seed=seed) for seed in range(12)]

    delivered = run_pipeline(stage, frames)

    assert len(delivered) == 12
    assert tracker.peak == 1
    assert len(tracker.threads) == 1
    assert threading.get_ident() not in tracker.threads
    assert log == ["restore", "upscale"] * 12


def test_frames_come_out_in_input_order_through_the_pipeline() -> None:
    frames = [frame(seed=seed) for seed in range(6)]

    delivered = run_pipeline(build_composed_stage(plus(0)), frames)

    for source, output in zip(frames, delivered, strict=True):
        np.testing.assert_array_equal(output, source)


# ---------------------------------------------------------------- cuadros duplicados


def test_a_frame_byte_equal_to_the_previous_one_reuses_the_output_without_calling_the_models() -> None:
    log: list[str] = []
    restore = Spy(plus(10), log, "restore")
    upscale = Spy(enlarge(2), log, "upscale")
    stage = build_composed_stage(restore, upscale)
    source = frame()

    [first] = stage.process(source)
    [second] = stage.process(source.copy())

    np.testing.assert_array_equal(second, first)
    assert restore.calls == 1
    assert upscale.calls == 1
    assert stage.report().duplicates_reused == 1
    assert stage.report().frames == 2


def test_a_run_of_duplicates_counts_every_reused_frame() -> None:
    restore = Spy(plus(1), [], "restore")
    stage = build_composed_stage(restore)
    source = frame()

    for _ in range(4):
        stage.process(source)

    assert restore.calls == 1
    assert stage.report().duplicates_reused == 3


def test_a_changed_frame_after_a_duplicate_is_processed_again() -> None:
    restore = Spy(plus(1), [], "restore")
    stage = build_composed_stage(restore)
    changed = frame().copy()
    changed[0, 0, 0, 0] ^= 1

    stage.process(frame())
    stage.process(frame())
    [output] = stage.process(changed)

    assert restore.calls == 2
    np.testing.assert_array_equal(output, plus(1)(changed))


def test_only_the_previous_frame_is_compared_so_a_b_a_runs_the_model_three_times() -> None:
    restore = Spy(plus(1), [], "restore")
    stage = build_composed_stage(restore)

    for source in (frame(seed=1), frame(seed=2), frame(seed=1)):
        stage.process(source)

    assert restore.calls == 3
    assert stage.report().duplicates_reused == 0


def test_a_duplicate_does_not_reuse_the_output_of_a_frame_that_failed() -> None:
    calls = {"count": 0}

    def flaky(batch: np.ndarray) -> np.ndarray:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("model failed")
        return batch.copy()

    stage = build_composed_stage(flaky)
    source = frame()
    with pytest.raises(RuntimeError, match="model failed"):
        stage.process(source)

    [output] = stage.process(source)

    np.testing.assert_array_equal(output, source)
    assert stage.report().duplicates_reused == 0


# ---------------------------------------------------------------- OSD


def test_osd_pixels_come_from_the_decoded_frame_enlarged_by_nearest_neighbour() -> None:
    source = frame(24, 32)
    box = (4, 2, 10, 6)
    stage = build_composed_stage(plus(40), enlarge(2), osd_boxes=(box,))

    [output] = stage.process(source)

    x, y, w, h = box
    expected = enlarge(2)(source[:, y : y + h, x : x + w])
    interior = output[:, 2 * y + 1 : 2 * (y + h) - 1, 2 * x + 1 : 2 * (x + w) - 1]
    np.testing.assert_array_equal(interior, expected[:, 1:-1, 1:-1])


def test_osd_box_edges_are_blended_over_one_pixel_and_the_rest_keeps_the_model_output() -> None:
    source = frame(24, 32)
    box = (4, 2, 10, 6)
    stage = build_composed_stage(plus(40), enlarge(2), osd_boxes=(box,))
    processed = enlarge(2)(plus(40)(source))

    [output] = stage.process(source)

    x, y, w, h = box
    original = enlarge(2)(source)
    edge_row = 2 * y
    columns = slice(2 * x + 1, 2 * (x + w) - 1)
    alpha = OSD_EDGE_ALPHA / 255.0
    blended = original[0, edge_row, columns] * alpha + processed[0, edge_row, columns] * (1 - alpha)
    np.testing.assert_allclose(output[0, edge_row, columns], blended, atol=0.5 + 1e-3)
    outside = np.ones(output.shape[1:3], dtype=bool)
    outside[2 * y : 2 * (y + h), 2 * x : 2 * (x + w)] = False
    np.testing.assert_array_equal(output[0][outside], processed[0][outside])


def test_osd_boxes_without_upscale_are_pasted_at_the_same_scale() -> None:
    source = frame(24, 32)
    stage = build_composed_stage(plus(40), osd_boxes=((0, 0, 8, 4),))

    [output] = stage.process(source)

    np.testing.assert_array_equal(output[0, 1:3, 1:7], source[0, 1:3, 1:7])


def test_every_osd_box_is_pasted() -> None:
    source = frame(24, 32)
    boxes = ((0, 0, 8, 4), (20, 18, 10, 5))
    stage = build_composed_stage(plus(40), enlarge(2), osd_boxes=boxes)

    [output] = stage.process(source)

    for x, y, w, h in boxes:
        interior = output[0, 2 * y + 1 : 2 * (y + h) - 1, 2 * x + 1 : 2 * (x + w) - 1]
        expected = enlarge(2)(source[:, y : y + h, x : x + w])[0, 1:-1, 1:-1]
        np.testing.assert_array_equal(interior, expected)


def test_osd_paste_never_writes_into_the_model_output_buffer() -> None:
    held: list[np.ndarray] = []

    def upscale(batch: np.ndarray) -> np.ndarray:
        result = enlarge(2)(batch)
        held.append(result)
        return result

    source = frame(24, 32)
    stage = build_composed_stage(plus(40), upscale, osd_boxes=((0, 0, 8, 4),))

    [output] = stage.process(source)

    assert output is not held[0]
    np.testing.assert_array_equal(held[0], enlarge(2)(plus(40)(source)))


def test_a_duplicate_frame_reuses_the_output_with_the_osd_already_pasted() -> None:
    source = frame(24, 32)
    stage = build_composed_stage(plus(40), enlarge(2), osd_boxes=((0, 0, 8, 4),))

    [first] = stage.process(source)
    [second] = stage.process(source.copy())

    np.testing.assert_array_equal(second, first)


@pytest.mark.parametrize("box", [(0, 0, 0, 4), (0, 0, 4, 0), (-1, 0, 4, 4), (0, -2, 4, 4)])
def test_osd_boxes_with_negative_origin_or_empty_size_are_rejected(box) -> None:
    with pytest.raises(ValueError, match="OSD box"):
        build_composed_stage(plus(0), osd_boxes=(box,))


@pytest.mark.parametrize("box", [(28, 0, 8, 4), (0, 22, 4, 4)])
def test_an_osd_box_outside_the_frame_is_an_error(box) -> None:
    stage = build_composed_stage(plus(0), osd_boxes=(box,))

    with pytest.raises(ValueError, match="outside"):
        stage.process(frame(24, 32))


def test_osd_edge_mask_is_opaque_inside_and_soft_on_a_one_pixel_border() -> None:
    mask = osd_edge_mask(4, 5)

    assert mask.dtype == np.uint8
    assert (mask[1:-1, 1:-1] == 255).all()
    for edge in (mask[0], mask[-1], mask[:, 0], mask[:, -1]):
        assert (edge == OSD_EDGE_ALPHA).all()


@pytest.mark.parametrize(
    ("source", "output", "factor"),
    [((24, 32), (24, 32), 1), ((24, 32), (48, 64), 2), ((24, 32), (96, 128), 4)],
)
def test_output_factor_is_the_integer_scale_between_input_and_output(source, output, factor) -> None:
    assert output_factor(source, output) == factor


@pytest.mark.parametrize("output", [(36, 48), (48, 96), (50, 64)])
def test_output_factor_rejects_non_integer_or_anisotropic_scales(output) -> None:
    with pytest.raises(ValueError, match="scale"):
        output_factor((24, 32), output)


# ---------------------------------------------------------------- FrameRestorer


class RecordingRunner:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def __call__(self, frame_nhwc: np.ndarray) -> np.ndarray:
        self.log.append("restore")
        return frame_nhwc.copy()

    def report(self) -> str:
        return "restore-report"


def test_frame_restorer_builds_the_restore_runner_before_the_upscaler() -> None:
    log: list[str] = []
    seen: dict = {}

    def runner_builder(engine, device, sample, level, *, cancel_event=None):
        log.append("build-restore")
        seen.update(engine=engine, device=device, sample=sample, level=level, cancel_event=cancel_event)
        return RecordingRunner(log)

    def upscaler_factory():
        log.append("build-upscale")
        return enlarge(2)

    engine = object()
    cancel = threading.Event()
    sample = frame()
    restorer = FrameRestorer(engine, runner_builder=runner_builder)

    stage = restorer.build_stage(
        "dml:0", sample, 60, upscaler_factory=upscaler_factory, cancel_event=cancel
    )
    [output] = stage.process(sample)

    assert log == ["build-restore", "build-upscale", "restore"]
    assert seen["engine"] is engine
    assert seen["device"] == "dml:0"
    assert seen["sample"] is sample
    assert seen["level"] == level_for_strength(60)
    assert seen["cancel_event"] is cancel
    assert output.shape == (1, 48, 64, 3)


def test_frame_restorer_report_carries_the_restore_runner_report() -> None:
    restorer = FrameRestorer(object(), runner_builder=lambda *args, **kwargs: RecordingRunner([]))
    stage = restorer.build_stage(CPU, frame(), 40, osd_boxes=((0, 0, 4, 4),))

    stage.process(frame())
    report = stage.report()

    assert report.restore == "restore-report"
    assert report.upscaled is False
    assert report.osd_boxes == 1
    assert report.frames == 1


def test_frame_restorer_rejects_a_strength_outside_0_100_before_touching_the_device() -> None:
    def runner_builder(*args, **kwargs):
        raise AssertionError("the runner must not be built")

    with pytest.raises(ValueError, match="Strength"):
        FrameRestorer(object(), runner_builder=runner_builder).build_stage(CPU, frame(), 120)


def test_real_uint8_graph_on_the_cpu_ep_through_the_composed_stage(tmp_path) -> None:
    spec = video_spec()
    model_dir = tmp_path / "restore"
    model_dir.mkdir()
    write_u8_plus_strength_graph(model_dir / spec.filename)
    settings = Settings(_env_file=None, RESTORE_MODEL_DIR=str(model_dir), RUNTIME_DIR=str(tmp_path / "runtime"))
    engine = PhotoRestoreEngine(settings, RecordingCoordinator(), models={spec.id: spec})
    source = frame(30, 44, seed=12)
    box = (2, 2, 12, 6)
    stage = FrameRestorer(engine).build_stage(
        CPU, source, 40, upscaler_factory=lambda: enlarge(2), osd_boxes=(box,)
    )

    delivered = run_pipeline(stage, [source, source.copy(), frame(30, 44, seed=13)])

    shifted = source.astype(np.float32) + np.float32(level_for_strength(40)) * np.float32(100)
    processed = enlarge(2)(np.clip(np.rint(shifted), 0, 255).astype(np.uint8))
    np.testing.assert_array_equal(delivered[0][0, 20:, 40:], processed[0, 20:, 40:])
    np.testing.assert_array_equal(delivered[0][0, 5:15, 5:27], enlarge(2)(source)[0, 5:15, 5:27])
    np.testing.assert_array_equal(delivered[1], delivered[0])
    assert stage.report().duplicates_reused == 1
    assert stage.report().restore.model_id == VIDEO_DEBLOCK_MODEL_ID
