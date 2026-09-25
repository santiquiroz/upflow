from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services import ep_registry
from app.services.engines.onnx_common import blend_tiles, blend_tiles_float
from app.services.engines.tiled_restore_runner import (
    TDR_BUDGET_REASON,
    CalibrationCache,
    CalibrationKey,
    CalibrationSpec,
    Padding,
    RestoreCancelled,
    TileCalibration,
    TilePlan,
    calibrate_tile,
    crop_padding,
    pad_to_requirements,
    predict_call_ms,
    run_tiled,
    session_tile_infer,
)
from seam_detector import measure_seams

SEAM_THRESHOLD = 2.0


class _IoInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class Identity1xSession:
    def __init__(self) -> None:
        self.input_shapes: list[tuple[int, ...]] = []

    def get_inputs(self) -> list[_IoInfo]:
        return [_IoInfo("input")]

    def get_outputs(self) -> list[_IoInfo]:
        return [_IoInfo("output")]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = input_feed["input"]
        self.input_shapes.append(batch.shape)
        return [batch.copy()]


class BiasedPerTile1xSession(Identity1xSession):
    """Suma un sesgo que alterna por llamada: sin mezcla en el solape, dos tiles
    vecinos dejan un escalon visible."""

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = input_feed["input"]
        bias = 0.04 if len(self.input_shapes) % 2 == 0 else -0.04
        self.input_shapes.append(batch.shape)
        return [batch + np.float32(bias)]


class Nearest2xSession(Identity1xSession):
    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = input_feed["input"]
        self.input_shapes.append(batch.shape)
        return [batch.repeat(2, axis=2).repeat(2, axis=3)]


class FirstThreeChannels1xSession(Identity1xSession):
    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch = input_feed["input"]
        self.input_shapes.append(batch.shape)
        return [batch[:, :3].copy()]


def make_smooth_image(width: int, height: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    red = 90 + 80 * np.sin(xx / 97) + 40 * np.cos(yy / 61)
    green = 110 + 70 * np.cos((xx + yy) / 83)
    blue = 80 + 60 * np.sin(xx / 51) * np.cos(yy / 71)
    noise = np.random.default_rng(7).normal(0, 3, (height, width, 3))
    rgb = np.clip(np.stack([red, green, blue], axis=-1) + noise, 0, 255)
    return (rgb / 255.0).astype(np.float32)


def make_16bit_image(height: int, width: int, channels: int = 3) -> tuple[np.ndarray, np.ndarray]:
    raw = np.random.default_rng(11).integers(0, 65536, (height, width, channels), dtype=np.uint16)
    return raw, raw.astype(np.float32) / np.float32(65535.0)


def to_uint16(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------- padding


def test_pad_reaches_minimum_and_multiple_and_crop_restores_exactly() -> None:
    image = np.random.default_rng(1).random((5, 13, 3)).astype(np.float32)

    padded, padding = pad_to_requirements(image, minimum=8, multiple=4, square=False)

    assert padded.shape == (8, 16, 3)
    assert padding == Padding(bottom=3, right=3)
    assert np.array_equal(crop_padding(padded, padding), image)


def test_pad_square_uses_the_longest_side_rounded_to_the_multiple() -> None:
    image = np.zeros((5, 13, 3), dtype=np.float32)

    padded, padding = pad_to_requirements(image, minimum=1, multiple=4, square=True)

    assert padded.shape == (16, 16, 3)
    assert padding == Padding(bottom=11, right=3)


def test_pad_reflects_without_repeating_the_edge() -> None:
    image = np.arange(4, dtype=np.float32).reshape(1, 4, 1).repeat(2, axis=0)

    padded, _ = pad_to_requirements(image, minimum=1, multiple=6, square=False)

    assert padded[0, :, 0].tolist() == [0, 1, 2, 3, 2, 1]


def test_pad_is_a_noop_when_requirements_already_hold() -> None:
    image = np.ones((8, 8, 3), dtype=np.float32)

    padded, padding = pad_to_requirements(image, minimum=8, multiple=8, square=True)

    assert padded is image
    assert padding == Padding(bottom=0, right=0)


def test_crop_padding_scales_the_padding_for_nx_outputs() -> None:
    upscaled = np.zeros((16, 32, 3), dtype=np.float32)

    cropped = crop_padding(upscaled, Padding(bottom=3, right=5), scale=2)

    assert cropped.shape == (10, 22, 3)


def test_pad_rejects_non_positive_requirements() -> None:
    with pytest.raises(ValueError):
        pad_to_requirements(np.zeros((4, 4, 3), dtype=np.float32), minimum=1, multiple=0, square=False)


# ---------------------------------------------------------------- blend_tiles_float


def test_blend_tiles_float_keeps_values_outside_0_255_and_unrounded() -> None:
    tile = np.full((4, 4, 3), 0.123456, dtype=np.float32)
    tiles = [(0, 0, 4, 4, tile), (0, 4, 4, 4, tile)]

    blended = blend_tiles_float(tiles, 4, 8, 3, 1, feather=0)

    assert blended.dtype == np.float32
    assert np.allclose(blended, 0.123456, atol=1e-7)


def test_blend_tiles_float_does_not_clip_negative_or_above_one() -> None:
    tile = np.full((4, 4, 1), -0.5, dtype=np.float32)

    blended = blend_tiles_float([(0, 0, 4, 4, tile)], 4, 4, 1, 1)

    assert np.all(blended == np.float32(-0.5))


def test_blend_tiles_is_the_uint8_finalization_of_blend_tiles_float() -> None:
    rng = np.random.default_rng(3)
    tiles = [
        (0, 0, 8, 8, rng.uniform(-20, 280, (16, 16, 3)).astype(np.float32)),
        (0, 4, 8, 8, rng.uniform(-20, 280, (16, 16, 3)).astype(np.float32)),
    ]

    as_uint8 = blend_tiles(tiles, 8, 12, 3, 2, feather=8)
    as_float = blend_tiles_float(tiles, 8, 12, 3, 2, feather=8)

    assert as_uint8.dtype == np.uint8
    assert np.array_equal(as_uint8, np.rint(np.clip(as_float, 0, 255)).astype(np.uint8))


# ---------------------------------------------------------------- runner: identidad y forma fija


def test_identity_1x_round_trips_16_bit_input_bit_for_bit() -> None:
    raw, image = make_16bit_image(150, 203)
    session = Identity1xSession()

    output = run_tiled(session_tile_infer(session), image, TilePlan(tile=64, overlap=16))

    assert output.dtype == np.float32
    assert output.shape == image.shape
    assert np.array_equal(to_uint16(output), raw)
    assert float(np.max(np.abs(output - image))) < 1e-6


def test_every_tile_has_the_fixed_shape_even_on_ragged_or_small_images() -> None:
    session = Identity1xSession()
    image = make_16bit_image(37, 101)[1]

    output = run_tiled(session_tile_infer(session), image, TilePlan(tile=64, overlap=16, multiple=8))

    assert output.shape == image.shape
    assert set(session.input_shapes) == {(1, 3, 64, 64)}


def test_nx_model_output_is_cropped_to_the_exact_scaled_size() -> None:
    session = Nearest2xSession()
    image = make_16bit_image(45, 70)[1]

    output = run_tiled(session_tile_infer(session), image, TilePlan(tile=32, overlap=8))

    assert output.shape == (90, 140, 3)
    expected = image.repeat(2, axis=0).repeat(2, axis=1)
    assert float(np.max(np.abs(output - expected))) < 1e-6


@pytest.mark.parametrize(
    ("tile", "overlap", "multiple"),
    [(0, 0, 1), (64, 64, 1), (64, -1, 1), (60, 16, 8)],
)
def test_tile_plan_rejects_inconsistent_geometry(tile: int, overlap: int, multiple: int) -> None:
    with pytest.raises(ValueError):
        TilePlan(tile=tile, overlap=overlap, multiple=multiple)


# ---------------------------------------------------------------- costuras


def test_feathered_blend_hides_per_tile_bias_below_the_seam_threshold() -> None:
    image = make_smooth_image(256, 160)
    plan = TilePlan(tile=64, overlap=16)

    output = run_tiled(session_tile_infer(BiasedPerTile1xSession()), image, plan)

    report = measure_seams(to_uint8(output), plan.tile - plan.overlap)
    assert report.worst_ratio < SEAM_THRESHOLD, report


def test_without_overlap_the_detector_sees_the_per_tile_bias() -> None:
    image = make_smooth_image(256, 192)
    plan = TilePlan(tile=64, overlap=0)

    output = run_tiled(session_tile_infer(BiasedPerTile1xSession()), image, plan)

    report = measure_seams(to_uint8(output), plan.tile)
    assert report.worst_ratio > 3, report


# ---------------------------------------------------------------- canales 1<->3


def test_gray_image_is_replicated_for_a_color_model_and_averaged_back() -> None:
    session = Identity1xSession()
    raw, gray = make_16bit_image(40, 50, channels=1)

    output = run_tiled(session_tile_infer(session), gray[:, :, 0], TilePlan(tile=32, overlap=8, channels=3))

    assert output.shape == (40, 50)
    assert set(session.input_shapes) == {(1, 3, 32, 32)}
    assert np.array_equal(to_uint16(output), raw[:, :, 0])


def test_gray_hwc1_image_keeps_its_single_channel_axis() -> None:
    gray = make_16bit_image(20, 20, channels=1)[1]

    output = run_tiled(session_tile_infer(Identity1xSession()), gray, TilePlan(tile=16, overlap=4, channels=3))

    assert output.shape == (20, 20, 1)


def test_color_image_on_a_gray_model_runs_one_pass_per_channel() -> None:
    session = Identity1xSession()
    raw, image = make_16bit_image(40, 50)

    output = run_tiled(session_tile_infer(session), image, TilePlan(tile=32, overlap=8, channels=1))

    assert output.shape == (40, 50, 3)
    assert set(session.input_shapes) == {(1, 1, 32, 32)}
    assert np.array_equal(to_uint16(output), raw)


def test_extra_input_channels_pass_through_when_they_match_the_model() -> None:
    rgb = make_16bit_image(30, 30)[1]
    strength = np.full((30, 30, 1), 0.5, dtype=np.float32)
    session = FirstThreeChannels1xSession()

    output = run_tiled(
        session_tile_infer(session), np.concatenate([rgb, strength], axis=2), TilePlan(tile=16, overlap=4, channels=4)
    )

    assert output.shape == (30, 30, 3)
    assert set(session.input_shapes) == {(1, 4, 16, 16)}
    assert float(np.max(np.abs(output - rgb))) < 1e-6


def test_unsupported_channel_combination_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-channel image"):
        run_tiled(
            session_tile_infer(Identity1xSession()),
            np.zeros((8, 8, 2), dtype=np.float32),
            TilePlan(tile=8, overlap=2, channels=3),
        )


# ---------------------------------------------------------------- progreso y cancelacion


def test_progress_reports_every_tile_monotonically_up_to_the_total() -> None:
    calls: list[tuple[int, int]] = []
    image = make_16bit_image(100, 100)[1]

    run_tiled(
        session_tile_infer(Identity1xSession()),
        image,
        TilePlan(tile=64, overlap=16),
        on_progress=lambda d, t: calls.append((d, t)),
    )

    assert calls == [(done, 4) for done in range(1, 5)]


def test_progress_total_counts_the_passes_of_a_gray_model_on_color_input() -> None:
    calls: list[tuple[int, int]] = []
    image = make_16bit_image(100, 100)[1]

    run_tiled(
        session_tile_infer(Identity1xSession()),
        image,
        TilePlan(tile=64, overlap=16, channels=1),
        on_progress=lambda d, t: calls.append((d, t)),
    )

    assert calls == [(done, 12) for done in range(1, 13)]


def test_cancellation_between_tiles_stops_before_the_next_run() -> None:
    session = Identity1xSession()
    cancel = threading.Event()

    def cancel_after_second(done: int, total: int) -> None:
        if done == 2:
            cancel.set()

    with pytest.raises(RestoreCancelled):
        run_tiled(
            session_tile_infer(session),
            make_16bit_image(100, 100)[1],
            TilePlan(tile=64, overlap=16),
            cancel_event=cancel,
            on_progress=cancel_after_second,
        )

    assert len(session.input_shapes) == 2


def test_an_already_cancelled_job_never_runs_the_model() -> None:
    session = Identity1xSession()
    cancel = threading.Event()
    cancel.set()

    with pytest.raises(RestoreCancelled):
        run_tiled(
            session_tile_infer(session),
            make_16bit_image(20, 20)[1],
            TilePlan(tile=16, overlap=4),
            cancel_event=cancel,
        )

    assert session.input_shapes == []


# ---------------------------------------------------------------- grafo ONNX real en el CPU EP


def write_identity_graph(path: Path) -> None:
    from onnx import TensorProto, helper, save

    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "identity1x",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "h", "w"])],
    )
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(path))


def test_real_identity_graph_on_the_cpu_ep_round_trips_16_bit_input(tmp_path: Path) -> None:
    model_path = tmp_path / "identity1x.onnx"
    write_identity_graph(model_path)
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    session = ep_registry.create_session(str(model_path), "cpu", settings)
    raw, image = make_16bit_image(70, 90)

    output = run_tiled(session_tile_infer(session), image, TilePlan(tile=64, overlap=16, multiple=8))

    assert np.array_equal(to_uint16(output), raw)


# ---------------------------------------------------------------- calibracion TDR con reloj falso

BUDGET_MS = 1200.0
COMPILE_MS = 5000.0


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance_ms(self, ms: float) -> None:
        self.seconds += ms / 1000.0


class TimedFakeModel:
    """Cada forma nueva paga COMPILE_MS en su primera llamada, como DML con H/W
    dinamicos; despues cuesta lo que diga ms_for_tile."""

    def __init__(self, clock: FakeClock, ms_for_tile: Callable[[int], float]) -> None:
        self.clock = clock
        self.ms_for_tile = ms_for_tile
        self.calls: list[tuple[int, ...]] = []
        self.compiled: set[int] = set()

    def __call__(self, tile: np.ndarray) -> np.ndarray:
        size = tile.shape[0]
        self.calls.append(tile.shape)
        compile_ms = 0.0 if size in self.compiled else COMPILE_MS
        self.compiled.add(size)
        self.clock.advance_ms(compile_ms + self.ms_for_tile(size))
        return tile

    def sizes(self) -> list[int]:
        return [shape[0] for shape in self.calls]


def per_mpx(ms_per_mpx: float) -> Callable[[int], float]:
    return lambda size: ms_per_mpx * size * size / 1_000_000


def calibrate_fake(model: TimedFakeModel, spec: CalibrationSpec, precision: str = "fp16") -> TileCalibration:
    return calibrate_tile(model, spec, precision=precision, budget_ms=BUDGET_MS, clock=model.clock)


DYNAMIC_SPEC = CalibrationSpec(tile_min=128, tile_candidates=(256, 384, 512))


def test_calibration_excludes_the_compile_warmup_from_the_timing() -> None:
    model = TimedFakeModel(FakeClock(), per_mpx(1500))

    result = calibrate_fake(model, DYNAMIC_SPEC)

    assert result.tile == 512
    assert result.ms_per_mpx == pytest.approx(1500)
    assert model.sizes()[0] == 128


def test_calibration_measures_at_least_two_stable_calls_and_keeps_the_slowest() -> None:
    steady = iter([0.0, 20.0, 40.0])
    model = TimedFakeModel(FakeClock(), lambda size: next(steady))

    result = calibrate_fake(model, CalibrationSpec(tile_min=128, fixed_shape=True))

    assert model.sizes() == [128, 128, 128]
    assert result.ms_per_mpx == pytest.approx(40.0 / (128 * 128 / 1_000_000))


def test_calibration_extrapolates_by_pixels_to_the_largest_tile_within_half_budget() -> None:
    model = TimedFakeModel(FakeClock(), per_mpx(3000))

    result = calibrate_fake(model, DYNAMIC_SPEC)

    assert predict_call_ms(3000, 512) > BUDGET_MS / 2 >= predict_call_ms(3000, 384)
    assert result.tile == 384
    assert 512 not in model.sizes()
    assert result.cpu_fallback_reason is None


def test_calibration_warms_and_verifies_the_chosen_tile_with_two_calls() -> None:
    model = TimedFakeModel(FakeClock(), per_mpx(3000))

    calibrate_fake(model, DYNAMIC_SPEC)

    assert model.sizes().count(384) == 3


def test_calibration_steps_down_when_a_verification_call_exceeds_three_quarters_of_budget() -> None:
    linear = per_mpx(1500)
    model = TimedFakeModel(FakeClock(), lambda size: 0.76 * BUDGET_MS if size == 512 else linear(size))

    result = calibrate_fake(model, DYNAMIC_SPEC)

    assert result.tile == 384
    assert model.sizes().index(512) < model.sizes().index(384)


def test_calibration_that_steps_down_below_the_minimum_tile_falls_back_to_cpu() -> None:
    model = TimedFakeModel(FakeClock(), lambda size: 0.8 * BUDGET_MS)

    result = calibrate_fake(model, DYNAMIC_SPEC)

    assert result.runs_on_cpu
    assert result.tile is None
    assert result.cpu_fallback_reason == TDR_BUDGET_REASON


def test_calibration_of_a_minimum_tile_slower_than_the_budget_falls_back_to_cpu() -> None:
    model = TimedFakeModel(FakeClock(), lambda size: BUDGET_MS + 1)

    result = calibrate_fake(model, DYNAMIC_SPEC)

    assert result.runs_on_cpu
    assert result.cpu_fallback_reason == TDR_BUDGET_REASON
    assert set(model.sizes()) == {128}


def test_calibration_of_a_fixed_shape_model_over_half_budget_runs_on_cpu() -> None:
    model = TimedFakeModel(FakeClock(), lambda size: BUDGET_MS / 2 + 1)

    result = calibrate_fake(model, CalibrationSpec(tile_min=512, fixed_shape=True))

    assert result.runs_on_cpu
    assert result.cpu_fallback_reason == TDR_BUDGET_REASON
    assert set(model.sizes()) == {512}


def test_calibration_of_a_fixed_shape_model_within_half_budget_keeps_its_shape() -> None:
    model = TimedFakeModel(FakeClock(), lambda size: BUDGET_MS / 2)

    result = calibrate_fake(model, CalibrationSpec(tile_min=512, tile_candidates=(256, 1024), fixed_shape=True))

    assert result.tile == 512
    assert set(model.sizes()) == {512}


def test_calibration_never_goes_above_the_ceiling() -> None:
    model = TimedFakeModel(FakeClock(), per_mpx(100))

    result = calibrate_fake(model, CalibrationSpec(tile_min=128, tile_candidates=(256, 384, 512), ceiling=384))

    assert result.tile == 384
    assert 512 not in model.sizes()


def test_calibration_ignores_candidates_below_the_minimum_tile() -> None:
    model = TimedFakeModel(FakeClock(), lambda size: 0.8 * BUDGET_MS)

    result = calibrate_fake(model, CalibrationSpec(tile_min=128, tile_candidates=(64, 256)))

    assert result.runs_on_cpu
    assert 64 not in model.sizes()


def test_calibration_probes_with_the_model_channel_count() -> None:
    model = TimedFakeModel(FakeClock(), per_mpx(100))

    calibrate_fake(model, CalibrationSpec(tile_min=64, fixed_shape=True, channels=1))

    assert set(model.calls) == {(64, 64, 1)}


def test_calibration_of_fp32_gets_its_own_tile_never_the_fp16_one() -> None:
    cache = CalibrationCache()
    fp16 = TimedFakeModel(FakeClock(), per_mpx(1500))
    fp32 = TimedFakeModel(FakeClock(), per_mpx(1500 * 7.26))

    fast = cache.get_or_calibrate(CalibrationKey("drunet", "dml:0", "fp16"), lambda: calibrate_fake(fp16, DYNAMIC_SPEC))
    slow = cache.get_or_calibrate(
        CalibrationKey("drunet", "dml:0", "fp32"), lambda: calibrate_fake(fp32, DYNAMIC_SPEC, precision="fp32")
    )

    assert (fast.precision, fast.tile) == ("fp16", 512)
    assert (slow.precision, slow.tile) == ("fp32", 128)


def test_calibration_runs_once_per_model_device_and_precision() -> None:
    cache = CalibrationCache()
    runs: list[str] = []

    def calibrate() -> TileCalibration:
        runs.append("run")
        return TileCalibration(precision="fp16", tile=256, ms_per_mpx=1.0)

    key = CalibrationKey("drunet", "dml:0", "fp16")
    first = cache.get_or_calibrate(key, calibrate)
    again = cache.get_or_calibrate(key, calibrate)
    cache.get_or_calibrate(CalibrationKey("drunet", "dml:1", "fp16"), calibrate)
    cache.get_or_calibrate(CalibrationKey("gfpgan", "dml:0", "fp16"), calibrate)

    assert again is first
    assert len(runs) == 3
    assert cache.get(key) is first


def test_calibration_cache_rejects_a_result_of_another_precision() -> None:
    cache = CalibrationCache()
    fp16_result = TileCalibration(precision="fp16", tile=512, ms_per_mpx=1.0)
    fp32_key = CalibrationKey("drunet", "dml:0", "fp32")

    with pytest.raises(ValueError, match="precision"):
        cache.get_or_calibrate(fp32_key, lambda: fp16_result)

    assert cache.get(fp32_key) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_min": 0},
        {"tile_min": 128, "channels": 0},
        {"tile_min": 128, "tile_candidates": (0, 256)},
        {"tile_min": 256, "ceiling": 128},
    ],
)
def test_calibration_spec_rejects_inconsistent_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CalibrationSpec(**kwargs)


@pytest.mark.parametrize("budget_ms", [0.0, -5.0])
def test_calibration_rejects_a_non_positive_budget(budget_ms: float) -> None:
    model = TimedFakeModel(FakeClock(), per_mpx(100))

    with pytest.raises(ValueError, match="budget"):
        calibrate_tile(model, DYNAMIC_SPEC, precision="fp16", budget_ms=budget_ms, clock=model.clock)

    assert model.calls == []


def test_calibration_rejects_an_unknown_precision() -> None:
    model = TimedFakeModel(FakeClock(), per_mpx(100))

    with pytest.raises(ValueError, match="precision"):
        calibrate_tile(model, DYNAMIC_SPEC, precision="int8", budget_ms=BUDGET_MS, clock=model.clock)

    assert model.calls == []
