from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.services.engines.drunet_restore import (
    DEBLOCK_MODEL_ID,
    DENOISE_MODEL_ID,
    CpuFallback,
    DrunetContext,
    canary_sample,
    deblock_level,
    deblock_photo,
    denoise_level,
    denoise_photo,
    denoise_sigma,
    run_drunet,
    with_strength_channel,
)
from app.services.engines.photo_restore_engine import NonFiniteOutputError, PhotoRestoreEngine
from app.services.engines.tiled_restore_runner import (
    TDR_BUDGET_REASON,
    CalibrationCache,
    CalibrationKey,
    TileCalibration,
)
from app.services.restore_models import RestoreModelSpec

GPU = "dml:0"
CPU = "cpu"


class OrtLikeSession:
    def __init__(self, transform) -> None:
        self.transform = transform
        self.batches: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, output_names, feeds):
        batch = feeds["input"]
        self.batches.append(batch)
        return [self.transform(batch)]


def strength_echo(batch: np.ndarray) -> np.ndarray:
    # StrengthEchoSession: cada canal de salida es el 4.o canal de la entrada.
    return np.repeat(batch[:, 3:4], 3, axis=1)


def rgb_passthrough(batch: np.ndarray) -> np.ndarray:
    return batch[:, :3].copy()


def per_channel_offsets(batch: np.ndarray) -> np.ndarray:
    offsets = np.array([0.0, 0.1, 0.2], dtype=np.float32).reshape(1, 3, 1, 1)
    return batch[:, :3] + offsets


def constant(value: float):
    return lambda batch: np.full_like(batch[:, :3], value)


def out_of_range(batch: np.ndarray) -> np.ndarray:
    return batch[:, :3] * 3.0 - 1.0


def with_nan(batch: np.ndarray) -> np.ndarray:
    out = batch[:, :3].copy()
    out[0, 0, 0, 0] = np.nan
    return out


class SessionFactory:
    def __init__(self, transform=rgb_passthrough, clock=None, ms_per_call: float = 0.0) -> None:
        self.transform = transform
        self.clock = clock
        self.ms_per_call = ms_per_call
        self.sessions: list[tuple[str, str, OrtLikeSession]] = []

    def __call__(self, model_path: str, device: str, settings: Settings, **kwargs) -> OrtLikeSession:
        session = OrtLikeSession(self._timed(self.transform))
        self.sessions.append((Path(model_path).name, device, session))
        return session

    def _timed(self, transform):
        def run(batch):
            if self.clock is not None:
                self.clock.advance_ms(self.ms_per_call)
            return transform(batch)

        return run

    def devices(self) -> list[str]:
        return [device for _, device, _ in self.sessions]

    def batches_on(self, device: str) -> list[np.ndarray]:
        return [batch for _, dev, session in self.sessions if dev == device for batch in session.batches]


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance_ms(self, ms: float) -> None:
        self.seconds += ms / 1000.0


class CountingCoordinator:
    def register(self, owner: object) -> None:
        pass

    def acquire(self, device: str, owner: object) -> None:
        pass

    def invalidate_device(self, device: str) -> None:
        pass


def drunet_spec(model_id: str, *, fp16: bool = False, ceiling: dict[str, int] | None = None) -> RestoreModelSpec:
    return RestoreModelSpec(
        id=model_id,
        name="DRUNet",
        bundle="core",
        filename=f"{model_id}.onnx",
        fp16_filename=f"{model_id}-fp16.onnx" if fp16 else None,
        license_spdx="MIT",
        license_url="https://example.com/LICENSE",
        copyright="Copyright (c) test",
        attribution="Test model",
        data_lineage="D1a",
        commercial_use="yes",
        source_url="https://example.com/model",
        source_revision="abc123",
        source_sha256="a" * 64,
        modifications=("exported to ONNX",),
        tile_min=64,
        tile_candidates=(64, 96, 128),
        overlap=16,
        tile_by_precision=ceiling or {},
    )


def make_context(tmp_path: Path, factory, *specs: RestoreModelSpec, clock=None, **overrides) -> DrunetContext:
    model_dir = tmp_path / "restore"
    model_dir.mkdir(exist_ok=True)
    for spec in specs:
        for filename in spec.files_by_precision().values():
            (model_dir / filename).write_bytes(b"\0" * 1024)
    values = {"RESTORE_MODEL_DIR": str(model_dir), "RUNTIME_DIR": str(tmp_path / "runtime"), **overrides}
    settings = Settings(_env_file=None, **values)
    engine = PhotoRestoreEngine(
        settings, CountingCoordinator(), models={spec.id: spec for spec in specs}, create_session=factory
    )
    engine.begin_phase(GPU)
    engine.begin_phase(CPU)
    return DrunetContext(
        engine=engine,
        calibrations=CalibrationCache(),
        budget_ms=float(settings.restore_call_budget_ms),
        clock=clock or FakeClock(),
    )


def smooth_photo(height: int = 90, width: int = 130) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([x / width, y / height, (x + y) / (width + height)], axis=2).astype(np.float32)


# ---------------------------------------------------------------- escala del 4.o canal


@pytest.mark.parametrize(("strength", "level"), [(0.0, 0.05), (0.5, 0.475), (1.0, 0.90)])
def test_deblock_strength_maps_onto_the_trained_jpeg_quality_range(strength, level) -> None:
    assert deblock_level(strength) == pytest.approx(level)


@pytest.mark.parametrize(
    ("strength", "estimated_sigma", "sigma"),
    [(1.0, 20 / 255, 20.0), (0.5, 20 / 255, 10.0), (0.1, 5 / 255, 2.0), (1.0, 90 / 255, 50.0)],
)
def test_denoise_sigma_is_strength_times_the_estimate_bounded_to_2_50(strength, estimated_sigma, sigma) -> None:
    assert denoise_sigma(strength, estimated_sigma) == pytest.approx(sigma)
    assert denoise_level(strength, estimated_sigma) == pytest.approx(sigma / 255.0)


@pytest.mark.parametrize("strength", [-0.01, 1.01, float("nan")])
def test_strength_outside_zero_to_one_is_rejected(strength) -> None:
    with pytest.raises(ValueError):
        deblock_level(strength)
    with pytest.raises(ValueError):
        denoise_level(strength, 0.1)


def test_negative_noise_estimate_is_rejected() -> None:
    with pytest.raises(ValueError):
        denoise_sigma(0.5, -0.01)


def test_strength_channel_is_appended_as_a_constant_fourth_channel() -> None:
    tile = smooth_photo(16, 24)

    four = with_strength_channel(tile, 0.3)

    assert four.shape == (16, 24, 4)
    assert four.dtype == np.float32
    np.testing.assert_array_equal(four[:, :, :3], tile)
    np.testing.assert_array_equal(four[:, :, 3], np.float32(0.3))


def test_the_model_receives_the_agreed_level_in_its_fourth_channel(tmp_path) -> None:
    factory = SessionFactory(strength_echo)
    context = make_context(tmp_path, factory, drunet_spec(DEBLOCK_MODEL_ID))

    result = deblock_photo(context, smooth_photo(), 0.5, CPU)

    batches = factory.batches_on(CPU)
    assert batches and all(batch.shape[1] == 4 for batch in batches)
    np.testing.assert_allclose(result.image, 0.475, atol=1e-6)


def test_denoise_feeds_sigma_over_255_to_the_model(tmp_path) -> None:
    factory = SessionFactory(strength_echo)
    context = make_context(tmp_path, factory, drunet_spec(DENOISE_MODEL_ID))

    result = denoise_photo(context, smooth_photo(), 0.5, 30 / 255, CPU)

    np.testing.assert_allclose(result.image, 15.0 / 255.0, atol=1e-6)


# ---------------------------------------------------------------- gris replicado y promediado


def test_gray_photo_is_replicated_to_three_channels_and_the_output_averaged(tmp_path) -> None:
    factory = SessionFactory(per_channel_offsets)
    context = make_context(tmp_path, factory, drunet_spec(DEBLOCK_MODEL_ID))
    gray = smooth_photo()[:, :, 0] * 0.5

    result = deblock_photo(context, gray, 0.5, CPU)

    batch = factory.batches_on(CPU)[0]
    np.testing.assert_array_equal(batch[0, 0], batch[0, 1])
    np.testing.assert_array_equal(batch[0, 0], batch[0, 2])
    assert result.image.shape == gray.shape
    np.testing.assert_allclose(result.image, gray + 0.1, atol=1e-5)


def test_gray_hwc1_photo_keeps_its_single_channel_axis(tmp_path) -> None:
    context = make_context(tmp_path, SessionFactory(), drunet_spec(DEBLOCK_MODEL_ID))
    gray = smooth_photo()[:, :, :1]

    result = deblock_photo(context, gray, 0.5, CPU)

    assert result.image.shape == gray.shape
    np.testing.assert_allclose(result.image, gray, atol=1e-6)


# ---------------------------------------------------------------- clamp en el host despues de la guardia


def test_output_is_clamped_to_unit_range_on_the_host(tmp_path) -> None:
    context = make_context(tmp_path, SessionFactory(out_of_range), drunet_spec(DEBLOCK_MODEL_ID))

    result = deblock_photo(context, smooth_photo(), 0.5, CPU)

    assert result.image.min() == 0.0
    assert result.image.max() == 1.0


def test_nan_is_caught_by_the_guard_before_the_clamp_could_hide_it(tmp_path) -> None:
    context = make_context(tmp_path, SessionFactory(with_nan), drunet_spec(DEBLOCK_MODEL_ID))

    with pytest.raises(NonFiniteOutputError):
        deblock_photo(context, smooth_photo(), 0.5, CPU)


# ---------------------------------------------------------------- tile y precision


def test_cpu_runs_fp32_with_the_largest_tile_and_never_calibrates(tmp_path) -> None:
    factory = SessionFactory()
    context = make_context(tmp_path, factory, drunet_spec(DEBLOCK_MODEL_ID, fp16=True))

    result = deblock_photo(context, smooth_photo(200, 260), 0.5, CPU)

    assert (result.device, result.precision, result.tile) == (CPU, "fp32", 128)
    assert {batch.shape[2:] for batch in factory.batches_on(CPU)} == {(128, 128)}
    assert context.calibrations.get(CalibrationKey(DEBLOCK_MODEL_ID, CPU, "fp32")) is None


def test_gpu_tile_comes_from_calibration_capped_by_the_p0_gpu_ceiling(tmp_path) -> None:
    factory = SessionFactory()
    spec = drunet_spec(DEBLOCK_MODEL_ID, ceiling={"fp32": 96})
    context = make_context(tmp_path, factory, spec)

    result = deblock_photo(context, smooth_photo(200, 260), 0.5, GPU)

    assert (result.device, result.precision, result.tile) == (GPU, "fp32", 96)
    assert result.cpu_fallback is None
    calibration = context.calibrations.get(CalibrationKey(DEBLOCK_MODEL_ID, GPU, "fp32"))
    assert calibration is not None and calibration.tile == 96


# Calentamiento + 2 llamadas estables al tile minimo, y calentamiento + 2 de verificacion al elegido.
CALIBRATION_CALLS = 6


def test_gpu_calibration_runs_once_per_model_device_and_precision(tmp_path) -> None:
    factory = SessionFactory()
    context = make_context(tmp_path, factory, drunet_spec(DEBLOCK_MODEL_ID))
    deblock_photo(context, smooth_photo(), 0.5, GPU)
    first_run = len(factory.batches_on(GPU))

    deblock_photo(context, smooth_photo(), 0.5, GPU)

    second_run = len(factory.batches_on(GPU)) - first_run
    assert second_run == first_run - CALIBRATION_CALLS


def test_fp16_is_used_on_gpu_after_the_canary_against_a_cpu_reference(tmp_path) -> None:
    factory = SessionFactory()
    context = make_context(tmp_path, factory, drunet_spec(DEBLOCK_MODEL_ID, fp16=True))

    result = deblock_photo(context, smooth_photo(), 0.5, GPU)

    assert result.precision == "fp16"
    created = [(name, device) for name, device, _ in factory.sessions]
    assert (f"{DEBLOCK_MODEL_ID}-fp16.onnx", GPU) in created
    assert (f"{DEBLOCK_MODEL_ID}.onnx", CPU) in created, "without a calibrated fp32 tile the reference runs on CPU"
    assert context.calibrations.get(CalibrationKey(DEBLOCK_MODEL_ID, GPU, "fp16")) is not None


def test_fp16_canary_uses_the_gpu_fp32_reference_once_fp32_has_a_tile(tmp_path) -> None:
    factory = SessionFactory()
    context = make_context(tmp_path, factory, drunet_spec(DEBLOCK_MODEL_ID, fp16=True))
    fp32_key = CalibrationKey(DEBLOCK_MODEL_ID, GPU, "fp32")
    context.calibrations.get_or_calibrate(fp32_key, lambda: TileCalibration("fp32", 64, 100.0))

    result = deblock_photo(context, smooth_photo(), 0.5, GPU)

    assert result.precision == "fp16"
    assert CPU not in factory.devices()
    assert (f"{DEBLOCK_MODEL_ID}.onnx", GPU) in [(name, device) for name, device, _ in factory.sessions]


def test_canary_sample_is_a_three_channel_tile_of_the_minimum_size() -> None:
    gray = smooth_photo(40, 50)[:, :, 0]

    sample = canary_sample(gray, 64)

    assert sample.shape == (64, 64, 3)
    np.testing.assert_array_equal(sample[:, :, 0], sample[:, :, 2])


def test_a_model_too_slow_for_the_budget_falls_back_to_cpu_and_says_so(tmp_path) -> None:
    clock = FakeClock()
    factory = SessionFactory(clock=clock, ms_per_call=5000.0)
    context = make_context(tmp_path, factory, drunet_spec(DEBLOCK_MODEL_ID), clock=clock)

    result = deblock_photo(context, smooth_photo(), 0.5, GPU)

    assert result.device == CPU
    assert result.precision == "fp32"
    assert result.cpu_fallback == CpuFallback(DEBLOCK_MODEL_ID, TDR_BUDGET_REASON)
    assert result.cpu_fallback.to_metadata() == {"model": DEBLOCK_MODEL_ID, "reason": TDR_BUDGET_REASON}
    assert factory.batches_on(CPU)


# ---------------------------------------------------------------- keep grain


def test_keep_grain_blends_back_part_of_the_removed_residual(tmp_path) -> None:
    context = make_context(tmp_path, SessionFactory(constant(0.5)), drunet_spec(DENOISE_MODEL_ID))
    photo = smooth_photo()

    result = denoise_photo(context, photo, 0.5, 20 / 255, CPU, keep_grain_amount=0.25)

    np.testing.assert_allclose(result.image, 0.5 + 0.25 * (photo - 0.5), atol=1e-6)


def test_without_keep_grain_the_denoised_output_is_returned_as_is(tmp_path) -> None:
    context = make_context(tmp_path, SessionFactory(constant(0.5)), drunet_spec(DENOISE_MODEL_ID))

    result = denoise_photo(context, smooth_photo(), 0.5, 20 / 255, CPU)

    np.testing.assert_allclose(result.image, 0.5, atol=1e-6)


def test_run_drunet_rejects_a_level_outside_zero_to_one(tmp_path) -> None:
    context = make_context(tmp_path, SessionFactory(), drunet_spec(DEBLOCK_MODEL_ID))

    with pytest.raises(ValueError):
        run_drunet(context, DEBLOCK_MODEL_ID, smooth_photo(), 1.5, CPU)


# ---------------------------------------------------------------- grafo 4->3 real en el CPU EP


def write_rgb_plus_strength_graph(path: Path) -> None:
    from onnx import TensorProto, helper, save

    graph = helper.make_graph(
        [
            helper.make_node("Slice", ["input", "rgb_start", "rgb_end", "axis"], ["rgb"]),
            helper.make_node("Slice", ["input", "rgb_end", "map_end", "axis"], ["strength"]),
            helper.make_node("Add", ["rgb", "strength"], ["output"]),
        ],
        "rgb_plus_strength",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, "h", "w"])],
        initializer=[
            helper.make_tensor("rgb_start", TensorProto.INT64, [1], [0]),
            helper.make_tensor("rgb_end", TensorProto.INT64, [1], [3]),
            helper.make_tensor("map_end", TensorProto.INT64, [1], [4]),
            helper.make_tensor("axis", TensorProto.INT64, [1], [1]),
        ],
    )
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(path))


def test_real_four_to_three_channel_graph_on_the_cpu_ep(tmp_path) -> None:
    spec = drunet_spec(DEBLOCK_MODEL_ID)
    model_dir = tmp_path / "restore"
    model_dir.mkdir()
    write_rgb_plus_strength_graph(model_dir / spec.filename)
    settings = Settings(_env_file=None, RESTORE_MODEL_DIR=str(model_dir), RUNTIME_DIR=str(tmp_path / "runtime"))
    engine = PhotoRestoreEngine(settings, CountingCoordinator(), models={spec.id: spec})
    engine.begin_phase(CPU)
    context = DrunetContext(engine=engine, calibrations=CalibrationCache(), budget_ms=1200.0)
    photo = smooth_photo(70, 100)

    result = deblock_photo(context, photo, 0.0, CPU)

    np.testing.assert_allclose(result.image, np.clip(photo + 0.05, 0.0, 1.0), atol=1e-6)
