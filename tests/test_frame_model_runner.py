from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.services.devices_service import DevicesService
from app.services.engines.drunet_restore import deblock_level
from app.services.engines.frame_model_runner import (
    VIDEO_DEBLOCK_MODEL_ID,
    TdrBudget,
    TdrBudgetExceeded,
    build_frame_model_runner,
    canary_crop,
    level_for_strength,
    oom_tile_ladder,
    pad_frame,
)
from app.services.engines.photo_restore_engine import DeviceRemovedError, PhotoRestoreEngine
from app.services.engines.tiled_restore_runner import RestoreCancelled
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.restore_models import RestoreModelSpec

GPU = "dml:0"
CPU = "cpu"
OOM_TEXT = "D3D12 failed to allocate memory: out of memory"
DEVICE_REMOVED_TEXT = "Non-zero status code returned while running Conv node. 887A0005 DXGI_ERROR_DEVICE_REMOVED"
FP32_FILE = f"{VIDEO_DEBLOCK_MODEL_ID}.onnx"
FP16_FILE = f"{VIDEO_DEBLOCK_MODEL_ID}-fp16.onnx"


def video_spec(*, fp16: bool = False, ceiling: dict[str, int] | None = None) -> RestoreModelSpec:
    return RestoreModelSpec(
        id=VIDEO_DEBLOCK_MODEL_ID,
        name="DRUNet",
        bundle="core",
        filename=FP32_FILE,
        fp16_filename=FP16_FILE if fp16 else None,
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
        tile_min=16,
        tile_candidates=(16, 24, 30, 32),
        overlap=4,
        tile_by_precision=ceiling or {},
    )


def identity(batch: np.ndarray, level: float) -> np.ndarray:
    return batch.copy()


def plus_level(batch: np.ndarray, level: float) -> np.ndarray:
    return np.clip(batch.astype(np.int16) + round(level * 100), 0, 255).astype(np.uint8)


def brown(batch: np.ndarray, level: float) -> np.ndarray:
    return np.broadcast_to(np.array([115, 77, 51], dtype=np.uint8), batch.shape).copy()


def raising(message: str):
    def transform(batch: np.ndarray, level: float) -> np.ndarray:
        raise RuntimeError(message)

    return transform


def raising_above(pixels: int, message: str = OOM_TEXT):
    def transform(batch: np.ndarray, level: float) -> np.ndarray:
        if batch.shape[1] * batch.shape[2] > pixels:
            raise RuntimeError(message)
        return batch.copy()

    return transform


class FakeOrtValue:
    def __init__(self, array: np.ndarray, device_type: str, device_id: int) -> None:
        self.array = array
        self.device_type = device_type
        self.device_id = device_id


class FakeBinding:
    def __init__(self) -> None:
        self.inputs: dict[str, object] = {}
        self.cpu_inputs: dict[str, np.ndarray] = {}
        self.outputs: list[tuple] = []
        self.result: np.ndarray | None = None

    def bind_ortvalue_input(self, name: str, value: FakeOrtValue) -> None:
        self.inputs[name] = value

    def bind_cpu_input(self, name: str, array: np.ndarray) -> None:
        self.cpu_inputs[name] = array

    def bind_output(self, name: str, device_type: str = "cpu", device_id: int = 0) -> None:
        self.outputs.append((name, device_type, device_id))

    def copy_outputs_to_cpu(self) -> list[np.ndarray]:
        return [self.result]


class U8Session:
    def __init__(self, transform, *, binding_error: str | None = None) -> None:
        self.transform = transform
        self.binding_error = binding_error
        self.runs: list[dict[str, np.ndarray]] = []
        self.bindings: list[FakeBinding] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input"), SimpleNamespace(name="strength")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, output_names, feeds):
        assert output_names == ["output"]
        self.runs.append(feeds)
        return [self.transform(feeds["input"], float(feeds["strength"]))]

    def io_binding(self) -> FakeBinding:
        binding = FakeBinding()
        self.bindings.append(binding)
        return binding

    def run_with_iobinding(self, binding: FakeBinding) -> None:
        if self.binding_error is not None:
            raise RuntimeError(self.binding_error)
        frame = binding.inputs["input"].array
        binding.result = self.transform(frame, float(binding.cpu_inputs["strength"]))

    def shapes(self) -> list[tuple[int, ...]]:
        bound = [binding.inputs["input"].array.shape for binding in self.bindings if "input" in binding.inputs]
        return [feeds["input"].shape for feeds in self.runs] + bound


class U8SessionFactory:
    def __init__(self, transforms: dict[str, object] | None = None, **session_kwargs) -> None:
        self.transforms = transforms or {}
        self.session_kwargs = session_kwargs
        self.calls: list[dict] = []
        self.sessions: list[tuple[str, str, U8Session]] = []

    def __call__(self, model_path: str, device: str, settings: Settings, **kwargs) -> U8Session:
        name = Path(model_path).name
        self.calls.append({"name": name, "device": device, **kwargs})
        session = U8Session(self.transforms.get(name, identity), **self.session_kwargs)
        self.sessions.append((name, device, session))
        return session

    def session_for(self, name: str, device: str) -> U8Session:
        return next(session for file, dev, session in self.sessions if file == name and dev == device)


class RecordingCoordinator:
    def __init__(self) -> None:
        self.invalidated: list[str] = []

    def register(self, owner: object) -> None:
        pass

    def acquire(self, device: str, owner: object) -> None:
        pass

    def invalidate_device(self, device: str) -> None:
        self.invalidated.append(device)


class OtherOwner:
    def release_device(self, device: str) -> None:
        pass


def make_engine(tmp_path: Path, factory, spec: RestoreModelSpec, *, coordinator=None, health=None, **overrides):
    model_dir = tmp_path / "restore"
    model_dir.mkdir(exist_ok=True)
    for filename in spec.files_by_precision().values():
        (model_dir / filename).write_bytes(b"\0" * 1024)
    settings = Settings(_env_file=None, RESTORE_MODEL_DIR=str(model_dir), **overrides)
    return PhotoRestoreEngine(
        settings,
        coordinator or RecordingCoordinator(),
        models={spec.id: spec},
        create_session=factory,
        free_vram_mb=lambda device: None,
        device_health=health or DevicesService(settings),
    )


def frame(height: int = 30, width: int = 44, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (1, height, width, 3), dtype=np.uint8)


def whole_frame(*args) -> None:
    return None


def build(engine, device=CPU, sample=None, level=0.4, **kwargs):
    # La calibracion TDR hace sus propias llamadas: se prueba aparte para no contar sus corridas aca.
    kwargs.setdefault("start_tile_for", whole_frame)
    return build_frame_model_runner(
        engine, device, frame() if sample is None else sample, level, ortvalue_factory=FakeOrtValue, **kwargs
    )


class PixelClock:
    def __init__(self, ms_per_pixel: float) -> None:
        self.ms_per_pixel = ms_per_pixel
        self.now_ms = 0.0

    def __call__(self) -> float:
        return self.now_ms / 1000.0

    def transform(self, batch: np.ndarray, level: float) -> np.ndarray:
        self.now_ms += batch.shape[1] * batch.shape[2] * self.ms_per_pixel
        return batch.copy()


def build_with_tdr(tmp_path, ms_per_pixel: float, sample):
    clock = PixelClock(ms_per_pixel)
    factory = U8SessionFactory({FP32_FILE: clock.transform})
    engine = make_engine(tmp_path, factory, video_spec())
    runner = build_frame_model_runner(
        engine, GPU, sample, 0.4, ortvalue_factory=FakeOrtValue, tdr_budget=TdrBudget(1200.0, clock)
    )
    return runner, factory.session_for(FP32_FILE, GPU)


# ---------------------------------------------------------------- escala de la fuerza


@pytest.mark.parametrize("strength", [0, 40, 60, 100])
def test_strength_percent_uses_the_deblock_map_scale(strength) -> None:
    assert level_for_strength(strength) == pytest.approx(deblock_level(strength / 100))


@pytest.mark.parametrize("strength", [-1, 101])
def test_strength_percent_outside_0_100_is_rejected(strength) -> None:
    with pytest.raises(ValueError):
        level_for_strength(strength)


@pytest.mark.parametrize("level", [-0.1, 1.1, float("nan")])
def test_a_level_outside_zero_to_one_is_rejected(tmp_path, level) -> None:
    engine = make_engine(tmp_path, U8SessionFactory(), video_spec())

    with pytest.raises(ValueError):
        build(engine, level=level)


# ---------------------------------------------------------------- cuadro entero, forma fija


def test_whole_frame_runs_once_padded_to_a_multiple_of_8_with_the_scalar_strength(tmp_path) -> None:
    factory = U8SessionFactory()
    runner = build(make_engine(tmp_path, factory, video_spec()), level=0.39)

    output = runner(frame(30, 44, seed=1))

    session = factory.session_for(FP32_FILE, CPU)
    feeds = session.runs[-1]
    assert feeds["input"].shape == (1, 32, 48, 3)
    assert feeds["input"].dtype == np.uint8
    assert feeds["input"].flags["C_CONTIGUOUS"]
    assert feeds["strength"].shape == ()
    assert feeds["strength"].dtype == np.float32
    assert float(feeds["strength"]) == pytest.approx(0.39)
    assert output.shape == (1, 30, 44, 3)
    assert output.dtype == np.uint8


def test_padding_is_cropped_back_so_an_identity_model_returns_the_frame(tmp_path) -> None:
    runner = build(make_engine(tmp_path, U8SessionFactory(), video_spec()))
    source = frame(30, 44, seed=2)

    np.testing.assert_array_equal(runner(source), source)


def test_the_model_output_reaches_the_caller(tmp_path) -> None:
    factory = U8SessionFactory({FP32_FILE: plus_level})
    source = np.full((1, 16, 24, 3), 100, dtype=np.uint8)
    runner = build(make_engine(tmp_path, factory, video_spec()), sample=source, level=0.2)

    np.testing.assert_array_equal(runner(source), np.full_like(source, 120))


def test_a_frame_with_another_shape_is_rejected_because_the_shape_is_fixed_per_job(tmp_path) -> None:
    runner = build(make_engine(tmp_path, U8SessionFactory(), video_spec()), sample=frame(30, 44))

    with pytest.raises(ValueError, match="fixed"):
        runner(frame(32, 44))


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((1, 30, 44, 3), dtype=np.float32),
        np.zeros((2, 30, 44, 3), dtype=np.uint8),
        np.zeros((1, 30, 44, 4), dtype=np.uint8),
        np.zeros((30, 44), dtype=np.uint8),
    ],
)
def test_frames_that_are_not_uint8_nhwc_rgb_are_rejected(tmp_path, bad) -> None:
    engine = make_engine(tmp_path, U8SessionFactory(), video_spec())

    with pytest.raises(ValueError):
        build(engine, sample=bad)


def test_an_hwc_frame_is_accepted_and_answered_as_nhwc(tmp_path) -> None:
    runner = build(make_engine(tmp_path, U8SessionFactory(), video_spec()), sample=frame()[0])

    output = runner(frame()[0])

    assert output.shape == (1, 30, 44, 3)


def test_the_session_is_resolved_once_and_reused_for_every_frame(tmp_path) -> None:
    factory = U8SessionFactory()
    runner = build(make_engine(tmp_path, factory, video_spec()))

    for seed in range(4):
        runner(frame(seed=seed))

    assert [call["name"] for call in factory.calls] == [FP32_FILE]
    assert len(factory.session_for(FP32_FILE, CPU).runs) == 4


def test_sessions_are_created_with_prefer_native_false(tmp_path) -> None:
    factory = U8SessionFactory()

    build(make_engine(tmp_path, factory, video_spec()))

    assert factory.calls and all(call["prefer_native"] is False for call in factory.calls)


def test_the_held_session_survives_another_owner_taking_the_device(tmp_path) -> None:
    # En la etapa compuesta el escalador ONNX toma el device despues de armar el runner.
    coordinator = GpuSessionCoordinator()
    factory = U8SessionFactory()
    runner = build(make_engine(tmp_path, factory, video_spec(), coordinator=coordinator), device=GPU)

    coordinator.acquire(GPU, OtherOwner())
    source = frame(seed=3)

    np.testing.assert_array_equal(runner(source), source)


def test_a_session_whose_names_break_the_video_contract_is_refused(tmp_path) -> None:
    class Renamed(U8Session):
        def get_inputs(self):
            return [SimpleNamespace(name="frame"), SimpleNamespace(name="strength")]

    class RenamedFactory(U8SessionFactory):
        def __call__(self, model_path, device, settings, **kwargs):
            return Renamed(identity)

    with pytest.raises(RuntimeError, match="input"):
        build(make_engine(tmp_path, RenamedFactory(), video_spec()))


def test_a_model_output_with_the_wrong_shape_is_an_error(tmp_path) -> None:
    factory = U8SessionFactory({FP32_FILE: lambda batch, level: batch[:, :8]})
    runner = build(make_engine(tmp_path, factory, video_spec()))

    with pytest.raises(RuntimeError, match="shape"):
        runner(frame())


def test_a_cancelled_job_stops_before_the_next_frame(tmp_path) -> None:
    cancel = threading.Event()
    factory = U8SessionFactory()
    runner = build(make_engine(tmp_path, factory, video_spec()), cancel_event=cancel)
    cancel.set()

    with pytest.raises(RestoreCancelled):
        runner(frame())
    assert factory.session_for(FP32_FILE, CPU).runs == []


def test_the_report_says_whole_frame_precision_and_device(tmp_path) -> None:
    runner = build(make_engine(tmp_path, U8SessionFactory(), video_spec()))

    report = runner.report()

    assert (report.model_id, report.device, report.precision) == (VIDEO_DEBLOCK_MODEL_ID, CPU, "fp32")
    assert report.tile is None
    assert report.io_binding is False


# ---------------------------------------------------------------- IOBinding


def test_iobinding_on_dml_binds_the_frame_on_the_device_and_the_strength_on_the_host(tmp_path) -> None:
    factory = U8SessionFactory({FP32_FILE: plus_level})
    source = np.full((1, 16, 24, 3), 50, dtype=np.uint8)
    runner = build(make_engine(tmp_path, factory, video_spec()), device=GPU, sample=source, level=0.1)

    output = runner(source)

    session = factory.session_for(FP32_FILE, GPU)
    binding = session.bindings[-1]
    value = binding.inputs["input"]
    assert (value.device_type, value.device_id) == ("dml", 0)
    assert value.array.shape == (1, 16, 24, 3)
    assert float(binding.cpu_inputs["strength"]) == pytest.approx(0.1)
    assert binding.outputs == [("output", "dml", 0)]
    assert session.runs == []
    np.testing.assert_array_equal(output, np.full_like(source, 60))
    assert runner.report().io_binding is True


def test_cpu_frames_never_use_iobinding(tmp_path) -> None:
    factory = U8SessionFactory()
    runner = build(make_engine(tmp_path, factory, video_spec()))

    runner(frame())

    assert factory.session_for(FP32_FILE, CPU).bindings == []


def test_an_iobinding_failure_falls_back_to_plain_run_for_the_rest_of_the_job(tmp_path) -> None:
    factory = U8SessionFactory(binding_error="IOBinding is not supported by this EP")
    runner = build(make_engine(tmp_path, factory, video_spec()), device=GPU)
    first, second = frame(seed=4), frame(seed=5)

    np.testing.assert_array_equal(runner(first), first)
    np.testing.assert_array_equal(runner(second), second)

    session = factory.session_for(FP32_FILE, GPU)
    assert len(session.bindings) == 1
    assert len(session.runs) == 2
    assert runner.report().io_binding is False


def test_an_iobinding_oom_goes_to_tiles_instead_of_a_plain_whole_frame_run(tmp_path) -> None:
    factory = U8SessionFactory(binding_error=OOM_TEXT)
    runner = build(make_engine(tmp_path, factory, video_spec()), device=GPU)

    with pytest.raises(RuntimeError, match="out of memory"):
        runner(frame())

    session = factory.session_for(FP32_FILE, GPU)
    assert session.runs == []
    assert session.shapes()[:2] == [(1, 32, 48, 3), (1, 32, 32, 3)]


# ---------------------------------------------------------------- OOM -> tiles


def test_a_whole_frame_oom_retries_the_frame_in_tiles_and_stays_tiled(tmp_path) -> None:
    factory = U8SessionFactory({FP32_FILE: raising_above(32 * 32)})
    runner = build(make_engine(tmp_path, factory, video_spec()), sample=frame(30, 44))
    first, second = frame(seed=8), frame(seed=9)

    np.testing.assert_array_equal(runner(first), first)
    np.testing.assert_array_equal(runner(second), second)

    shapes = factory.session_for(FP32_FILE, CPU).shapes()
    assert shapes[0] == (1, 32, 48, 3)
    assert set(shapes[1:]) == {(1, 32, 32, 3)}
    assert runner.report().tile == 32


def test_an_oom_on_tiles_steps_down_the_tile_ladder(tmp_path) -> None:
    factory = U8SessionFactory({FP32_FILE: raising_above(24 * 24)})
    runner = build(make_engine(tmp_path, factory, video_spec()), sample=frame(30, 44))
    source = frame(seed=10)

    np.testing.assert_array_equal(runner(source), source)

    assert runner.report().tile == 24


def test_an_oom_below_the_smallest_tile_fails_the_job(tmp_path) -> None:
    factory = U8SessionFactory({FP32_FILE: raising_above(8 * 8)})
    runner = build(make_engine(tmp_path, factory, video_spec()))

    with pytest.raises(RuntimeError, match="out of memory"):
        runner(frame())


def test_a_non_oom_error_is_not_retried_in_tiles(tmp_path) -> None:
    factory = U8SessionFactory({FP32_FILE: raising("Invalid rank for input")})
    runner = build(make_engine(tmp_path, factory, video_spec()))

    with pytest.raises(RuntimeError, match="Invalid rank"):
        runner(frame())

    assert len(factory.session_for(FP32_FILE, CPU).runs) == 1


def test_oom_tiles_never_exceed_the_precision_ceiling_nor_the_frame() -> None:
    spec = video_spec(ceiling={"fp32": 24})

    assert oom_tile_ladder(spec, "fp32", (32, 48)) == (24, 16)
    assert oom_tile_ladder(video_spec(), "fp32", (16, 24)) == (16,)
    assert oom_tile_ladder(video_spec(), "fp32", (16, 16)) == ()


def test_oom_tiles_skip_sizes_that_are_not_multiples_of_8() -> None:
    assert oom_tile_ladder(video_spec(), "fp32", (64, 64)) == (32, 24, 16)


def test_a_cancel_between_tiles_stops_the_frame(tmp_path) -> None:
    cancel = threading.Event()

    def cancel_on_first_tile(batch, level):
        if batch.shape[1] * batch.shape[2] > 32 * 32:
            raise RuntimeError(OOM_TEXT)
        cancel.set()
        return batch.copy()

    factory = U8SessionFactory({FP32_FILE: cancel_on_first_tile})
    runner = build(make_engine(tmp_path, factory, video_spec()), cancel_event=cancel)

    with pytest.raises(RestoreCancelled):
        runner(frame())


# ---------------------------------------------------------------- remocion del device


def test_a_device_removal_marks_the_device_and_invalidates_every_owner(tmp_path) -> None:
    coordinator = RecordingCoordinator()
    settings = Settings(_env_file=None)
    health = DevicesService(settings)
    factory = U8SessionFactory({FP32_FILE: raising(DEVICE_REMOVED_TEXT)})
    engine = make_engine(tmp_path, factory, video_spec(), coordinator=coordinator, health=health)
    runner = build(engine, device=GPU)

    with pytest.raises(DeviceRemovedError):
        runner(frame())

    assert coordinator.invalidated == [GPU]
    assert not health.is_healthy(GPU)
    assert factory.session_for(FP32_FILE, GPU).bindings[0].result is None


def test_a_device_removal_is_never_retried_in_tiles(tmp_path) -> None:
    factory = U8SessionFactory({FP32_FILE: raising(DEVICE_REMOVED_TEXT)})
    runner = build(make_engine(tmp_path, factory, video_spec()), device=GPU)

    with pytest.raises(DeviceRemovedError):
        runner(frame())

    session = factory.session_for(FP32_FILE, GPU)
    assert len(session.bindings) + len(session.runs) == 1


def test_a_removed_device_refuses_to_build_a_runner(tmp_path) -> None:
    health = DevicesService(Settings(_env_file=None))
    health.mark_unhealthy(GPU)
    engine = make_engine(tmp_path, U8SessionFactory(), video_spec(), health=health)

    with pytest.raises(DeviceRemovedError):
        build(engine, device=GPU)


# ---------------------------------------------------------------- canario fp16


def test_canary_brown_fp16_is_rejected_and_the_job_runs_in_fp32(tmp_path) -> None:
    factory = U8SessionFactory({FP16_FILE: brown})
    engine = make_engine(tmp_path, factory, video_spec(fp16=True))

    runner = build(engine, device=GPU)
    source = frame(seed=11)

    np.testing.assert_array_equal(runner(source), source)
    assert runner.report().precision == "fp32"
    assert [rejection.model_id for rejection in engine.fp16_rejections()] == [VIDEO_DEBLOCK_MODEL_ID]


def test_canary_compares_a_crop_against_the_cpu_reference_not_the_whole_frame(tmp_path) -> None:
    factory = U8SessionFactory()
    engine = make_engine(tmp_path, factory, video_spec(fp16=True))

    build(engine, device=GPU, sample=frame(40, 60))

    fp16_first = factory.session_for(FP16_FILE, GPU).runs[0]["input"]
    reference = factory.session_for(FP32_FILE, CPU).runs[0]["input"]
    assert fp16_first.shape == reference.shape == (1, 16, 16, 3)


def test_canary_good_fp16_keeps_fp16_and_uses_its_session_for_frames(tmp_path) -> None:
    factory = U8SessionFactory()
    runner = build(make_engine(tmp_path, factory, video_spec(fp16=True)), device=GPU)

    runner(frame())

    assert runner.report().precision == "fp16"
    assert len(factory.session_for(FP16_FILE, GPU).bindings) == 1


def test_canary_scores_uint8_outputs_on_the_unit_scale(tmp_path) -> None:
    # Un nivel de diferencia en el 1% de los pixeles es ~68 dB: tiene que pasar el umbral de 50 dB.
    def one_level_off(batch, level):
        out = batch.copy()
        flat = out.reshape(-1)
        flat[:: 100] = np.where(flat[::100] < 255, flat[::100] + 1, flat[::100] - 1)
        return out

    factory = U8SessionFactory({FP16_FILE: one_level_off})
    runner = build(make_engine(tmp_path, factory, video_spec(fp16=True)), device=GPU)

    assert runner.report().precision == "fp16"


def test_fp16_is_not_used_when_the_setting_prefers_fp32(tmp_path) -> None:
    factory = U8SessionFactory()
    runner = build(make_engine(tmp_path, factory, video_spec(fp16=True), ONNX_PREFER_FP16=False), device=GPU)

    assert runner.report().precision == "fp32"
    assert [call["name"] for call in factory.calls] == [FP32_FILE]


# ---------------------------------------------------------------- funciones puras


def test_pad_frame_reflects_up_to_a_multiple_of_8() -> None:
    image = frame(30, 44)[0]

    padded, padding = pad_frame(image)

    assert padded.shape == (32, 48, 3)
    assert (padding.bottom, padding.right) == (2, 4)
    np.testing.assert_array_equal(padded[:30, :44], image)


def test_canary_crop_is_centered_and_padded_to_a_multiple_of_8() -> None:
    image = frame(40, 60)[0]

    crop = canary_crop(image, 16)

    assert crop.shape == (16, 16, 3)
    np.testing.assert_array_equal(crop, image[12:28, 22:38])
    assert canary_crop(frame(10, 12)[0], 16).shape == (16, 16, 3)


# ---------------------------------------------------------------- grafo uint8 NHWC real en el CPU EP


def write_u8_plus_strength_graph(path: Path) -> None:
    from onnx import TensorProto, helper, save

    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["input"], ["pixels"], to=TensorProto.FLOAT),
            helper.make_node("Mul", ["strength", "hundred"], ["offset"]),
            helper.make_node("Add", ["pixels", "offset"], ["shifted"]),
            helper.make_node("Round", ["shifted"], ["rounded"]),
            helper.make_node("Clip", ["rounded", "zero", "peak"], ["clamped"]),
            helper.make_node("Cast", ["clamped"], ["output"], to=TensorProto.UINT8),
        ],
        "u8_plus_strength",
        [
            helper.make_tensor_value_info("input", TensorProto.UINT8, [1, "height", "width", 3]),
            helper.make_tensor_value_info("strength", TensorProto.FLOAT, []),
        ],
        [helper.make_tensor_value_info("output", TensorProto.UINT8, [1, "height", "width", 3])],
        initializer=[
            helper.make_tensor("hundred", TensorProto.FLOAT, [], [100.0]),
            helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0]),
            helper.make_tensor("peak", TensorProto.FLOAT, [], [255.0]),
        ],
    )
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(path))


def test_real_uint8_nhwc_graph_with_a_scalar_strength_on_the_cpu_ep(tmp_path) -> None:
    spec = video_spec()
    model_dir = tmp_path / "restore"
    model_dir.mkdir()
    write_u8_plus_strength_graph(model_dir / spec.filename)
    settings = Settings(_env_file=None, RESTORE_MODEL_DIR=str(model_dir), RUNTIME_DIR=str(tmp_path / "runtime"))
    engine = PhotoRestoreEngine(settings, RecordingCoordinator(), models={spec.id: spec})
    source = np.clip(frame(30, 44, seed=12).astype(np.int16), 0, 200).astype(np.uint8)

    runner = build_frame_model_runner(engine, CPU, source, 0.3)
    output = runner(source)

    np.testing.assert_array_equal(output, source + 30)
    assert output.shape == (1, 30, 44, 3)


# ---------------------------------------------------------------- presupuesto TDR (no verificado en GPU)


def test_a_frame_whose_whole_call_would_pass_the_tdr_budget_starts_in_the_largest_verified_tile(tmp_path) -> None:
    source = frame(30, 44)
    runner, session = build_with_tdr(tmp_path, 0.5, source)
    calibration_calls = len(session.shapes())

    np.testing.assert_array_equal(runner(source), source)

    frame_shapes = session.shapes()[calibration_calls:]
    assert (1, 32, 48, 3) not in session.shapes()
    assert set(frame_shapes) == {(1, 32, 32, 3)}
    report = runner.report()
    assert (report.tile, report.tile_reason) == (32, "tdrBudget")


def test_a_fast_model_keeps_the_whole_frame_after_the_tdr_calibration(tmp_path) -> None:
    source = frame(30, 44)
    runner, session = build_with_tdr(tmp_path, 0.01, source)
    calibration_calls = len(session.shapes())

    runner(source)

    assert session.shapes()[calibration_calls:] == [(1, 32, 48, 3)]
    assert (runner.report().tile, runner.report().tile_reason) == (None, None)


def test_a_model_too_slow_even_for_the_smallest_tile_refuses_to_start(tmp_path) -> None:
    with pytest.raises(TdrBudgetExceeded, match="driver timeout"):
        build_with_tdr(tmp_path, 10.0, frame(30, 44))


def test_the_cpu_skips_the_tdr_calibration(tmp_path) -> None:
    clock = PixelClock(10.0)
    factory = U8SessionFactory({FP32_FILE: clock.transform})
    engine = make_engine(tmp_path, factory, video_spec())

    runner = build_frame_model_runner(engine, CPU, frame(), 0.4, tdr_budget=TdrBudget(1200.0, clock))

    assert factory.session_for(FP32_FILE, CPU).shapes() == []
    assert runner.report().tile is None


def test_an_oom_after_the_start_reports_oom_as_the_tile_reason(tmp_path) -> None:
    factory = U8SessionFactory({FP32_FILE: raising_above(32 * 32)})
    runner = build(make_engine(tmp_path, factory, video_spec()), sample=frame(30, 44))

    runner(frame(seed=8))

    assert (runner.report().tile, runner.report().tile_reason) == (32, "oom")


def test_a_device_removed_during_the_tdr_calibration_is_classified(tmp_path) -> None:
    coordinator = RecordingCoordinator()
    factory = U8SessionFactory({FP32_FILE: raising(DEVICE_REMOVED_TEXT)})
    engine = make_engine(tmp_path, factory, video_spec(), coordinator=coordinator)

    with pytest.raises(DeviceRemovedError):
        build_frame_model_runner(engine, GPU, frame(), 0.4, ortvalue_factory=FakeOrtValue)

    assert coordinator.invalidated == [GPU]
