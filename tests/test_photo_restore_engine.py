from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.services.devices_service import DevicesService
from app.services.engines import photo_restore_engine
from app.services.engines.photo_restore_engine import (
    DeviceRemovedError,
    NonFiniteOutputError,
    PhotoRestoreEngine,
    SessionKey,
    clamp_unit,
    configure_session_options,
    dml_free_vram_mb,
    estimated_vram_mb,
    guard_finite,
    run_cancellable,
)
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.restore_models import RestoreModelSpec

SHA = "a" * 64
KIB = 1024
# Con 1 KiB de archivo, vram_factor = 1024 * N da N MiB estimados: tamanos chicos en disco.
MIB_PER_KIB_FACTOR = 1024.0


def make_spec(model_id: str, *, mib: int = 100, fp16: bool = False, disable_all: bool = False) -> RestoreModelSpec:
    return RestoreModelSpec(
        id=model_id,
        name=model_id.upper(),
        bundle="core",
        filename=f"{model_id}.onnx",
        fp16_filename=f"{model_id}-fp16.onnx" if fp16 else None,
        license_spdx="MIT",
        license_url="https://example.com/LICENSE",
        copyright="Copyright (c) test",
        attribution="Test model",
        data_lineage="D1b",
        commercial_use="yes",
        source_url="https://example.com/model",
        source_revision="abc123",
        source_sha256=SHA,
        modifications=("exported to ONNX",),
        tile_min=128,
        vram_factor=MIB_PER_KIB_FACTOR * mib,
        ort_disable_all=disable_all,
    )


class CountingCoordinator:
    def __init__(self) -> None:
        self.registered: list[object] = []
        self.acquired: list[tuple[str, object]] = []
        self.invalidated: list[str] = []

    def register(self, owner: object) -> None:
        self.registered.append(owner)

    def acquire(self, device: str, owner: object) -> None:
        self.acquired.append((device, owner))

    def invalidate_device(self, device: str) -> None:
        self.invalidated.append(device)


class SessionFactory:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, model_path: str, device: str, settings: Settings, **kwargs) -> object:
        self.calls.append({"path": model_path, "device": device, **kwargs})
        return SimpleNamespace(path=model_path, device=device)

    @property
    def paths(self) -> list[str]:
        return [Path(call["path"]).name for call in self.calls]


class OtherOwner:
    def release_device(self, device: str) -> None:
        pass


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "restore"
    directory.mkdir()
    return directory


def write_models(model_dir: Path, *specs: RestoreModelSpec) -> dict[str, RestoreModelSpec]:
    for spec in specs:
        for filename in spec.files_by_precision().values():
            (model_dir / filename).write_bytes(b"\0" * KIB)
    return {spec.id: spec for spec in specs}


def make_settings(model_dir: Path, **overrides) -> Settings:
    values = {"RESTORE_MODEL_DIR": str(model_dir), "RESTORE_MAX_LIVE_SESSIONS": 3, "RESTORE_SESSION_CACHE_MB": 3000}
    return Settings(_env_file=None, **{**values, **overrides})


def make_engine(model_dir: Path, models, coordinator=None, factory=None, free_vram=None, **overrides):
    return PhotoRestoreEngine(
        make_settings(model_dir, **overrides),
        coordinator or CountingCoordinator(),
        models=models,
        create_session=factory or SessionFactory(),
        free_vram_mb=free_vram or (lambda device: None),
    )


def live_models(engine: PhotoRestoreEngine, device: str = "dml:0") -> list[str]:
    return [key.model_id for key in engine.live_sessions(device)]


def test_session_owner_registers_once_with_the_coordinator(model_dir):
    coordinator = CountingCoordinator()
    engine = make_engine(model_dir, {}, coordinator=coordinator)

    assert coordinator.registered == [engine]


def test_phase_acquires_the_device_once_for_several_models(model_dir):
    models = write_models(model_dir, make_spec("drunet"), make_spec("deblock"))
    coordinator = CountingCoordinator()
    engine = make_engine(model_dir, models, coordinator=coordinator)

    engine.begin_phase("dml:0")
    engine.session("drunet", "dml:0", "fp32")
    engine.session("deblock", "dml:0", "fp32")
    engine.session("drunet", "dml:0", "fp32")

    assert coordinator.acquired == [("dml:0", engine)]


def test_phase_acquires_again_on_every_new_phase(model_dir):
    coordinator = CountingCoordinator()
    engine = make_engine(model_dir, {}, coordinator=coordinator)

    engine.begin_phase("dml:0")
    engine.begin_phase("dml:0")

    assert coordinator.acquired == [("dml:0", engine), ("dml:0", engine)]


def test_session_outside_a_phase_is_refused(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    factory = SessionFactory()
    engine = make_engine(model_dir, models, factory=factory)

    with pytest.raises(RuntimeError, match="begin_phase"):
        engine.session("drunet", "dml:0", "fp32")
    assert factory.calls == []


def test_eviction_by_another_owner_mid_phase_takes_the_device_back(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    coordinator = GpuSessionCoordinator()
    factory = SessionFactory()
    engine = make_engine(model_dir, models, coordinator=coordinator, factory=factory)
    engine.begin_phase("dml:0")
    engine.session("drunet", "dml:0", "fp32")
    other = RecordingOwner()

    coordinator.acquire("dml:0", other)
    assert live_models(engine) == []
    engine.session("drunet", "dml:0", "fp32")

    assert factory.paths == ["drunet.onnx", "drunet.onnx"]
    assert other.released == ["dml:0"]
    assert live_models(engine) == ["drunet"]


def test_eviction_reacquires_once_and_not_on_every_session(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    coordinator = CountingCoordinator()
    engine = make_engine(model_dir, models, coordinator=coordinator)
    engine.begin_phase("dml:0")
    engine.release_device("dml:0")

    engine.session("drunet", "dml:0", "fp32")
    engine.session("drunet", "dml:0", "fp32")

    assert coordinator.acquired == [("dml:0", engine), ("dml:0", engine)]


def test_an_evicted_device_that_was_removed_is_not_taken_back(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    health = DevicesService(make_settings(model_dir))
    engine = health_engine(model_dir, models, SessionFactory(), health=health)
    engine.begin_phase("dml:0")
    health.mark_unhealthy("dml:0")
    engine.release_device("dml:0")

    with pytest.raises(DeviceRemovedError):
        engine.session("drunet", "dml:0", "fp32")


def test_cpu_phase_stays_out_of_the_coordinator(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    coordinator = GpuSessionCoordinator()
    counting = CountingCoordinator()
    engine = make_engine(model_dir, models, coordinator=coordinator)
    cpu_only = make_engine(model_dir, models, coordinator=counting)
    engine.begin_phase("cpu")
    engine.session("drunet", "cpu", "fp32")
    cpu_only.begin_phase("cpu")

    coordinator.acquire("cpu", OtherOwner())

    assert live_models(engine, "cpu") == ["drunet"]
    assert counting.acquired == []


def test_phase_after_eviction_recreates_the_session(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    coordinator = GpuSessionCoordinator()
    factory = SessionFactory()
    engine = make_engine(model_dir, models, coordinator=coordinator, factory=factory)
    engine.begin_phase("dml:0")
    engine.session("drunet", "dml:0", "fp32")
    coordinator.acquire("dml:0", OtherOwner())

    engine.begin_phase("dml:0")
    engine.session("drunet", "dml:0", "fp32")

    assert factory.paths == ["drunet.onnx", "drunet.onnx"]


def test_session_is_created_with_prefer_native_false(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    factory = SessionFactory()
    engine = make_engine(model_dir, models, factory=factory)
    engine.begin_phase("dml:0")

    session = engine.session("drunet", "dml:0", "fp32")

    assert factory.calls[0]["prefer_native"] is False
    assert factory.calls[0]["device"] == "dml:0"
    assert Path(factory.calls[0]["path"]) == model_dir / "drunet.onnx"
    assert callable(factory.calls[0]["sess_options_factory"])
    assert session.path == factory.calls[0]["path"]


def test_session_uses_the_fp16_file_for_fp16(model_dir):
    models = write_models(model_dir, make_spec("drunet", fp16=True))
    factory = SessionFactory()
    engine = make_engine(model_dir, models, factory=factory)
    engine.begin_phase("dml:0")

    engine.session("drunet", "dml:0", "fp16")
    engine.session("drunet", "dml:0", "fp32")

    assert factory.paths == ["drunet-fp16.onnx", "drunet.onnx"]
    assert engine.live_sessions("dml:0") == (
        SessionKey("drunet", "dml:0", "fp16"),
        SessionKey("drunet", "dml:0", "fp32"),
    )


def test_session_for_a_precision_without_file_is_refused(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    engine = make_engine(model_dir, models)
    engine.begin_phase("dml:0")

    with pytest.raises(ValueError, match="fp16"):
        engine.session("drunet", "dml:0", "fp16")


def test_session_for_a_missing_file_names_the_pack(model_dir):
    models = {"drunet": make_spec("drunet")}
    factory = SessionFactory()
    engine = make_engine(model_dir, models, factory=factory)
    engine.begin_phase("dml:0")

    with pytest.raises(RuntimeError, match="restauración de fotos"):
        engine.session("drunet", "dml:0", "fp32")
    assert factory.calls == []


def test_session_for_an_unknown_model_is_refused(model_dir):
    engine = make_engine(model_dir, {})
    engine.begin_phase("dml:0")

    with pytest.raises(KeyError, match="nope"):
        engine.session("nope", "dml:0", "fp32")


def test_session_is_reused_while_cached(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    factory = SessionFactory()
    engine = make_engine(model_dir, models, factory=factory)
    engine.begin_phase("dml:0")

    first = engine.session("drunet", "dml:0", "fp32")
    second = engine.session("drunet", "dml:0", "fp32")

    assert first is second
    assert len(factory.calls) == 1


def test_lru_count_cap_evicts_the_least_recently_used(model_dir):
    models = write_models(model_dir, make_spec("a"), make_spec("b"), make_spec("c"))
    engine = make_engine(model_dir, models, RESTORE_MAX_LIVE_SESSIONS=2)
    engine.begin_phase("dml:0")

    engine.session("a", "dml:0", "fp32")
    engine.session("b", "dml:0", "fp32")
    engine.session("c", "dml:0", "fp32")

    assert live_models(engine) == ["b", "c"]


def test_lru_use_refreshes_a_session(model_dir):
    models = write_models(model_dir, make_spec("a"), make_spec("b"), make_spec("c"))
    engine = make_engine(model_dir, models, RESTORE_MAX_LIVE_SESSIONS=2)
    engine.begin_phase("dml:0")

    engine.session("a", "dml:0", "fp32")
    engine.session("b", "dml:0", "fp32")
    engine.session("a", "dml:0", "fp32")
    engine.session("c", "dml:0", "fp32")

    assert live_models(engine) == ["a", "c"]


def test_lru_estimated_vram_cap_evicts_until_the_new_session_fits(model_dir):
    models = write_models(
        model_dir, make_spec("a", mib=400), make_spec("b", mib=400), make_spec("c", mib=500)
    )
    engine = make_engine(model_dir, models, RESTORE_SESSION_CACHE_MB=1000)
    engine.begin_phase("dml:0")

    engine.session("a", "dml:0", "fp32")
    engine.session("b", "dml:0", "fp32")
    engine.session("c", "dml:0", "fp32")

    assert live_models(engine) == ["b", "c"]
    assert engine.estimated_vram_mb("dml:0") == pytest.approx(900)


def test_lru_evicts_before_creating_the_new_session(model_dir):
    models = write_models(model_dir, make_spec("a"), make_spec("b"))
    live_at_creation: list[int] = []
    factory = SessionFactory()

    def recording_factory(*args, **kwargs):
        live_at_creation.append(len(engine.live_sessions("dml:0")))
        return factory(*args, **kwargs)

    engine = make_engine(model_dir, models, factory=recording_factory, RESTORE_MAX_LIVE_SESSIONS=1)
    engine.begin_phase("dml:0")
    engine.session("a", "dml:0", "fp32")
    engine.session("b", "dml:0", "fp32")

    assert live_at_creation == [0, 0]


def test_lru_session_larger_than_the_cache_still_loads_alone(model_dir):
    models = write_models(model_dir, make_spec("small", mib=100), make_spec("huge", mib=5000))
    engine = make_engine(model_dir, models, RESTORE_SESSION_CACHE_MB=1000)
    engine.begin_phase("dml:0")

    engine.session("small", "dml:0", "fp32")
    engine.session("huge", "dml:0", "fp32")

    assert live_models(engine) == ["huge"]


def test_lru_caps_are_counted_per_device(model_dir):
    models = write_models(model_dir, make_spec("a"), make_spec("b"))
    engine = make_engine(model_dir, models, RESTORE_MAX_LIVE_SESSIONS=1)
    engine.begin_phase("dml:0")
    engine.begin_phase("dml:1")

    engine.session("a", "dml:0", "fp32")
    engine.session("b", "dml:1", "fp32")

    assert live_models(engine, "dml:0") == ["a"]
    assert live_models(engine, "dml:1") == ["b"]


def test_session_failed_creation_is_not_cached(model_dir):
    models = write_models(model_dir, make_spec("a"))
    attempts: list[str] = []

    def failing_factory(model_path, device, settings, **kwargs):
        attempts.append(model_path)
        raise RuntimeError("bad graph")

    engine = make_engine(model_dir, models, factory=failing_factory)
    engine.begin_phase("dml:0")

    with pytest.raises(RuntimeError, match="bad graph"):
        engine.session("a", "dml:0", "fp32")
    assert live_models(engine) == []


def test_session_release_device_frees_everything_on_that_device(model_dir):
    models = write_models(model_dir, make_spec("a"), make_spec("b"))
    engine = make_engine(model_dir, models)
    engine.begin_phase("dml:0")
    engine.begin_phase("dml:1")
    engine.session("a", "dml:0", "fp32")
    engine.session("b", "dml:0", "fp32")
    engine.session("a", "dml:1", "fp32")

    engine.release_device("dml:0")

    assert live_models(engine, "dml:0") == []
    assert live_models(engine, "dml:1") == ["a"]
    assert engine.estimated_vram_mb("dml:0") == 0


def test_ncnn_releases_sessions_when_free_vram_is_below_the_headroom(model_dir):
    models = write_models(model_dir, make_spec("a"))
    engine = make_engine(model_dir, models, free_vram=lambda device: 2047)
    engine.begin_phase("dml:0")
    engine.session("a", "dml:0", "fp32")

    assert engine.release_before_ncnn("dml:0") is True
    assert live_models(engine) == []


def test_ncnn_releases_sessions_when_free_vram_cannot_be_measured(model_dir):
    models = write_models(model_dir, make_spec("a"))
    engine = make_engine(model_dir, models, free_vram=lambda device: None)
    engine.begin_phase("dml:0")
    engine.session("a", "dml:0", "fp32")

    assert engine.release_before_ncnn("dml:0") is True
    assert live_models(engine) == []


def test_ncnn_keeps_sessions_with_enough_free_vram(model_dir):
    models = write_models(model_dir, make_spec("a"))
    asked: list[str] = []

    def free_vram(device: str) -> int:
        asked.append(device)
        return 2048

    engine = make_engine(model_dir, models, free_vram=free_vram)
    engine.begin_phase("dml:0")
    engine.session("a", "dml:0", "fp32")

    assert engine.release_before_ncnn("dml:0") is False
    assert live_models(engine) == ["a"]
    assert asked == ["dml:0"]


def test_ncnn_headroom_follows_the_setting(model_dir):
    models = write_models(model_dir, make_spec("a"))
    engine = make_engine(model_dir, models, free_vram=lambda device: 3000, RESTORE_NCNN_HEADROOM_MB=4096)
    engine.begin_phase("dml:0")
    engine.session("a", "dml:0", "fp32")

    assert engine.release_before_ncnn("dml:0") is True


def test_ncnn_leaves_cpu_sessions_alone(model_dir):
    models = write_models(model_dir, make_spec("a"))
    engine = make_engine(model_dir, models, free_vram=lambda device: None)
    engine.begin_phase("cpu")
    engine.session("a", "cpu", "fp32")

    assert engine.release_before_ncnn("cpu") is False
    assert live_models(engine, "cpu") == ["a"]


def test_ncnn_release_ends_the_phase(model_dir):
    models = write_models(model_dir, make_spec("a"))
    engine = make_engine(model_dir, models, free_vram=lambda device: 0)
    engine.begin_phase("dml:0")
    engine.release_before_ncnn("dml:0")

    with pytest.raises(RuntimeError, match="begin_phase"):
        engine.session("a", "dml:0", "fp32")


def test_session_estimated_vram_is_file_size_times_vram_factor(tmp_path):
    model_file = tmp_path / "m.onnx"
    model_file.write_bytes(b"\0" * (2 * 1024 * 1024))

    assert estimated_vram_mb(model_file, 3.0) == pytest.approx(6.0)


def test_session_options_on_dml_disable_mem_pattern_and_use_one_thread():
    options = SimpleNamespace(enable_mem_pattern=True, intra_op_num_threads=0, graph_optimization_level="default")

    configure_session_options(options, make_spec("a"), "dml:0", disable_all_level="off")

    assert options.enable_mem_pattern is False
    assert options.intra_op_num_threads == 1
    assert options.graph_optimization_level == "default"


def test_session_options_on_cpu_keep_ort_defaults():
    options = SimpleNamespace(enable_mem_pattern=True, intra_op_num_threads=0, graph_optimization_level="default")

    configure_session_options(options, make_spec("a"), "cpu", disable_all_level="off")

    assert options == SimpleNamespace(
        enable_mem_pattern=True, intra_op_num_threads=0, graph_optimization_level="default"
    )


@pytest.mark.parametrize("device", ["cpu", "dml:0"])
def test_session_options_disable_all_optimizations_only_when_the_spec_asks(device):
    options = SimpleNamespace(enable_mem_pattern=True, intra_op_num_threads=0, graph_optimization_level="default")

    configure_session_options(options, make_spec("a", disable_all=True), device, disable_all_level="off")

    assert options.graph_optimization_level == "off"


def test_session_options_factory_builds_real_ort_options(model_dir):
    ort = pytest.importorskip("onnxruntime")
    models = write_models(model_dir, make_spec("a", disable_all=True))
    factory = SessionFactory()
    engine = make_engine(model_dir, models, factory=factory)
    engine.begin_phase("dml:0")
    engine.session("a", "dml:0", "fp32")

    options = factory.calls[0]["sess_options_factory"]()

    assert isinstance(options, ort.SessionOptions)
    assert options.enable_mem_pattern is False
    assert options.intra_op_num_threads == 1
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL


def test_session_options_factory_is_fresh_per_attempt(model_dir):
    pytest.importorskip("onnxruntime")
    models = write_models(model_dir, make_spec("a"))
    factory = SessionFactory()
    engine = make_engine(model_dir, models, factory=factory)
    engine.begin_phase("dml:0")
    engine.session("a", "dml:0", "fp32")
    build = factory.calls[0]["sess_options_factory"]

    assert build() is not build()


def test_ncnn_free_vram_probe_reads_the_dml_adapter(monkeypatch):
    asked: list[int] = []

    def fake_adapter_free_vram_mb(index: int) -> int:
        asked.append(index)
        return 1234

    monkeypatch.setattr(photo_restore_engine, "adapter_free_vram_mb", fake_adapter_free_vram_mb)

    assert dml_free_vram_mb("dml:1") == 1234
    assert dml_free_vram_mb("cpu") is None
    assert asked == [1]


class OrtLikeSession:
    def __init__(self, transform) -> None:
        self.transform = transform
        self.runs = 0

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, output_names, feeds):
        self.runs += 1
        return [self.transform(feeds["input"])]


def identity(batch):
    return batch.copy()


def brown(batch):
    # El modo de falla medido de GFPGAN fp16 ingenuo: una imagen marron finita.
    values = np.array([0.45, 0.3, 0.2], dtype=np.float32).reshape(1, 3, 1, 1)
    return np.broadcast_to(values, batch.shape).copy()


def with_nan(batch):
    out = batch.copy()
    out[0, 0, 0, 0] = np.nan
    return out


def with_inf(batch):
    out = batch.copy()
    out[0, 1, 0, 0] = np.inf
    return out


def out_of_range(batch):
    return batch * 2.0 - 0.5


def offset_by(amount: float):
    return lambda batch: batch + np.float32(amount)


def raising(message: str):
    def transform(batch):
        raise RuntimeError(message)

    return transform


DEVICE_REMOVED_TEXT = "Non-zero status code returned while running Conv node. 887A0005 DXGI_ERROR_DEVICE_REMOVED"


class OrtSessionFactory:
    def __init__(self, transforms: dict[str, object]) -> None:
        self.transforms = transforms
        self.calls: list[tuple[str, str]] = []
        self.sessions: list[OrtLikeSession] = []

    def __call__(self, model_path: str, device: str, settings: Settings, **kwargs) -> OrtLikeSession:
        name = Path(model_path).name
        self.calls.append((name, device))
        transform = self.transforms.get(name, identity)
        if isinstance(transform, Exception):
            raise transform
        session = OrtLikeSession(transform)
        self.sessions.append(session)
        return session

    def runs_of(self, filename: str) -> int:
        created = [call for call in self.calls if not isinstance(self.transforms.get(call[0]), Exception)]
        return sum(session.runs for (name, _), session in zip(created, self.sessions) if name == filename)


class RecordingOwner:
    def __init__(self) -> None:
        self.released: list[str] = []

    def release_device(self, device: str) -> None:
        self.released.append(device)


def sample_tile(side: int = 16) -> np.ndarray:
    ramp = np.linspace(0.1, 0.9, side * side, dtype=np.float32).reshape(side, side)
    return np.stack([ramp, ramp[::-1], ramp.T], axis=2)


def health_engine(model_dir, models, factory, *, coordinator=None, health=None, **overrides):
    return PhotoRestoreEngine(
        make_settings(model_dir, **overrides),
        coordinator or CountingCoordinator(),
        models=models,
        create_session=factory,
        free_vram_mb=lambda device: None,
        device_health=health or DevicesService(make_settings(model_dir)),
    )


def fp16_engine(model_dir, factory, **overrides):
    models = write_models(model_dir, make_spec("drunet", fp16=True))
    return health_engine(model_dir, models, factory, **overrides)


def test_canary_brown_fp16_marks_fp16_bad_and_reruns_in_fp32(model_dir):
    factory = OrtSessionFactory({"drunet-fp16.onnx": brown})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")

    ready = engine.ready_infer("drunet", "dml:0", sample_tile())
    output = ready.infer(sample_tile())

    assert ready.precision == "fp32"
    assert np.array_equal(output, sample_tile())
    assert [rejection.model_id for rejection in engine.fp16_rejections()] == ["drunet"]
    assert engine.fp16_rejections()[0].device == "dml:0"
    assert "dB" in engine.fp16_rejections()[0].reason


def test_canary_rejected_fp16_session_leaves_the_cache(model_dir):
    factory = OrtSessionFactory({"drunet-fp16.onnx": brown})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")

    engine.ready_infer("drunet", "dml:0", sample_tile())

    assert engine.live_sessions("dml:0") == (SessionKey("drunet", "dml:0", "fp32"),)


def test_canary_catches_a_finite_wrong_output_a_nan_guard_alone_would_miss(model_dir):
    factory = OrtSessionFactory({"drunet-fp16.onnx": brown})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")

    fp16_output = engine.tile_infer("drunet", "dml:0", "fp16")(sample_tile())

    assert np.isfinite(fp16_output).all()
    assert engine.precision_for("drunet", "dml:0", sample_tile()) == "fp32"


def test_canary_good_fp16_keeps_fp16(model_dir):
    factory = OrtSessionFactory({})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")

    assert engine.precision_for("drunet", "dml:0", sample_tile()) == "fp16"
    assert engine.fp16_rejections() == ()


def test_canary_runs_once_per_model_and_device(model_dir):
    factory = OrtSessionFactory({})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")
    engine.begin_phase("dml:1")

    engine.precision_for("drunet", "dml:0", sample_tile())
    engine.precision_for("drunet", "dml:0", sample_tile())
    engine.precision_for("drunet", "dml:1", sample_tile())

    assert factory.runs_of("drunet-fp16.onnx") == 2
    assert factory.runs_of("drunet.onnx") == 2


def test_canary_bad_verdict_lasts_for_the_process(model_dir):
    factory = OrtSessionFactory({"drunet-fp16.onnx": brown})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")
    engine.precision_for("drunet", "dml:0", sample_tile())

    assert engine.precision_for("drunet", "dml:0", sample_tile()) == "fp32"
    assert factory.runs_of("drunet-fp16.onnx") == 1
    assert len(engine.fp16_rejections()) == 1


def test_canary_is_skipped_on_cpu(model_dir):
    factory = OrtSessionFactory({"drunet-fp16.onnx": brown})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("cpu")

    assert engine.precision_for("drunet", "cpu", sample_tile()) == "fp32"
    assert factory.calls == []


def test_canary_is_skipped_when_fp16_is_not_preferred(model_dir):
    factory = OrtSessionFactory({})
    engine = fp16_engine(model_dir, factory, ONNX_PREFER_FP16=False)
    engine.begin_phase("dml:0")

    assert engine.precision_for("drunet", "dml:0", sample_tile()) == "fp32"
    assert factory.calls == []


def test_canary_is_skipped_for_a_model_without_fp16_file(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    factory = OrtSessionFactory({})
    engine = health_engine(model_dir, models, factory)
    engine.begin_phase("dml:0")

    assert engine.precision_for("drunet", "dml:0", sample_tile()) == "fp32"
    assert factory.calls == []


def test_canary_reference_on_cpu_uses_a_transient_session(model_dir):
    factory = OrtSessionFactory({"drunet-fp16.onnx": brown})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")

    precision = engine.precision_for("drunet", "dml:0", sample_tile(), reference_device="cpu")

    assert precision == "fp32"
    assert factory.calls == [("drunet-fp16.onnx", "dml:0"), ("drunet.onnx", "cpu")]
    assert engine.live_sessions("cpu") == ()


def test_canary_uses_the_faces_threshold_for_face_models(model_dir):
    # Un corrimiento de 0,0045 da ~46,9 dB: pasa el umbral de caras (45) y no el de restauracion (50).
    factory = OrtSessionFactory({"gfpgan-v1.4-fp16.onnx": offset_by(0.0045), "drunet-fp16.onnx": offset_by(0.0045)})
    models = write_models(model_dir, make_spec("gfpgan-v1.4", fp16=True), make_spec("drunet", fp16=True))
    engine = health_engine(model_dir, models, factory)
    engine.begin_phase("dml:0")

    assert engine.precision_for("gfpgan-v1.4", "dml:0", sample_tile()) == "fp16"
    assert engine.precision_for("drunet", "dml:0", sample_tile()) == "fp32"


def test_canary_feeds_the_sample_through_the_caller_input_contract(model_dir):
    factory = OrtSessionFactory({})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")
    fed: list[tuple[int, ...]] = []

    def infer_for(session):
        def infer(tile):
            fed.append(tile.shape)
            return tile[:, :, :3]

        return infer

    engine.precision_for("drunet", "dml:0", np.zeros((8, 8, 4), np.float32), infer_for=infer_for)

    assert fed == [(8, 8, 4), (8, 8, 4)]


def test_nan_in_the_fp16_canary_marks_fp16_bad_and_reruns_in_fp32(model_dir):
    factory = OrtSessionFactory({"drunet-fp16.onnx": with_nan})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")

    ready = engine.ready_infer("drunet", "dml:0", sample_tile())

    assert ready.precision == "fp32"
    assert np.array_equal(ready.infer(sample_tile()), sample_tile())
    assert "NaN" in engine.fp16_rejections()[0].reason


def test_nan_guard_refuses_non_finite_output(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    engine = health_engine(model_dir, models, OrtSessionFactory({"drunet.onnx": with_nan}))
    engine.begin_phase("dml:0")

    with pytest.raises(NonFiniteOutputError, match="drunet"):
        engine.tile_infer("drunet", "dml:0", "fp32")(sample_tile())


def test_nan_guard_runs_before_the_clamp_so_inf_never_turns_white(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    engine = health_engine(model_dir, models, OrtSessionFactory({"drunet.onnx": with_inf}))
    engine.begin_phase("dml:0")

    with pytest.raises(NonFiniteOutputError):
        engine.tile_infer("drunet", "dml:0", "fp32")(sample_tile())


def test_nan_in_fp16_after_a_passed_canary_marks_fp16_bad(model_dir):
    factory = OrtSessionFactory({})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")
    assert engine.precision_for("drunet", "dml:0", sample_tile()) == "fp16"
    factory.sessions[0].transform = with_nan

    with pytest.raises(NonFiniteOutputError):
        engine.tile_infer("drunet", "dml:0", "fp16")(sample_tile())
    assert engine.precision_for("drunet", "dml:0", sample_tile()) == "fp32"
    assert "NaN" in engine.fp16_rejections()[0].reason


def test_nan_guard_then_clamp_on_the_host(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    engine = health_engine(model_dir, models, OrtSessionFactory({"drunet.onnx": out_of_range}))
    engine.begin_phase("dml:0")

    output = engine.tile_infer("drunet", "dml:0", "fp32")(sample_tile())

    assert output.dtype == np.float32
    assert (output.min(), output.max()) == (0.0, 1.0)
    assert np.array_equal(output, np.clip(sample_tile() * 2.0 - 0.5, 0.0, 1.0))


def test_nan_guard_keeps_running_without_clamp_for_outputs_outside_unit_range(model_dir):
    models = write_models(model_dir, make_spec("ddcolor"))
    engine = health_engine(model_dir, models, OrtSessionFactory({"ddcolor.onnx": out_of_range}))
    engine.begin_phase("dml:0")

    output = engine.tile_infer("ddcolor", "dml:0", "fp32", clamp=False)(sample_tile())

    assert output.min() < 0.0


def test_nan_guard_and_clamp_helpers_are_pure():
    raw = np.array([[-0.5, 0.5, 1.5]], dtype=np.float32)

    assert np.array_equal(clamp_unit(raw), np.array([[0.0, 0.5, 1.0]], dtype=np.float32))
    assert np.array_equal(raw, np.array([[-0.5, 0.5, 1.5]], dtype=np.float32))
    assert guard_finite(raw, "m", "fp32") is raw
    with pytest.raises(NonFiniteOutputError, match="m / fp16"):
        guard_finite(np.array([np.nan], dtype=np.float32), "m", "fp16")


def test_removed_device_fails_the_job_with_device_removed(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    engine = health_engine(model_dir, models, OrtSessionFactory({"drunet.onnx": raising(DEVICE_REMOVED_TEXT)}))
    engine.begin_phase("dml:0")

    with pytest.raises(DeviceRemovedError) as caught:
        engine.tile_infer("drunet", "dml:0", "fp32")(sample_tile())

    assert caught.value.code == "restore.error.deviceRemoved"
    assert str(caught.value) == "The GPU driver reset. Restart Upflow to use the GPU again."
    assert caught.value.device == "dml:0"


def test_removed_device_invalidation_reaches_every_owner(model_dir):
    models = write_models(model_dir, make_spec("drunet"), make_spec("other"))
    coordinator = GpuSessionCoordinator()
    other_owner = RecordingOwner()
    coordinator.register(other_owner)
    factory = OrtSessionFactory({"drunet.onnx": raising(DEVICE_REMOVED_TEXT)})
    engine = health_engine(model_dir, models, factory, coordinator=coordinator)
    engine.begin_phase("dml:0")
    engine.session("other", "dml:0", "fp32")

    with pytest.raises(DeviceRemovedError):
        engine.tile_infer("drunet", "dml:0", "fp32")(sample_tile())

    assert other_owner.released == ["dml:0"]
    assert live_models(engine) == []


def test_removed_device_is_marked_unhealthy_and_the_next_jobs_are_rejected(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    coordinator = GpuSessionCoordinator()
    health = DevicesService(make_settings(model_dir))
    factory = OrtSessionFactory({"drunet.onnx": raising(DEVICE_REMOVED_TEXT)})
    engine = health_engine(model_dir, models, factory, coordinator=coordinator, health=health)
    engine.begin_phase("dml:0")
    with pytest.raises(DeviceRemovedError):
        engine.tile_infer("drunet", "dml:0", "fp32")(sample_tile())
    other_owner = RecordingOwner()
    coordinator.acquire("dml:0", other_owner)

    with pytest.raises(DeviceRemovedError, match="Restart Upflow"):
        engine.begin_phase("dml:0")

    assert health.is_healthy("dml:0") is False
    assert other_owner.released == []


def test_removed_device_is_not_retried(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    coordinator = CountingCoordinator()
    factory = OrtSessionFactory({"drunet.onnx": raising(DEVICE_REMOVED_TEXT)})
    engine = health_engine(model_dir, models, factory, coordinator=coordinator)
    engine.begin_phase("dml:0")

    with pytest.raises(DeviceRemovedError):
        engine.tile_infer("drunet", "dml:0", "fp32")(sample_tile())

    assert factory.calls == [("drunet.onnx", "dml:0")]
    assert factory.sessions[0].runs == 1
    assert coordinator.invalidated == ["dml:0"]


def test_removed_device_while_creating_the_session_is_classified_too(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    factory = OrtSessionFactory({"drunet.onnx": RuntimeError("D3D12 887A0006 DXGI_ERROR_DEVICE_HUNG")})
    engine = health_engine(model_dir, models, factory)
    engine.begin_phase("dml:0")

    with pytest.raises(DeviceRemovedError):
        engine.tile_infer("drunet", "dml:0", "fp32")(sample_tile())


def test_removed_device_during_the_canary_is_classified_too(model_dir):
    factory = OrtSessionFactory({"drunet-fp16.onnx": raising(DEVICE_REMOVED_TEXT)})
    engine = fp16_engine(model_dir, factory)
    engine.begin_phase("dml:0")

    with pytest.raises(DeviceRemovedError):
        engine.precision_for("drunet", "dml:0", sample_tile())
    assert engine.fp16_rejections() == ()


def test_removed_classification_leaves_oom_alone(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    factory = OrtSessionFactory({"drunet.onnx": raising("Failed to allocate memory: out of memory")})
    health = DevicesService(make_settings(model_dir))
    engine = health_engine(model_dir, models, factory, health=health)
    engine.begin_phase("dml:0")

    with pytest.raises(RuntimeError, match="out of memory") as caught:
        engine.tile_infer("drunet", "dml:0", "fp32")(sample_tile())

    assert not isinstance(caught.value, DeviceRemovedError)
    assert health.is_healthy("dml:0") is True
    assert live_models(engine) == ["drunet"]


def test_removed_device_leaves_other_devices_working(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    coordinator = GpuSessionCoordinator()
    engine = health_engine(model_dir, models, OrtSessionFactory({}), coordinator=coordinator)
    engine.begin_phase("dml:0")
    engine.begin_phase("dml:1")
    engine.session("drunet", "dml:1", "fp32")
    engine.session("drunet", "dml:0", "fp32").transform = raising(DEVICE_REMOVED_TEXT)

    with pytest.raises(DeviceRemovedError):
        engine.tile_infer("drunet", "dml:0", "fp32")(sample_tile())

    assert live_models(engine, "dml:1") == ["drunet"]
    engine.begin_phase("dml:1")


def test_removed_text_on_cpu_is_not_a_device_removal(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    factory = OrtSessionFactory({"drunet.onnx": raising(DEVICE_REMOVED_TEXT)})
    engine = health_engine(model_dir, models, factory)
    engine.begin_phase("cpu")

    with pytest.raises(RuntimeError) as caught:
        engine.tile_infer("drunet", "cpu", "fp32")(sample_tile())

    assert not isinstance(caught.value, DeviceRemovedError)
    engine.begin_phase("cpu")


async def test_cancel_releases_the_device_permit_only_after_the_worker_ends():
    permit = asyncio.Semaphore(1)
    started = threading.Event()
    finished = threading.Event()

    def blocking(cancel_event: threading.Event) -> None:
        started.set()
        while not cancel_event.wait(0.01):
            pass
        time.sleep(0.05)
        finished.set()

    async def job() -> None:
        async with permit:
            await run_cancellable(blocking)

    task = asyncio.create_task(job())
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()

    await asyncio.wait_for(permit.acquire(), timeout=5)

    assert finished.is_set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancel_signals_the_worker_through_the_cancel_event():
    seen: list[bool] = []
    started = threading.Event()

    def blocking(cancel_event: threading.Event) -> None:
        started.set()
        seen.append(cancel_event.wait(5))

    task = asyncio.create_task(run_cancellable(blocking))
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert seen == [True]


async def test_cancel_free_run_returns_the_worker_result():
    def blocking(a: int, b: int, cancel_event: threading.Event) -> int:
        assert not cancel_event.is_set()
        return a + b

    assert await run_cancellable(blocking, 2, 3) == 5


async def test_cancel_free_run_propagates_the_worker_error():
    def blocking(cancel_event: threading.Event) -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await run_cancellable(blocking)
