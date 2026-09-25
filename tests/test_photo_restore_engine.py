from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.services.engines import photo_restore_engine
from app.services.engines.photo_restore_engine import (
    PhotoRestoreEngine,
    SessionKey,
    configure_session_options,
    dml_free_vram_mb,
    estimated_vram_mb,
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

    def register(self, owner: object) -> None:
        self.registered.append(owner)

    def acquire(self, device: str, owner: object) -> None:
        self.acquired.append((device, owner))


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


def test_phase_ends_when_another_owner_takes_the_device(model_dir):
    models = write_models(model_dir, make_spec("drunet"))
    coordinator = GpuSessionCoordinator()
    engine = make_engine(model_dir, models, coordinator=coordinator)
    engine.begin_phase("dml:0")
    engine.session("drunet", "dml:0", "fp32")

    coordinator.acquire("dml:0", OtherOwner())

    assert live_models(engine) == []
    with pytest.raises(RuntimeError, match="begin_phase"):
        engine.session("drunet", "dml:0", "fp32")


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
