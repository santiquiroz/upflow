from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services.devices_service import DevicesService
from app.services.engines.onnx_upscaler import (
    SESSION_CACHE_SIZE,
    TILE_OVERLAP_PX,
    OnnxUpscaler,
    _build_providers,
    _detect_scale,
    _tile_starts,
)
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus

# ---------------------------------------------------------------------------
# SP1 Task 4 - engines/onnx_upscaler: in-process ONNX Runtime DirectML
# upscaling engine.
#
# No real onnxruntime session is exercised for the tiling/blending/cache
# tests below -- `OnnxUpscaler._create_session` is a monkeypatchable seam
# that unit tests replace with `Double2xSession`, a numpy fake that just
# duplicates H/W (mirrors onnxruntime's InferenceSession API surface:
# get_inputs()/get_outputs()/run()). `UpscaleJob` does not carry model_id/
# device yet (that wiring lands in Task 7) so tests use a minimal `StubJob`
# with the shape the engine actually reads.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


@dataclass
class StubJob:
    source_path: Path
    original_filename: str
    output_format: str
    model_id: str
    device: str
    id: str = "job-1"
    metadata: dict = field(default_factory=dict)


class _IoInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class Double2xSession:
    """Fake ONNX session: doubles H/W per-pixel (no receptive-field context)."""

    def __init__(self) -> None:
        self._input = _IoInfo("input")
        self._output = _IoInfo("output")

    def get_inputs(self) -> list[_IoInfo]:
        return [self._input]

    def get_outputs(self) -> list[_IoInfo]:
        return [self._output]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        array = input_feed[self._input.name]
        doubled = np.repeat(np.repeat(array, 2, axis=2), 2, axis=3)
        return [doubled]


class FailingSession:
    def __init__(self, message: str) -> None:
        self._message = message
        self._input = _IoInfo("input")
        self._output = _IoInfo("output")

    def get_inputs(self) -> list[_IoInfo]:
        return [self._input]

    def get_outputs(self) -> list[_IoInfo]:
        return [self._output]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        raise RuntimeError(self._message)


class PerTileConstantSession:
    """Position-SENSITIVE fake: fills each successive tile (in the engine's
    row-major processing order) with a distinct constant, ignoring pixel
    content. Unlike Double2xSession, the same source pixel yields a DIFFERENT
    output value depending on which tile inferred it -- so the value written
    into the overlap region is fully determined by the blend weights, which
    makes the feather taper observable (and a broken/flat feather detectable).
    """

    def __init__(self, fill_values: list[float], scale: int = 2) -> None:
        self._fill_values = fill_values
        self._scale = scale
        self._call = 0
        self._input = _IoInfo("input")
        self._output = _IoInfo("output")

    def get_inputs(self) -> list[_IoInfo]:
        return [self._input]

    def get_outputs(self) -> list[_IoInfo]:
        return [self._output]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        array = input_feed[self._input.name]
        _, channels, height, width = array.shape
        value = self._fill_values[self._call]
        self._call += 1
        shape = (1, channels, height * self._scale, width * self._scale)
        return [np.full(shape, value, dtype=np.float32)]


def make_onnx_entry(**overrides: object) -> ModelEntry:
    defaults: dict[str, object] = {
        "id": "fake-2x",
        "name": "Fake 2x",
        "kind": ModelKind.onnx,
        "source": "https://huggingface.co/example/fake-2x",
        "size_bytes": 1_234,
        "scale": 2,
        "arch": "fake",
        "file_path": "onnx/fake-2x.onnx",
        "status": ModelStatus.installed,
    }
    defaults.update(overrides)
    return ModelEntry(**defaults)


def make_engine(tmp_path: Path, **settings_overrides: object) -> tuple[OnnxUpscaler, ModelRegistry, Settings]:
    settings = make_settings(tmp_path, **settings_overrides)
    registry = ModelRegistry(settings)
    devices = DevicesService(settings)
    return OnnxUpscaler(settings, registry, devices, GpuSessionCoordinator()), registry, settings


def make_gradient_array(height: int, width: int) -> np.ndarray:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    array[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    array[..., 2] = 128
    return array


def write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def make_job(
    tmp_path: Path,
    *,
    model_id: str,
    device: str,
    job_id: str = "job-1",
    width: int = 10,
    height: int = 6,
) -> StubJob:
    source_path = tmp_path / f"{job_id}-source.png"
    write_image(source_path, make_gradient_array(height, width))
    return StubJob(
        source_path=source_path,
        original_filename="source.png",
        output_format="png",
        model_id=model_id,
        device=device,
        id=job_id,
    )


# ---------------------------------------------------------------------------
# available()
# ---------------------------------------------------------------------------


def test_available_true_when_onnxruntime_installed(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    engine, _, _ = make_engine(tmp_path)
    assert engine.available() is True


def test_available_false_when_onnxruntime_import_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _, _ = make_engine(tmp_path)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert engine.available() is False


# ---------------------------------------------------------------------------
# device -> provider mapping
# ---------------------------------------------------------------------------


def test_build_providers_for_cpu() -> None:
    assert _build_providers("cpu") == ["CPUExecutionProvider"]


def test_build_providers_for_dml_zero() -> None:
    assert _build_providers("dml:0") == [
        ("DmlExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]


def test_build_providers_for_dml_one() -> None:
    assert _build_providers("dml:1") == [
        ("DmlExecutionProvider", {"device_id": 1}),
        "CPUExecutionProvider",
    ]


def test_build_providers_rejects_unknown_device() -> None:
    with pytest.raises(RuntimeError, match="Unsupported device"):
        _build_providers("npu:0")


# ---------------------------------------------------------------------------
# _tile_starts
# ---------------------------------------------------------------------------


def test_tile_starts_single_tile_when_image_smaller_than_tile() -> None:
    assert _tile_starts(100, 256, 16) == [0]


def test_tile_starts_evenly_divisible_with_overlap() -> None:
    assert _tile_starts(48, 32, 16) == [0, 16]


def test_tile_starts_last_tile_flush_with_edge() -> None:
    starts = _tile_starts(100, 32, 16)
    assert starts[-1] == 100 - 32
    assert starts[0] == 0


# ---------------------------------------------------------------------------
# _detect_scale
# ---------------------------------------------------------------------------


def test_detect_scale_doubles() -> None:
    output = np.zeros((64, 64, 3), dtype=np.float32)
    assert _detect_scale(32, 32, output) == 2


def test_detect_scale_raises_on_non_integer_ratio() -> None:
    output = np.zeros((50, 50, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="integer upscale ratio"):
        _detect_scale(32, 32, output)


def test_detect_scale_raises_on_non_uniform_scale() -> None:
    output = np.zeros((64, 96, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="non-uniform scale"):
        _detect_scale(32, 32, output)


# ---------------------------------------------------------------------------
# Tiling + blending (algorithm-level, no file I/O)
# ---------------------------------------------------------------------------


def test_non_tiled_upscale_doubles_dimensions(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    session = Double2xSession()
    array = make_gradient_array(height=6, width=10)

    result = engine._upscale_array(session, array, tile_size=256)

    assert result.shape == (12, 20, 3)
    assert result.dtype == np.uint8


def test_tiled_upscale_reconstructs_gradient_exactly(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    session = Double2xSession()
    array = make_gradient_array(height=48, width=48)

    whole_image_result = engine._upscale_array(session, array, tile_size=0)
    tiled_result = engine._upscale_array(session, array, tile_size=32)

    assert tiled_result.shape == whole_image_result.shape == (96, 96, 3)
    assert np.array_equal(tiled_result, whole_image_result)


def test_tiled_upscale_handles_non_square_non_divisible_image(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    session = Double2xSession()
    array = make_gradient_array(height=70, width=50)

    whole_image_result = engine._upscale_array(session, array, tile_size=0)
    tiled_result = engine._upscale_array(session, array, tile_size=32)

    assert tiled_result.shape == whole_image_result.shape == (140, 100, 3)
    assert np.array_equal(tiled_result, whole_image_result)


def test_tile_overlap_constant_is_16px() -> None:
    assert TILE_OVERLAP_PX == 16


# ---------------------------------------------------------------------------
# Tile progress reporting (SP5 Task 4)
# ---------------------------------------------------------------------------


class RecordingProgressSession:
    """Fake session that snapshots the job's metadata BEFORE each tile is
    inferred, so the test can assert progress was updated incrementally
    between tiles rather than only once at the very end."""

    def __init__(self, job: StubJob, scale: int = 2) -> None:
        self._job = job
        self._scale = scale
        self.snapshots: list[dict] = []
        self._input = _IoInfo("input")
        self._output = _IoInfo("output")

    def get_inputs(self) -> list[_IoInfo]:
        return [self._input]

    def get_outputs(self) -> list[_IoInfo]:
        return [self._output]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.snapshots.append(dict(self._job.metadata))
        array = input_feed[self._input.name]
        doubled = np.repeat(np.repeat(array, self._scale, axis=2), self._scale, axis=3)
        return [doubled]


def test_upscale_tiled_reports_incremental_tile_progress(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    job = make_job(tmp_path, model_id="fake-2x", device="cpu")
    session = RecordingProgressSession(job)
    array = make_gradient_array(height=48, width=48)  # tile=32, overlap=16 -> 2x2 = 4 tiles

    engine._upscale_tiled(session, array, tile_size=32, job=job)

    frames_done_before_each_tile = [snapshot.get("framesDone", 0) for snapshot in session.snapshots]
    assert frames_done_before_each_tile == [0, 1, 2, 3]
    assert job.metadata["framesDone"] == 4
    assert job.metadata["framesTotal"] == 4
    assert job.metadata["progress"] == pytest.approx(1.0)
    assert job.metadata["stage"] == "upscaling"


def test_upscale_array_tiled_updates_progress_via_job(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    job = make_job(tmp_path, model_id="fake-2x", device="cpu")
    session = Double2xSession()
    array = make_gradient_array(height=48, width=48)

    engine._upscale_array(session, array, tile_size=32, job=job)

    assert job.metadata["framesTotal"] == 4
    assert job.metadata["framesDone"] == 4


def test_upscale_array_non_tiled_leaves_job_metadata_untouched(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    job = make_job(tmp_path, model_id="fake-2x", device="cpu")
    session = Double2xSession()
    array = make_gradient_array(height=6, width=10)

    engine._upscale_array(session, array, tile_size=256, job=job)

    # Single-pass (non-tiled): no honest tile count exists, so no fake
    # tilesTotal=1 sub-progress is reported -- stays coarse.
    assert "framesDone" not in job.metadata
    assert "framesTotal" not in job.metadata


def test_upscale_array_job_none_is_backward_compatible(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    session = Double2xSession()
    array = make_gradient_array(height=48, width=48)

    result = engine._upscale_array(session, array, tile_size=32)

    assert result.shape == (96, 96, 3)


def test_tiling_feather_produces_gradient_taper_in_overlap(tmp_path: Path) -> None:
    # Two side-by-side tiles filled with distinct constants (left=0.2->51,
    # right=0.8->204). A position-invariant fake would reconstruct exactly
    # regardless of weights; here the overlap value depends ENTIRELY on the
    # feather ramp, so we can pin the taper shape.
    #
    # width=48, tile=32, overlap=16 -> tile starts [0, 16] (2 tiles);
    # height=16 <= 32 -> a single tile row. Overlap is output cols [32, 64).
    #   - hard seam  -> [51,...,51, 204,...,204]  (only 2 distinct values)
    #   - flat average (feather OFF, weights all 1) -> constant 128
    #   - feather ON -> monotonic gradient from ~51 toward ~204
    engine, _, _ = make_engine(tmp_path)
    session = PerTileConstantSession(fill_values=[0.2, 0.8], scale=2)
    array = make_gradient_array(height=16, width=48)

    result = engine._upscale_array(session, array, tile_size=32)

    assert result.shape == (32, 96, 3)
    overlap_profile = result[8, 32:64, 0].astype(int)

    # Not a flat average: a genuine gradient has many distinct levels.
    assert np.unique(overlap_profile).size >= 8
    # Monotonic left->right taper (a broken/reversed ramp breaks this).
    assert np.all(np.diff(overlap_profile) >= 0)
    assert overlap_profile[0] < overlap_profile[-1]
    # Not a hard seam: intermediate blended values exist between the two fills.
    assert np.any((overlap_profile > 70) & (overlap_profile < 185))
    # Endpoints are blended, not the raw per-tile constants.
    assert overlap_profile[0] > 51
    assert overlap_profile[-1] < 204


# ---------------------------------------------------------------------------
# Session caching LRU(2)
# ---------------------------------------------------------------------------


def test_get_session_caches_by_model_and_device(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_create_session(model_id: str, device: str, entry: ModelEntry) -> object:
        calls.append((model_id, device))
        return object()

    engine._create_session = fake_create_session  # type: ignore[method-assign]
    entry = make_onnx_entry()

    first = engine._get_session("m1", "cpu", entry)
    second = engine._get_session("m1", "cpu", entry)

    assert first is second
    assert calls == [("m1", "cpu")]


def test_get_session_evicts_least_recently_used_beyond_size_2(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_create_session(model_id: str, device: str, entry: ModelEntry) -> object:
        calls.append((model_id, device))
        return object()

    engine._create_session = fake_create_session  # type: ignore[method-assign]
    entry = make_onnx_entry()

    engine._get_session("m1", "cpu", entry)
    engine._get_session("m2", "cpu", entry)
    engine._get_session("m1", "cpu", entry)  # re-touch m1, m2 becomes LRU
    engine._get_session("m3", "cpu", entry)  # evicts m2

    assert len(engine._session_cache) == SESSION_CACHE_SIZE
    cached_keys = set(engine._session_cache.keys())
    assert cached_keys == {("m1", "cpu"), ("m3", "cpu")}
    assert calls == [("m1", "cpu"), ("m2", "cpu"), ("m3", "cpu")]


def test_get_session_treats_different_devices_as_different_cache_keys(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    engine._create_session = lambda model_id, device, entry: object()  # type: ignore[method-assign]
    entry = make_onnx_entry()

    engine._get_session("m1", "cpu", entry)
    engine._get_session("m1", "dml:0", entry)

    assert set(engine._session_cache.keys()) == {("m1", "cpu"), ("m1", "dml:0")}


# ---------------------------------------------------------------------------
# GpuSessionCoordinator wiring (Fase 1 Task 4) - release_device evicts every
# cache entry for that device regardless of model (cache is keyed by
# (model_id, device), a single device can have several model entries, unlike
# the flat per-device caches in Tasks 2-3), and acquire() runs before any
# session is built. Same pattern as GmfssEngine/AudioSrRestorer/ApolloRestorer.
# ---------------------------------------------------------------------------


def test_release_device_clears_all_cached_sessions_for_that_device(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)
    engine._session_cache[("model-a", "dml:0")] = "fake-a"
    engine._session_cache[("model-b", "dml:0")] = "fake-b"
    engine._session_cache[("model-a", "dml:1")] = "fake-a-1"

    engine.release_device("dml:0")

    assert ("model-a", "dml:0") not in engine._session_cache
    assert ("model-b", "dml:0") not in engine._session_cache
    assert ("model-a", "dml:1") in engine._session_cache


def test_release_device_on_empty_cache_is_a_noop(tmp_path: Path) -> None:
    engine, _, _ = make_engine(tmp_path)

    engine.release_device("dml:0")  # no debe lanzar


def test_get_session_calls_coordinator_acquire_before_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    devices = DevicesService(settings)
    gpu_coordinator = GpuSessionCoordinator()
    engine = OnnxUpscaler(settings, registry, devices, gpu_coordinator)
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(gpu_coordinator, "acquire", lambda device, owner: calls.append((device, owner)))
    monkeypatch.setattr(engine, "_create_session", lambda model_id, device, entry: "fake-session")
    entry = make_onnx_entry()

    engine._get_session("model-a", "dml:0", entry)

    assert calls == [("dml:0", engine)]


# ---------------------------------------------------------------------------
# run() end-to-end (fake session, real registry/devices/file I/O)
# ---------------------------------------------------------------------------


async def test_run_upscales_and_writes_non_empty_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, registry, settings = make_engine(tmp_path)
    registry.register(make_onnx_entry())
    monkeypatch.setattr(engine, "_create_session", lambda model_id, device, entry: Double2xSession())

    job = make_job(tmp_path, model_id="fake-2x", device="cpu", width=10, height=6)
    output_path = await engine.run(job)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    with Image.open(output_path) as out_img:
        assert out_img.size == (20, 12)


async def test_run_uses_configured_onnx_tile_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, registry, settings = make_engine(tmp_path, ONNX_TILE_SIZE=32)
    registry.register(make_onnx_entry())
    monkeypatch.setattr(engine, "_create_session", lambda model_id, device, entry: Double2xSession())

    job = make_job(tmp_path, model_id="fake-2x", device="cpu", width=48, height=48)
    output_path = await engine.run(job)

    with Image.open(output_path) as out_img:
        assert out_img.size == (96, 96)


async def test_run_reports_tile_progress_end_to_end_for_tiled_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, registry, settings = make_engine(tmp_path, ONNX_TILE_SIZE=32)
    registry.register(make_onnx_entry())
    monkeypatch.setattr(engine, "_create_session", lambda model_id, device, entry: Double2xSession())

    job = make_job(tmp_path, model_id="fake-2x", device="cpu", width=48, height=48)
    await engine.run(job)

    assert job.metadata["framesTotal"] == 4
    assert job.metadata["framesDone"] == 4
    assert job.metadata["progress"] == pytest.approx(1.0)
    assert job.metadata["stage"] == "upscaling"


async def test_run_does_not_report_tile_progress_for_single_pass_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, registry, settings = make_engine(tmp_path)  # default ONNX_TILE_SIZE=256
    registry.register(make_onnx_entry())
    monkeypatch.setattr(engine, "_create_session", lambda model_id, device, entry: Double2xSession())

    job = make_job(tmp_path, model_id="fake-2x", device="cpu", width=10, height=6)
    await engine.run(job)

    assert "framesDone" not in job.metadata
    assert "framesTotal" not in job.metadata


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


async def test_run_raises_when_engine_not_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, registry, _ = make_engine(tmp_path)
    registry.register(make_onnx_entry())
    monkeypatch.setattr(engine, "available", lambda: False)

    job = make_job(tmp_path, model_id="fake-2x", device="cpu")
    with pytest.raises(RuntimeError, match="onnxruntime is not installed"):
        await engine.run(job)


async def test_run_raises_for_unknown_model_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, _, _ = make_engine(tmp_path)
    monkeypatch.setattr(engine, "available", lambda: True)

    job = make_job(tmp_path, model_id="does-not-exist", device="cpu")
    with pytest.raises(RuntimeError, match="Unknown ONNX model id"):
        await engine.run(job)


async def test_run_raises_for_builtin_ncnn_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, registry, _ = make_engine(tmp_path)
    monkeypatch.setattr(engine, "available", lambda: True)
    builtin_id = next(iter(registry.list())).id

    job = make_job(tmp_path, model_id=builtin_id, device="cpu")
    with pytest.raises(RuntimeError, match="not an ONNX model"):
        await engine.run(job)


async def test_run_raises_when_model_not_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, registry, _ = make_engine(tmp_path)
    registry.register(make_onnx_entry(status=ModelStatus.converting, file_path=None))
    monkeypatch.setattr(engine, "available", lambda: True)

    job = make_job(tmp_path, model_id="fake-2x", device="cpu")
    with pytest.raises(RuntimeError, match="not ready for inference"):
        await engine.run(job)


async def test_run_raises_for_unknown_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, registry, _ = make_engine(tmp_path)
    registry.register(make_onnx_entry())
    monkeypatch.setattr(engine, "available", lambda: True)

    job = make_job(tmp_path, model_id="fake-2x", device="dml:99")
    with pytest.raises(ValueError, match="Unknown device id"):
        await engine.run(job)


async def test_run_wraps_session_creation_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, registry, _ = make_engine(tmp_path)
    registry.register(make_onnx_entry())

    def failing_create_session(model_id: str, device: str, entry: ModelEntry) -> object:
        raise Exception("corrupt graph")

    monkeypatch.setattr(engine, "_create_session", failing_create_session)

    job = make_job(tmp_path, model_id="fake-2x", device="cpu")
    with pytest.raises(RuntimeError, match="Failed to load ONNX model"):
        await engine.run(job)


async def test_run_wraps_inference_memory_failure_with_vram_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, registry, _ = make_engine(tmp_path)
    registry.register(make_onnx_entry())
    monkeypatch.setattr(
        engine, "_create_session", lambda model_id, device, entry: FailingSession("CUDA out of memory")
    )

    job = make_job(tmp_path, model_id="fake-2x", device="cpu")
    with pytest.raises(RuntimeError, match="VRAM"):
        await engine.run(job)


async def test_run_raises_when_output_file_not_produced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, registry, _ = make_engine(tmp_path)
    registry.register(make_onnx_entry())
    monkeypatch.setattr(engine, "_create_session", lambda model_id, device, entry: Double2xSession())
    monkeypatch.setattr(engine, "_run_and_save", lambda *args, **kwargs: None)

    job = make_job(tmp_path, model_id="fake-2x", device="cpu")
    with pytest.raises(RuntimeError, match="no output file was produced"):
        await engine.run(job)
