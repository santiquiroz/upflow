from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.config import Settings
from app.services.engines.gmfss import softsplat_cl
from app.services.engines.gmfss.assets import GRAPH_NAMES, GmfssAssets
from app.services.engines.gmfss.pipeline import GmfssDriver
from app.services.engines.gmfss_engine import (
    FP16_FUSIONNET_FILENAME,
    GmfssEngine,
    _build_interpolation_plan,
    _distribute_extra_frames,
    _graph_runner,
    _pair_timesteps,
)
import app.services.engines.gmfss_engine as gmfss_engine_module
from app.services.gpu_session_coordinator import GpuSessionCoordinator

# ---------------------------------------------------------------------------
# Task 4.1 - GmfssEngine: fake-graph inference path (patterned after
# test_audiosr_driver.py's FakeGraphRunner / test_audiosr_restorer.py's
# FakeSession), exact frame-count arithmetic for multiplier AND
# target_frame_count modes, availability gate, cancel (shield+await, no
# zombie thread), and the ORT_DISABLE_ALL DirectML session-hang workaround.
# No real onnxruntime session/model files are required for any test here.
# ---------------------------------------------------------------------------

FULL_H, FULL_W = 16, 24  # tiny fixed "padded" resolution, non-square on purpose
SOURCE_H, SOURCE_W = 8, 12  # deliberately different from FULL_H/W to exercise resize


def make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "gmfss"
    model_dir.mkdir(parents=True)
    manifest = {
        "resolution": {"fixed_padded_hw": [FULL_H, FULL_W]},
        "required_files": ["manifest.json"] + [f"{name}.onnx" for name in GRAPH_NAMES],
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    for name in GRAPH_NAMES:
        (model_dir / f"{name}.onnx").write_bytes(b"fake")
    return model_dir


def make_settings(tmp_path: Path, enabled: bool = True, **overrides: object) -> Settings:
    model_dir = make_model_dir(tmp_path)
    return Settings(
        _env_file=None,
        ENABLE_GMFSS=enabled,
        GMFSS_MODEL_DIR=str(model_dir),
        **overrides,  # type: ignore[arg-type]
    )


class FakeSession:
    """Deterministic stand-in for onnxruntime.InferenceSession.run, shaped to
    match GMFSS's graph conventions (featurenet downsamples by 2/4/8;
    fusionnet upsamples half-res inputs back to full padded res) -- same
    fake-graph pattern as the port project's own tests/test_pipeline.py and
    this repo's test_audiosr_driver.py."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, _outputs: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        if self.name == "featurenet":
            n, _c, h, w = feeds["img"].shape
            return [
                np.full((n, ch, h // div, w // div), 1.0, dtype=np.float32)
                for ch, div in zip((4, 6, 8), (2, 4, 8))
            ]
        if self.name == "gmflow":
            n, _c, h, w = feeds["img0_half"].shape
            return [np.full((n, 2, h, w), 2.0, dtype=np.float32)]
        if self.name == "metricnet":
            n, _c, h, w = feeds["img0_half"].shape
            metric = np.zeros((n, 1, h, w), dtype=np.float32)
            return [metric.copy(), metric.copy()]
        if self.name == "fusionnet":
            n = feeds["fusion_rgb"].shape[0]
            h_half, w_half = feeds["fusion_rgb"].shape[2], feeds["fusion_rgb"].shape[3]
            out = np.full((n, 3, h_half * 2, w_half * 2), 0.5, dtype=np.float32)
            return [out]
        raise AssertionError(self.name)


def fake_sessions(_device: str) -> dict[str, Any]:
    return {name: FakeSession(name) for name in GRAPH_NAMES}


def write_fake_source_frames(
    directory: Path, count: int, height: int = SOURCE_H, width: int = SOURCE_W
) -> None:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        value = (index * 17) % 256
        frame = np.full((height, width, 3), value, dtype=np.uint8)
        ok = cv2.imwrite(str(directory / f"{index + 1:08d}.png"), frame)
        assert ok


def count_frames(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.png"))


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------


def test_available_follows_settings_gate(tmp_path: Path) -> None:
    assert GmfssEngine(make_settings(tmp_path, enabled=True), GpuSessionCoordinator()).available() is True
    assert GmfssEngine(make_settings(tmp_path / "off", enabled=False), GpuSessionCoordinator()).available() is False


async def test_run_when_unavailable_raises_actionable_error(tmp_path: Path) -> None:
    engine = GmfssEngine(make_settings(tmp_path, enabled=False), GpuSessionCoordinator())
    frames_in = tmp_path / "frames-in"
    write_fake_source_frames(frames_in, 4)

    with pytest.raises(RuntimeError, match="ENABLE_GMFSS"):
        await engine.run(frames_in, tmp_path / "frames-out", source_frame_count=4, multiplier=2)


# ---------------------------------------------------------------------------
# frame-pair -> timestep -> output-frame arithmetic (pure functions)
# ---------------------------------------------------------------------------


def test_distribute_extra_frames_is_exact_and_within_one_of_mean() -> None:
    cases = [(9, 9), (9, 27), (4, 5), (100, 101), (96, 144), (1, 0), (1, 7)]
    for pair_count, extra in cases:
        counts = _distribute_extra_frames(pair_count, extra)
        assert len(counts) == pair_count
        assert sum(counts) == extra
        if counts:
            assert max(counts) - min(counts) <= 1


def test_distribute_extra_frames_rejects_nonzero_extra_with_no_pairs() -> None:
    with pytest.raises(ValueError):
        _distribute_extra_frames(0, 3)
    assert _distribute_extra_frames(0, 0) == []


def test_pair_timesteps_evenly_spaced_open_interval() -> None:
    assert _pair_timesteps(0) == []
    assert _pair_timesteps(1) == [0.5]
    assert _pair_timesteps(2) == [pytest.approx(1 / 3), pytest.approx(2 / 3)]
    assert _pair_timesteps(3) == [0.25, 0.5, 0.75]


@pytest.mark.parametrize(
    "source_frame_count,target_frame_count",
    [
        (10, 20),  # 2x multiplier
        (10, 30),  # 3x multiplier
        (5, 10),  # small N, 2x (known uneven Bresenham case)
        (97, 241),  # non-integer ratio target_fps case (~23.976fps -> 60fps)
        (24, 60),  # non-integer ratio (24fps -> 60fps)
        (2, 2),  # no interpolation needed
        (1, 1),  # single source frame, no interpolation needed
    ],
)
def test_build_interpolation_plan_produces_exact_total(
    source_frame_count: int, target_frame_count: int
) -> None:
    plan = _build_interpolation_plan(source_frame_count, target_frame_count)

    assert len(plan) == max(source_frame_count - 1, 0)
    total = source_frame_count + sum(len(timesteps) for timesteps in plan)
    assert total == target_frame_count
    for timesteps in plan:
        assert all(0.0 < t < 1.0 for t in timesteps)
        assert timesteps == sorted(timesteps)


def test_build_interpolation_plan_rejects_target_smaller_than_source() -> None:
    with pytest.raises(RuntimeError, match="smaller than"):
        _build_interpolation_plan(10, 5)


def test_build_interpolation_plan_rejects_single_frame_with_extra_target() -> None:
    with pytest.raises(RuntimeError, match="at least 2 source frames"):
        _build_interpolation_plan(1, 3)


def test_resolve_target_frame_count_matches_rife_semantics() -> None:
    assert GmfssEngine._resolve_target_frame_count(10, 2, None) == 20
    assert GmfssEngine._resolve_target_frame_count(10, 2, 25) == 25  # target overrides multiplier


# ---------------------------------------------------------------------------
# end-to-end run() with fake sessions: exact output frame count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_frame_count,multiplier,expected_total", [(6, 2, 12), (6, 3, 18)])
async def test_run_produces_exact_frame_count_for_multiplier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_frame_count: int,
    multiplier: int,
    expected_total: int,
) -> None:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(engine, "_create_sessions", fake_sessions)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_fake_source_frames(frames_in, source_frame_count)

    result = await engine.run(frames_in, frames_out, source_frame_count, multiplier, device="cpu")

    assert result == frames_out
    assert count_frames(frames_out) == expected_total


@pytest.mark.parametrize(
    "source_frame_count,target_frame_count",
    [
        (24, 60),
        (97, 241),
        (4, 5),  # some pairs get 0 extra frames -- exercises the skip-the-network-call branch
    ],
)
async def test_run_produces_exact_frame_count_for_target_fps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_frame_count: int, target_frame_count: int
) -> None:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(engine, "_create_sessions", fake_sessions)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_fake_source_frames(frames_in, source_frame_count)

    result = await engine.run(
        frames_in, frames_out, source_frame_count, target_frame_count=target_frame_count, device="cpu"
    )

    assert result == frames_out
    assert count_frames(frames_out) == target_frame_count


async def test_run_copies_boundary_frames_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Design decision: source frames are never re-synthesized through the
    # network at t=0/t=1 -- they are copied byte-for-byte into their output
    # slot. This verifies that decision instead of just the total count.
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(engine, "_create_sessions", fake_sessions)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_fake_source_frames(frames_in, 3)

    await engine.run(frames_in, frames_out, 3, 2, device="cpu")

    assert (frames_out / "00000001.png").read_bytes() == (frames_in / "00000001.png").read_bytes()


async def test_run_raises_when_source_frame_count_mismatches_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(engine, "_create_sessions", fake_sessions)
    frames_in = tmp_path / "frames-in"
    write_fake_source_frames(frames_in, 3)

    with pytest.raises(RuntimeError, match="expected 5"):
        await engine.run(frames_in, tmp_path / "frames-out", source_frame_count=5, multiplier=2)


# ---------------------------------------------------------------------------
# run_frames_fused (Fase 2 Task 7): pull-based generator yielding each output
# frame ALREADY interpolated + upscaled (via an injected upscale_frame
# callback), with NO intermediate PNG round-trip. Shares the load/resize/
# interpolate logic with run() but not its threaded save pipeline -- a
# generator is pull-based, so the caller (Task 8) owns threading/cancellation.
# ---------------------------------------------------------------------------


def test_run_frames_fused_calls_upscale_frame_for_every_output_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(engine, "_create_sessions", fake_sessions)
    frames_in = tmp_path / "frames-in"
    write_fake_source_frames(frames_in, 2)
    upscale_calls: list[tuple[int, ...]] = []

    def fake_upscale(frame: np.ndarray) -> np.ndarray:
        upscale_calls.append(frame.shape)
        return frame * 2  # marker: proves the yielded frame came from upscale_frame

    frames = list(
        engine.run_frames_fused(
            frames_in,
            source_frame_count=2,
            multiplier=2,
            target_frame_count=None,
            device="cpu",
            upscale_frame=fake_upscale,
        )
    )

    assert len(frames) == 4  # source_frame_count * multiplier
    assert len(upscale_calls) == 4  # every output frame passed through upscale_frame


def test_run_frames_fused_never_writes_intermediate_png(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(engine, "_create_sessions", fake_sessions)
    frames_in = tmp_path / "frames-in"
    write_fake_source_frames(frames_in, 2)
    intermediate_dir = tmp_path / "should-stay-empty"
    intermediate_dir.mkdir()

    list(
        engine.run_frames_fused(
            frames_in,
            source_frame_count=2,
            multiplier=2,
            target_frame_count=None,
            device="cpu",
            upscale_frame=lambda f: f,
        )
    )

    assert list(intermediate_dir.iterdir()) == []  # run_frames_fused writes no PNGs
    assert count_frames(frames_in) == 2  # source dir untouched, nothing written back


def test_run_frames_fused_yields_nhwc_uint8_with_pixel_identical_source_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Fused analog of test_run_copies_boundary_frames_byte_identical: the two
    # boundary source frames (t=0/t=1) are the RAW decoded pixels, NOT degraded
    # by the padding resize round-trip that interpolated frames go through, and
    # every frame is NHWC uint8 RGB at the ORIGINAL source resolution -- the
    # exact format OnnxVideoUpscaler._upscale_one consumes (Task 8 wiring).
    import cv2

    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(engine, "_create_sessions", fake_sessions)
    frames_in = tmp_path / "frames-in"
    write_fake_source_frames(frames_in, 2)

    frames = list(
        engine.run_frames_fused(
            frames_in,
            source_frame_count=2,
            multiplier=2,
            target_frame_count=None,
            device="cpu",
            upscale_frame=lambda f: f,
        )
    )

    for frame in frames:
        assert frame.dtype == np.uint8
        assert frame.shape == (1, SOURCE_H, SOURCE_W, 3)  # NHWC RGB at source res

    first = cv2.cvtColor(cv2.imread(str(frames_in / "00000001.png")), cv2.COLOR_BGR2RGB)
    last = cv2.cvtColor(cv2.imread(str(frames_in / "00000002.png")), cv2.COLOR_BGR2RGB)
    assert np.array_equal(frames[0][0], first)  # source[0] pixel-identical (t=0)
    assert np.array_equal(frames[-1][0], last)  # source[1] pixel-identical (t=1)


# ---------------------------------------------------------------------------
# session cache (LRU 1, AudioSrRestorer pattern) + ORT_DISABLE_ALL
# ---------------------------------------------------------------------------


def test_session_cache_keeps_single_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    built: list[str] = []

    def tracking_sessions(device: str) -> dict[str, Any]:
        built.append(device)
        return fake_sessions(device)

    monkeypatch.setattr(engine, "_create_sessions", tracking_sessions)

    engine._get_sessions("cpu")
    engine._get_sessions("cpu")
    assert built == ["cpu"]  # cached

    engine._get_sessions("dml:0")
    engine._get_sessions("cpu")  # evicted by dml:0 -> rebuilt
    assert built == ["cpu", "dml:0", "cpu"]


def test_create_sessions_disables_graph_optimization_for_every_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Real (non-fake) session-creation path: only onnxruntime.InferenceSession
    # itself is mocked, so the actual SessionOptions object built by
    # _create_sessions is what gets asserted on. Regression guard for the
    # DXGI_ERROR_DEVICE_HUNG MetricNet gotcha -- applied to EVERY graph, not
    # just metricnet's.
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    captured: list[tuple[str, Any, Any]] = []

    import onnxruntime as ort

    class FakeInferenceSession:
        def __init__(self, path: str, sess_options: Any = None, providers: Any = None) -> None:
            captured.append((path, sess_options, providers))

    monkeypatch.setattr(ort, "InferenceSession", FakeInferenceSession)

    sessions = engine._create_sessions("cpu")

    assert set(sessions) == set(GRAPH_NAMES)
    assert len(captured) == len(GRAPH_NAMES)
    for _path, sess_options, providers in captured:
        assert sess_options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        assert providers == ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# GpuSessionCoordinator wiring (Fase 1 Task 3) - release_device evicts only
# its own device's cache entry, acquire() runs before any session is built.
# Same pattern as AudioSrRestorer/ApolloRestorer (Fase 1 Task 2).
# ---------------------------------------------------------------------------


def test_release_device_clears_cached_sessions_for_that_device_only(tmp_path: Path) -> None:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    engine._session_cache["dml:0"] = {"featurenet": "fake"}
    engine._session_cache["dml:1"] = {"featurenet": "fake-1"}

    engine.release_device("dml:0")

    assert "dml:0" not in engine._session_cache
    assert "dml:1" in engine._session_cache


def test_release_device_on_empty_cache_is_a_noop(tmp_path: Path) -> None:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())

    engine.release_device("dml:0")  # no debe lanzar


def test_get_sessions_calls_coordinator_acquire_before_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gpu_coordinator = GpuSessionCoordinator()
    engine = GmfssEngine(make_settings(tmp_path), gpu_coordinator)
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(gpu_coordinator, "acquire", lambda device, owner: calls.append((device, owner)))
    monkeypatch.setattr(engine, "_create_sessions", lambda device: {"featurenet": "fake"})

    engine._get_sessions("dml:0")

    assert calls == [("dml:0", engine)]


# ---------------------------------------------------------------------------
# fp16 fusionnet + GPU splat selection (Task 4.2). Real measured config from
# the port project's Task 3.2 (0.72-0.73fps @1080p 2x on a 7800 XT): DirectML
# + fp16 fusionnet + OpenCL splat, vs ~0.2fps fp32/CPU-splat. fp16 only wins
# on a GPU EP -- CPU-EP fp16 is emulated (slower), so CPU always keeps fp32,
# same rule ONNX_PREFER_FP16 already applies to the builtin upscale models.
# ---------------------------------------------------------------------------


def _fake_session_path_capture(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import onnxruntime as ort

    captured_paths: list[str] = []

    class FakeInferenceSession:
        def __init__(self, path: str, sess_options: Any = None, providers: Any = None) -> None:
            captured_paths.append(path)

    monkeypatch.setattr(ort, "InferenceSession", FakeInferenceSession)
    return captured_paths


def test_create_sessions_uses_fp32_fusionnet_when_fp16_file_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    captured_paths = _fake_session_path_capture(monkeypatch)

    engine._create_sessions("dml:0")

    paths_by_name = dict(zip(GRAPH_NAMES, captured_paths))
    assert Path(paths_by_name["fusionnet"]).name == "fusionnet.onnx"


def test_create_sessions_prefers_fp16_fusionnet_on_gpu_device_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    (settings.gmfss_model_dir_path / FP16_FUSIONNET_FILENAME).write_bytes(b"fake-fp16")
    engine = GmfssEngine(settings, GpuSessionCoordinator())
    captured_paths = _fake_session_path_capture(monkeypatch)

    engine._create_sessions("dml:0")

    paths_by_name = dict(zip(GRAPH_NAMES, captured_paths))
    assert Path(paths_by_name["fusionnet"]).name == FP16_FUSIONNET_FILENAME
    # Only fusionnet switches -- the other 3 graphs have no fp16 variant (Task
    # 3.2: onnxconverter-common rejected them at load time).
    assert Path(paths_by_name["featurenet"]).name == "featurenet.onnx"


def test_create_sessions_uses_fp32_fusionnet_on_cpu_device_even_when_fp16_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    (settings.gmfss_model_dir_path / FP16_FUSIONNET_FILENAME).write_bytes(b"fake-fp16")
    engine = GmfssEngine(settings, GpuSessionCoordinator())
    captured_paths = _fake_session_path_capture(monkeypatch)

    engine._create_sessions("cpu")

    paths_by_name = dict(zip(GRAPH_NAMES, captured_paths))
    assert Path(paths_by_name["fusionnet"]).name == "fusionnet.onnx"


def test_run_blocking_passes_gpu_splat_fn_to_gmfss_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    monkeypatch.setattr(engine, "_create_sessions", fake_sessions)
    captured_kwargs: dict[str, Any] = {}

    class RecordingDriver(GmfssDriver):
        def __init__(self, assets: Any, run_graph: Any, splat_fn: Any = None) -> None:
            captured_kwargs["splat_fn"] = splat_fn
            super().__init__(assets, run_graph, splat_fn=splat_fn)

    monkeypatch.setattr(gmfss_engine_module, "GmfssDriver", RecordingDriver)

    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_out.mkdir()
    write_fake_source_frames(frames_in, 3)

    engine._run_blocking(frames_in, frames_out, 3, 4, "dml:0", threading.Event())

    assert captured_kwargs["splat_fn"] is softsplat_cl.splat_softmax


# ---------------------------------------------------------------------------
# cancel: shield+await (no work-dir race) and pre-set cancel (no zombie
# pipeline threads) -- AudioSrRestorer / OnnxVideoUpscaler pattern.
# ---------------------------------------------------------------------------


async def test_run_cancel_waits_for_worker_thread_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review lesson carried over from SP13 (AudioSR): if run() re-raises
    # CancelledError without waiting for the worker thread, the caller's
    # finally-block rmtree of the work dir races a straggler write that
    # would resurrect it. The contract is: by the time the cancel propagates,
    # the worker thread has ALREADY finished.
    engine = GmfssEngine(make_settings(tmp_path), GpuSessionCoordinator())
    worker_finished = threading.Event()

    def slow_worker(
        frames_in: Path,
        frames_out: Path,
        source_frame_count: int,
        target_frame_count: int,
        device: str,
        cancel_event: threading.Event,
    ) -> None:
        cancel_event.wait(timeout=10)
        time.sleep(0.2)  # simulates a non-interruptible write in flight
        worker_finished.set()

    monkeypatch.setattr(engine, "_run_blocking", slow_worker)
    frames_in = tmp_path / "frames-in"
    write_fake_source_frames(frames_in, 4)

    task = asyncio.create_task(engine.run(frames_in, tmp_path / "frames-out", 4, 2))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert worker_finished.is_set(), "run() propagated the cancel before the worker thread finished"


def test_run_pair_pipeline_stops_immediately_when_cancel_event_preset(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    engine = GmfssEngine(settings, GpuSessionCoordinator())
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_out.mkdir()
    write_fake_source_frames(frames_in, 6)
    frame_paths = sorted(frames_in.glob("*.png"))

    assets = GmfssAssets.load(settings.gmfss_model_dir_path)
    driver = GmfssDriver(assets, _graph_runner(fake_sessions("cpu")))
    plan = _build_interpolation_plan(6, 12)
    cancel_event = threading.Event()
    cancel_event.set()

    threads_before = set(threading.enumerate())
    engine._run_pair_pipeline(driver, assets.padded_hw, frame_paths, plan, frames_out, cancel_event)
    threads_after = set(threading.enumerate())

    assert count_frames(frames_out) == 0
    assert threads_after <= threads_before, "pipeline left a thread running after a pre-set cancel"
