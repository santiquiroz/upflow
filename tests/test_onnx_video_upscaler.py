from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services.devices_service import DevicesService
from app.services.engines.onnx_video_upscaler import (
    OnnxVideoUpscaler,
    _drain_queue,
    _put_until_cancelled,
    should_tile_frame,
)
from app.services.model_registry import ModelRegistry

# ---------------------------------------------------------------------------
# SP11 Task 2 - OnnxVideoUpscaler. No real onnxruntime session or GPU:
# _create_session is a monkeypatchable seam replaced by Double2xUint8Session,
# a numpy fake that mirrors an InferenceSession over a uint8-in/out graph
# (NHWC uint8 -> doubled NHWC uint8). cv2 is a real dependency here (frame
# PNG I/O), so frames are written to disk and round-tripped.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    # BUILTIN_ONNX_DIR is isolated to tmp so tests never read from or write into
    # the repo's real vendor/realesrgan-onnx/ folder.
    kwargs: dict[str, object] = {
        "RUNTIME_DIR": str(tmp_path / "runtime"),
        "BUILTIN_ONNX_DIR": str(tmp_path / "builtin-onnx"),
    }
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def make_engine(tmp_path: Path, **overrides: object) -> OnnxVideoUpscaler:
    settings = make_settings(tmp_path, **overrides)
    return OnnxVideoUpscaler(settings, ModelRegistry(settings), DevicesService(settings))


class _IoInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class Double2xUint8Session:
    """Fake uint8-graph session: doubles H/W per-pixel on NHWC uint8 input."""

    def __init__(self) -> None:
        self._input = _IoInfo("image")
        self._output = _IoInfo("upscaled")

    def get_inputs(self) -> list[_IoInfo]:
        return [self._input]

    def get_outputs(self) -> list[_IoInfo]:
        return [self._output]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        array = input_feed[self._input.name]  # NHWC uint8
        assert array.dtype == np.uint8
        doubled = np.repeat(np.repeat(array, 2, axis=1), 2, axis=2)
        return [doubled]


def write_frames(frames_in: Path, count: int, height: int = 8, width: int = 12) -> None:
    frames_in.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for index in range(1, count + 1):
        array = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        Image.fromarray(array, "RGB").save(frames_in / f"{index:08d}.png")


def touch_builtin_onnx(settings: Settings, filename: str) -> None:
    onnx_dir = settings.builtin_onnx_path
    onnx_dir.mkdir(parents=True, exist_ok=True)
    (onnx_dir / filename).write_bytes(b"fake-onnx-bytes")


# ---------------------------------------------------------------------------
# should_tile_frame
# ---------------------------------------------------------------------------


def test_should_tile_frame_false_when_under_threshold() -> None:
    assert should_tile_frame(input_pixels=921_600, max_whole_frame_pixels=8_294_400) is False


def test_should_tile_frame_true_when_over_threshold() -> None:
    assert should_tile_frame(input_pixels=10_000_000, max_whole_frame_pixels=8_294_400) is True


def test_should_tile_frame_disabled_when_threshold_zero() -> None:
    assert should_tile_frame(input_pixels=10_000_000_000, max_whole_frame_pixels=0) is False


# ---------------------------------------------------------------------------
# availability / capability probes
# ---------------------------------------------------------------------------


def test_available_false_when_onnxruntime_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine(tmp_path)
    monkeypatch.setattr(engine, "_onnxruntime_available", staticmethod(lambda: False))
    assert engine.available() is False


def test_available_false_when_opencv_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine(tmp_path)
    monkeypatch.setattr(engine, "_opencv_available", staticmethod(lambda: False))
    assert engine.available() is False


def test_has_gpu_execution_provider_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine(tmp_path)
    calls = {"n": 0}

    def probe() -> bool:
        calls["n"] += 1
        return True

    monkeypatch.setattr(engine, "_probe_gpu_execution_provider", staticmethod(probe))
    assert engine.has_gpu_execution_provider() is True
    assert engine.has_gpu_execution_provider() is True
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# builtin_onnx_available
# ---------------------------------------------------------------------------


def test_builtin_onnx_available_true_when_file_present(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    assert engine.builtin_onnx_available("realesr-animevideov3-x4") is True


def test_builtin_onnx_available_false_when_file_absent(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    assert engine.builtin_onnx_available("realesr-animevideov3-x4") is False


def test_builtin_onnx_available_false_for_unknown_model(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    assert engine.builtin_onnx_available("not-a-model") is False


# ---------------------------------------------------------------------------
# run_frames_builtin end-to-end (fake session, real cv2 I/O)
# ---------------------------------------------------------------------------


async def test_run_frames_builtin_upscales_all_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())

    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_frames(frames_in, count=5, height=8, width=12)

    result = await engine.run_frames_builtin(frames_in, frames_out, "realesr-animevideov3-x4", "cpu")

    assert result == frames_out
    output_frames = sorted(frames_out.glob("*.png"))
    assert len(output_frames) == 5
    with Image.open(output_frames[0]) as image:
        assert image.size == (24, 16)  # width*2, height*2


async def test_run_frames_builtin_raises_for_unconfigured_model(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_frames(frames_in, count=1)

    with pytest.raises(RuntimeError, match="No ONNX export configured"):
        await engine.run_frames_builtin(frames_in, frames_out, "does-not-exist", "cpu")


async def test_run_frames_builtin_raises_when_model_file_missing(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)  # no touch_builtin_onnx -> file absent
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_frames(frames_in, count=1)

    with pytest.raises(RuntimeError, match="ONNX model file not found"):
        await engine.run_frames_builtin(frames_in, frames_out, "realesr-animevideov3-x4", "cpu")


async def test_run_frames_builtin_validates_output_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())
    # Pipeline that only saves the first frame -> output count mismatch.
    monkeypatch.setattr(engine, "_run_pipeline", lambda *args, **kwargs: None)

    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_frames(frames_in, count=3)

    with pytest.raises(RuntimeError, match="no output frames were produced"):
        await engine.run_frames_builtin(frames_in, frames_out, "realesr-animevideov3-x4", "cpu")


# ---------------------------------------------------------------------------
# whole-frame vs tiling decision
# ---------------------------------------------------------------------------


def test_upscale_one_uses_whole_frame_under_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine(tmp_path)
    session = Double2xUint8Session()
    frame = np.random.default_rng(1).integers(0, 256, (1, 8, 12, 3), dtype=np.uint8)

    calls = {"whole": 0, "tiled": 0}
    monkeypatch.setattr(engine, "_infer_frame", lambda s, f, d: (calls.__setitem__("whole", calls["whole"] + 1), f)[1])
    monkeypatch.setattr(engine, "_infer_tiled", lambda s, f, d: (calls.__setitem__("tiled", calls["tiled"] + 1), f)[1])

    engine._upscale_one(session, frame, "cpu")

    assert calls == {"whole": 1, "tiled": 0}


def test_upscale_one_tiles_over_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine(tmp_path, ONNX_WHOLE_FRAME_MAX_PIXELS=10)
    session = Double2xUint8Session()
    frame = np.random.default_rng(1).integers(0, 256, (1, 8, 12, 3), dtype=np.uint8)  # 96 px > 10

    calls = {"whole": 0, "tiled": 0}
    monkeypatch.setattr(engine, "_infer_frame", lambda s, f, d: (calls.__setitem__("whole", calls["whole"] + 1), f)[1])
    monkeypatch.setattr(engine, "_infer_tiled", lambda s, f, d: (calls.__setitem__("tiled", calls["tiled"] + 1), f)[1])

    engine._upscale_one(session, frame, "cpu")

    assert calls == {"whole": 0, "tiled": 1}


def test_upscale_one_falls_back_to_tiling_on_oom_and_sticks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine(tmp_path)
    frame = np.random.default_rng(1).integers(0, 256, (1, 8, 12, 3), dtype=np.uint8)  # under threshold -> whole-frame

    def whole_frame_oom(s, f, d):
        raise RuntimeError("Failed to allocate memory: out of memory (D3D12)")

    tiled_calls = {"n": 0}
    monkeypatch.setattr(engine, "_infer_frame", whole_frame_oom)
    monkeypatch.setattr(engine, "_infer_tiled", lambda s, f, d: (tiled_calls.__setitem__("n", tiled_calls["n"] + 1), f)[1])

    # First frame: whole-frame OOM -> retries tiled, returns force_tiled=True.
    out1, force1 = engine._upscale_one(None, frame, "dml:0", force_tiled=False)
    assert force1 is True
    assert tiled_calls["n"] == 1
    # Next frame with force_tiled carried in: goes straight to tiling, no whole-frame attempt.
    out2, force2 = engine._upscale_one(None, frame, "dml:0", force_tiled=True)
    assert force2 is True
    assert tiled_calls["n"] == 2


def test_upscale_one_reraises_non_oom_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = make_engine(tmp_path)
    frame = np.random.default_rng(1).integers(0, 256, (1, 8, 12, 3), dtype=np.uint8)
    monkeypatch.setattr(engine, "_infer_frame", lambda s, f, d: (_ for _ in ()).throw(ValueError("bad shape")))
    monkeypatch.setattr(engine, "_infer_tiled", lambda s, f, d: f)
    with pytest.raises(ValueError, match="bad shape"):
        engine._upscale_one(None, frame, "dml:0")


def test_save_queue_maxsize_capped_by_byte_budget(tmp_path: Path) -> None:
    # 1024x1024 input x4 = 4096x4096x3 = 48.0 MB/frame. Budget 150MB // 48 = 3,
    # which sits strictly between the floor (n_save=2) and ceiling (n_save*2=4),
    # so the byte budget is what decides -> 3.
    engine = make_engine(tmp_path, ONNX_VIDEO_MAX_PIPELINE_MB=150, ONNX_VIDEO_SAVE_THREADS=2)
    frames_in = tmp_path / "big"
    write_frames(frames_in, count=1, height=1024, width=1024)
    paths = sorted(frames_in.glob("*.png"))
    assert engine._save_queue_maxsize(paths, scale=4, n_save=2) == 3


def test_save_queue_maxsize_floors_at_n_save(tmp_path: Path) -> None:
    # Tiny budget must never starve savers: floor = n_save.
    engine = make_engine(tmp_path, ONNX_VIDEO_MAX_PIPELINE_MB=1, ONNX_VIDEO_SAVE_THREADS=8)
    frames_in = tmp_path / "big"
    write_frames(frames_in, count=1, height=2048, width=2048)
    paths = sorted(frames_in.glob("*.png"))
    assert engine._save_queue_maxsize(paths, scale=4, n_save=8) == 8


def test_save_queue_maxsize_defaults_when_no_frames(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, ONNX_VIDEO_SAVE_THREADS=5)
    assert engine._save_queue_maxsize([], scale=4, n_save=5) == 10  # n_save*2


def test_infer_tiled_matches_whole_frame_for_double_session(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, ONNX_TILE_SIZE=16)
    session = Double2xUint8Session()
    frame = np.random.default_rng(2).integers(0, 256, (1, 40, 40, 3), dtype=np.uint8)

    whole = engine._infer_frame(session, frame, "cpu")
    tiled = engine._infer_tiled(session, frame, "cpu")

    assert tiled.shape == whole.shape == (1, 80, 80, 3)
    assert np.array_equal(tiled, whole)


# ---------------------------------------------------------------------------
# session cache
# ---------------------------------------------------------------------------


def test_get_session_caches_by_path_and_device(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    calls: list[tuple[str, str]] = []
    engine._create_session = lambda model_path, device: (calls.append((model_path, device)), object())[1]  # type: ignore[method-assign]

    first = engine._get_session("/models/x4.onnx", "cpu")
    second = engine._get_session("/models/x4.onnx", "cpu")

    assert first is second
    assert calls == [("/models/x4.onnx", "cpu")]


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------


def test_run_pipeline_stops_immediately_when_cancel_event_preset(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    session = Double2xUint8Session()
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    frames_out.mkdir(parents=True, exist_ok=True)
    write_frames(frames_in, count=5)
    frame_paths = sorted(frames_in.glob("*.png"))

    cancel_event = threading.Event()
    cancel_event.set()
    engine._run_pipeline(session, frame_paths, frames_out, "cpu", cancel_event)

    assert list(frames_out.glob("*.png")) == []


# ---------------------------------------------------------------------------
# Real vendored model (skip-if-missing): exercises a genuine onnxruntime
# session on CPUExecutionProvider (no GPU) end-to-end. Skipped in CI where
# vendor/realesrgan-onnx/ (gitignored) is absent -- run
# scripts/download-realesrgan-onnx.ps1 to populate it.
# ---------------------------------------------------------------------------

_REAL_ONNX = Settings(_env_file=None).builtin_onnx_path / "realesr-animevideov3-x4-uint8.onnx"


@pytest.mark.skipif(not _REAL_ONNX.exists(), reason="vendored realesr-animevideov3-x4 ONNX not present")
async def test_run_frames_builtin_with_real_model_on_cpu(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))  # real BUILTIN_ONNX_DIR
    engine = OnnxVideoUpscaler(settings, ModelRegistry(settings), DevicesService(settings))
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_frames(frames_in, count=3, height=16, width=24)

    result = await engine.run_frames_builtin(frames_in, frames_out, "realesr-animevideov3-x4", "cpu")

    output_frames = sorted(result.glob("*.png"))
    assert len(output_frames) == 3
    with Image.open(output_frames[0]) as image:
        assert image.size == (96, 64)  # width*4, height*4


async def test_run_frames_builtin_cancel_sets_event_and_reraises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    captured: dict[str, threading.Event] = {}

    def blocking_until_cancelled(frames_in, frames_out, onnx_path, device, cancel_event, scale=4) -> None:
        captured["event"] = cancel_event
        cancel_event.wait(timeout=5)

    monkeypatch.setattr(engine, "_run_frames_blocking", blocking_until_cancelled)

    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_frames(frames_in, count=2)

    task = asyncio.create_task(
        engine.run_frames_builtin(frames_in, frames_out, "realesr-animevideov3-x4", "cpu")
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert captured["event"].is_set()


# ---------------------------------------------------------------------------
# cancel-aware queue put (CRITICAL deadlock regression): a loader/infer thread
# blocked on a full queue must observe cancel_event and exit instead of hanging
# forever (which would leak a thread from the shared asyncio executor pool).
# ---------------------------------------------------------------------------


def test_put_until_cancelled_returns_true_when_slot_available() -> None:
    q: queue.Queue = queue.Queue(maxsize=1)
    cancel = threading.Event()
    assert _put_until_cancelled(q, "item", cancel, timeout=0.01) is True
    assert q.get_nowait() == "item"


def test_put_until_cancelled_returns_false_without_enqueue_when_cancelled_on_full_queue() -> None:
    q: queue.Queue = queue.Queue(maxsize=1)
    q.put("occupied")  # queue is now full
    cancel = threading.Event()
    cancel.set()
    assert _put_until_cancelled(q, "item", cancel, timeout=0.01) is False
    assert q.qsize() == 1  # the blocked item was NOT enqueued


def test_put_until_cancelled_unblocks_when_cancel_set_from_another_thread() -> None:
    # The real deadlock shape: queue stays full, and cancel arrives later. The
    # put must return (False) rather than hang forever.
    q: queue.Queue = queue.Queue(maxsize=1)
    q.put("occupied")
    cancel = threading.Event()
    result: dict[str, bool] = {}

    def worker() -> None:
        result["value"] = _put_until_cancelled(q, "item", cancel, timeout=0.02)

    thread = threading.Thread(target=worker)
    thread.start()
    cancel.set()
    thread.join(timeout=5)
    assert not thread.is_alive(), "put hung on a full queue after cancel (deadlock regression)"
    assert result["value"] is False


def test_drain_queue_empties_all_pending_items() -> None:
    q: queue.Queue = queue.Queue()
    for index in range(5):
        q.put(index)
    _drain_queue(q)
    assert q.empty()


async def test_run_frames_builtin_cancel_does_not_leave_worker_thread_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # After a cancel, run_frames_builtin must WAIT for the worker to fully unwind
    # before propagating, so the caller's work_dir cleanup can't race live I/O.
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    finished = threading.Event()

    def blocking_until_cancelled(frames_in, frames_out, onnx_path, device, cancel_event, scale=4) -> None:
        cancel_event.wait(timeout=5)
        finished.set()  # simulates the pipeline finishing its teardown

    monkeypatch.setattr(engine, "_run_frames_blocking", blocking_until_cancelled)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_frames(frames_in, count=2)

    task = asyncio.create_task(
        engine.run_frames_builtin(frames_in, frames_out, "realesr-animevideov3-x4", "cpu")
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The worker must have completed its teardown BEFORE the cancel propagated.
    assert finished.is_set()
