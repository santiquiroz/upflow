from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.services.devices_service import DevicesService
from app.services.engines.gmfss.assets import GRAPH_NAMES
from app.services.engines.gmfss_engine import GmfssEngine
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler
from app.services.model_registry import ModelRegistry
from app.models import VideoUpscaleJob
from app.services.video_upscaler import VideoUpscaler

# ---------------------------------------------------------------------------
# Fase 2 Task 8 - VideoUpscaler fuses GMFSS interpolate + ONNX upscale into a
# single in-process pass (no frames-interp PNG round-trip) ONLY when
# interp_engine == "gmfss" AND the resolved upscale backend is the in-process
# ONNX one. Every other combination (RIFE, the NCNN upscaler, HF ONNX models,
# no interpolation requested) keeps the unchanged two-pass path.
#
# The parity test is the phase's zero-tolerance quality gate: the fused output
# must be BYTE-IDENTICAL to the two-pass output for the same input, using the
# same deterministic fake GMFSS driver + fake ONNX session for both paths.
# Fake-session patterns mirror tests/test_gmfss_engine.py (FakeSession) and
# tests/test_onnx_video_upscaler.py (Double2xUint8Session) so no real
# onnxruntime session, model file, or GPU is needed.
# ---------------------------------------------------------------------------

# Tiny fixed resolutions, non-square on purpose (match test_gmfss_engine.py).
FULL_H, FULL_W = 16, 24  # GMFSS "padded" resolution
SOURCE_H, SOURCE_W = 8, 12  # source frame resolution (exercises the resize)


# --- fakes ------------------------------------------------------------------


class FakeNcnnEngine:
    def available(self) -> bool:
        return True


class FakeMediaTools:
    def available(self) -> bool:
        return True


class FakeDevicesService:
    def __init__(self, valid_ids: tuple[str, ...] = ("cpu", "dml:0")) -> None:
        self._valid_ids = valid_ids

    def list_devices(self) -> list[dict]:
        return [{"id": device_id} for device_id in self._valid_ids]

    def validate(self, device_id: str) -> dict:
        if device_id not in self._valid_ids:
            raise ValueError(f"Unknown device id: {device_id!r}")
        return {"id": device_id}


class FakeOnnxVideoEngine:
    """Backend-resolution stand-in for OnnxVideoUpscaler in the gate tests. The
    gate only ever asks it about availability/gpu-ep/builtin-model presence."""

    def __init__(self, *, available: bool = True, gpu_ep: bool = True, builtin_available: bool = True) -> None:
        self._available = available
        self._gpu_ep = gpu_ep
        self._builtin_available = builtin_available

    def available(self) -> bool:
        return self._available

    def has_gpu_execution_provider(self) -> bool:
        return self._gpu_ep

    def builtin_onnx_available(self, engine_model_name: str) -> bool:
        return self._builtin_available


class FakeGmfssSession:
    """Deterministic stand-in for onnxruntime.InferenceSession.run over GMFSS's
    4 graphs -- copied from tests/test_gmfss_engine.py's FakeSession."""

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
            return [np.full((n, 3, h_half * 2, w_half * 2), 0.5, dtype=np.float32)]
        raise AssertionError(self.name)


def fake_gmfss_sessions(_device: str) -> dict[str, Any]:
    return {name: FakeGmfssSession(name) for name in GRAPH_NAMES}


class Double2xUint8Session:
    """Deterministic fake uint8-graph ONNX session: doubles H/W per-pixel on
    NHWC uint8 input -- copied from tests/test_onnx_video_upscaler.py."""

    class _IoInfo:
        def __init__(self, name: str) -> None:
            self.name = name

    def __init__(self) -> None:
        self._input = Double2xUint8Session._IoInfo("image")
        self._output = Double2xUint8Session._IoInfo("upscaled")

    def get_inputs(self) -> list[Any]:
        return [self._input]

    def get_outputs(self) -> list[Any]:
        return [self._output]

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        array = input_feed[self._input.name]
        assert array.dtype == np.uint8
        return [np.repeat(np.repeat(array, 2, axis=1), 2, axis=2)]


# --- builders ---------------------------------------------------------------


def make_combined_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Settings satisfying BOTH the GMFSS engine (model dir + ENABLE_GMFSS) and
    the ONNX video engine (isolated builtin-onnx dir)."""
    gmfss_dir = tmp_path / "gmfss"
    gmfss_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "resolution": {"fixed_padded_hw": [FULL_H, FULL_W]},
        "required_files": ["manifest.json"] + [f"{name}.onnx" for name in GRAPH_NAMES],
    }
    (gmfss_dir / "manifest.json").write_text(json.dumps(manifest))
    for name in GRAPH_NAMES:
        (gmfss_dir / f"{name}.onnx").write_bytes(b"fake")
    kwargs: dict[str, object] = {
        "RUNTIME_DIR": str(tmp_path / "runtime"),
        "BUILTIN_ONNX_DIR": str(tmp_path / "builtin-onnx"),
        "ENABLE_GMFSS": True,
        "GMFSS_MODEL_DIR": str(gmfss_dir),
    }
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def touch_builtin_onnx(settings: Settings, filename: str) -> None:
    onnx_dir = settings.builtin_onnx_path
    onnx_dir.mkdir(parents=True, exist_ok=True)
    (onnx_dir / filename).write_bytes(b"fake-onnx-bytes")


def write_source_frames(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        value = (index * 17) % 256
        frame = np.full((SOURCE_H, SOURCE_W, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(directory / f"{index + 1:08d}.png"), frame)


def make_video_job(source_path: Path, **overrides: object) -> VideoUpscaleJob:
    fields: dict[str, object] = dict(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesr-animevideov3-x4",
        scale=4,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        model_id=None,
        device="cpu",
    )
    fields.update(overrides)
    return VideoUpscaleJob(**fields)


def make_gate_upscaler(
    tmp_path: Path, *, gmfss_engine: object, onnx_video_engine: object
) -> VideoUpscaler:
    settings = Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        BUILTIN_ONNX_DIR=str(tmp_path / "builtin-onnx"),
    )
    return VideoUpscaler(
        settings,
        FakeNcnnEngine(),  # type: ignore[arg-type]
        FakeMediaTools(),  # type: ignore[arg-type]
        gmfss_engine=gmfss_engine,  # type: ignore[arg-type]
        onnx_video_engine=onnx_video_engine,  # type: ignore[arg-type]
        model_registry=ModelRegistry(settings),
        devices=FakeDevicesService(),  # type: ignore[arg-type]
    )


def make_fused_upscaler(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> VideoUpscaler:
    """VideoUpscaler wired with a REAL GmfssEngine + REAL OnnxVideoUpscaler, each
    with its session-creation seam monkeypatched to a deterministic fake."""
    gmfss = GmfssEngine(settings, GpuSessionCoordinator())
    monkeypatch.setattr(gmfss, "_create_sessions", fake_gmfss_sessions)
    onnx_video = OnnxVideoUpscaler(
        settings, ModelRegistry(settings), DevicesService(settings), GpuSessionCoordinator()
    )
    monkeypatch.setattr(onnx_video, "_create_session", lambda model_path, device: Double2xUint8Session())
    return VideoUpscaler(
        settings,
        FakeNcnnEngine(),  # type: ignore[arg-type]
        FakeMediaTools(),  # type: ignore[arg-type]
        gmfss_engine=gmfss,
        onnx_video_engine=onnx_video,
        model_registry=ModelRegistry(settings),
        devices=DevicesService(settings),
    )


# ---------------------------------------------------------------------------
# Gate: fused path chosen ONLY for gmfss + in-process onnx backend + interp.
# ---------------------------------------------------------------------------


async def test_gate_selects_fused_for_gmfss_with_onnx_backend(tmp_path: Path) -> None:
    upscaler = make_gate_upscaler(tmp_path, gmfss_engine=object(), onnx_video_engine=FakeOnnxVideoEngine())
    job = make_video_job(tmp_path / "clip.mp4", interp_engine="gmfss", backend="onnx", fps_multiplier=2)

    assert await upscaler._should_fuse_interpolate_upscale(job, 2, None) is True


async def test_gate_selects_two_pass_for_rife_even_with_onnx_backend(tmp_path: Path) -> None:
    upscaler = make_gate_upscaler(tmp_path, gmfss_engine=object(), onnx_video_engine=FakeOnnxVideoEngine())
    job = make_video_job(tmp_path / "clip.mp4", interp_engine="rife", backend="onnx", fps_multiplier=2)

    assert await upscaler._should_fuse_interpolate_upscale(job, 2, None) is False


async def test_gate_selects_two_pass_for_gmfss_with_ncnn_backend(tmp_path: Path) -> None:
    upscaler = make_gate_upscaler(tmp_path, gmfss_engine=object(), onnx_video_engine=FakeOnnxVideoEngine())
    job = make_video_job(tmp_path / "clip.mp4", interp_engine="gmfss", backend="ncnn", fps_multiplier=2)

    assert await upscaler._should_fuse_interpolate_upscale(job, 2, None) is False


async def test_gate_selects_two_pass_for_gmfss_onnx_without_interpolation(tmp_path: Path) -> None:
    # No interpolation requested (multiplier 1, no target_fps): nothing to fuse.
    upscaler = make_gate_upscaler(tmp_path, gmfss_engine=object(), onnx_video_engine=FakeOnnxVideoEngine())
    job = make_video_job(tmp_path / "clip.mp4", interp_engine="gmfss", backend="onnx", fps_multiplier=1)

    assert await upscaler._should_fuse_interpolate_upscale(job, 1, None) is False


async def test_gate_selects_two_pass_when_no_onnx_video_engine(tmp_path: Path) -> None:
    upscaler = make_gate_upscaler(tmp_path, gmfss_engine=object(), onnx_video_engine=None)
    job = make_video_job(tmp_path / "clip.mp4", interp_engine="gmfss", backend="onnx", fps_multiplier=2)

    assert await upscaler._should_fuse_interpolate_upscale(job, 2, None) is False


async def test_interpolate_and_upscale_routes_to_fused_when_gate_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_gate_upscaler(tmp_path, gmfss_engine=object(), onnx_video_engine=FakeOnnxVideoEngine())
    job = make_video_job(tmp_path / "clip.mp4", interp_engine="gmfss", backend="onnx", fps_multiplier=2)
    frames_in = tmp_path / "frames-in"
    frames_in.mkdir()
    frames_out = tmp_path / "frames-out"
    called = {"fused": False, "two_pass": False}

    async def fake_fused(*args: object, **kwargs: object) -> str:
        called["fused"] = True
        return "48/1"

    async def fake_two_pass(*args: object, **kwargs: object) -> tuple[Path, str]:
        called["two_pass"] = True
        return frames_in, "48/1"

    monkeypatch.setattr(upscaler, "_run_fused_interpolate_upscale", fake_fused)
    monkeypatch.setattr(upscaler, "_maybe_interpolate", fake_two_pass)

    encode_dir, encode_fps = await upscaler._interpolate_and_upscale(
        job, frames_in, frames_out, "24/1", 2, tmp_path / "out.mp4", None, []
    )

    assert called["fused"] is True
    assert called["two_pass"] is False
    assert encode_dir == frames_out
    assert encode_fps == "48/1"


# ---------------------------------------------------------------------------
# Parity: fused output is BYTE-IDENTICAL to the two-pass output (the phase's
# zero-tolerance quality gate). Same deterministic fakes drive both paths.
# ---------------------------------------------------------------------------


async def test_fused_output_is_pixel_identical_to_two_pass_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_combined_settings(tmp_path)
    touch_builtin_onnx(settings, "realesr-animevideov3-x4-uint8.onnx")
    upscaler = make_fused_upscaler(settings, monkeypatch)
    job = make_video_job(tmp_path / "clip.mp4", interp_engine="gmfss", backend="onnx", fps_multiplier=2, device="cpu")

    frames_in = tmp_path / "frames-in"
    write_source_frames(frames_in, 3)  # 3 source -> 6 output at 2x
    source_frame_count = 3
    fps = "24/1"

    # --- fused path: writes upscaled frames directly to frames-out-fused ---
    frames_out_fused = tmp_path / "frames-out-fused"
    encode_fps = await upscaler._run_fused_interpolate_upscale(
        job, frames_in, frames_out_fused, fps, 2, None
    )

    # --- two-pass path: GMFSS run -> frames-interp, then ONNX run_frames_builtin ---
    frames_interp = tmp_path / "frames-interp"
    await upscaler.gmfss_engine.run(frames_in, frames_interp, source_frame_count, 2, device="cpu")
    frames_out_two_pass = tmp_path / "frames-out-two-pass"
    await upscaler.onnx_video_engine.run_frames_builtin(
        frames_interp, frames_out_two_pass, job.model_name, "cpu"
    )

    fused_files = sorted(frames_out_fused.glob("*.png"))
    two_pass_files = sorted(frames_out_two_pass.glob("*.png"))

    assert encode_fps == "48/1"
    assert [p.name for p in fused_files] == [p.name for p in two_pass_files]
    assert len(fused_files) == 6
    for fused_path, two_pass_path in zip(fused_files, two_pass_files):
        # Byte-exact, NOT np.allclose: same operation, only the disk hop removed.
        assert fused_path.read_bytes() == two_pass_path.read_bytes(), fused_path.name


# ---------------------------------------------------------------------------
# Cancellation of the fused path (documented cancel-cleanup race regression).
# ---------------------------------------------------------------------------


async def test_fused_path_cancel_waits_for_worker_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same contract as GmfssEngine.run / OnnxVideoUpscaler.run_frames_builtin: by
    # the time the cancel propagates, the worker thread has ALREADY finished, so
    # the caller's work-dir rmtree can't race a straggler PNG write.
    settings = make_combined_settings(tmp_path)
    touch_builtin_onnx(settings, "realesr-animevideov3-x4-uint8.onnx")
    upscaler = make_fused_upscaler(settings, monkeypatch)
    job = make_video_job(tmp_path / "clip.mp4", interp_engine="gmfss", backend="onnx", fps_multiplier=2, device="cpu")
    frames_in = tmp_path / "frames-in"
    write_source_frames(frames_in, 4)
    worker_finished = threading.Event()

    def slow_worker(*args: object, **kwargs: object) -> None:
        cancel_event = args[-1] if args else kwargs["cancel_event"]
        assert isinstance(cancel_event, threading.Event)
        cancel_event.wait(timeout=10)
        time.sleep(0.2)  # simulates a non-interruptible save in flight
        worker_finished.set()

    monkeypatch.setattr(upscaler, "_run_fused_frames_blocking", slow_worker)

    task = asyncio.create_task(
        upscaler._run_fused_interpolate_upscale(job, frames_in, tmp_path / "frames-out", "24/1", 2, None)
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert worker_finished.is_set(), "fused path propagated cancel before the worker thread finished"


def test_fused_blocking_stops_immediately_when_cancel_event_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pre-set cancel must stop the blocking generator loop with zero frames
    # written and NO leaked thread (the generator is pull-based, so closing it
    # unwinds it without a background thread outliving the call).
    settings = make_combined_settings(tmp_path)
    touch_builtin_onnx(settings, "realesr-animevideov3-x4-uint8.onnx")
    upscaler = make_fused_upscaler(settings, monkeypatch)
    job = make_video_job(tmp_path / "clip.mp4", interp_engine="gmfss", backend="onnx", fps_multiplier=2, device="cpu")
    frames_in = tmp_path / "frames-in"
    write_source_frames(frames_in, 6)
    frames_out = tmp_path / "frames-out"
    frames_out.mkdir()
    cancel_event = threading.Event()
    cancel_event.set()

    threads_before = set(threading.enumerate())
    upscaler._run_fused_frames_blocking(job, frames_in, frames_out, 6, 2, None, "cpu", cancel_event)
    threads_after = set(threading.enumerate())

    assert list(frames_out.glob("*.png")) == []
    assert threads_after <= threads_before, "fused blocking loop left a thread running after a pre-set cancel"
