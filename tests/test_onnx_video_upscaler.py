from __future__ import annotations

import asyncio
import ctypes
import logging
import queue
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services.devices_service import DevicesService
from app.services.engines.frame_workers import (
    FrameReadbackRing,
    derive_readback_ring_capacity,
    drain_queue,
    put_until_cancelled,
)
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler, should_tile_frame
from app.services.gpu_session_coordinator import GpuSessionCoordinator
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
    return OnnxVideoUpscaler(settings, ModelRegistry(settings), DevicesService(settings), GpuSessionCoordinator())


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
# fp16 model-file selection: GPU prefers the fp16 export when present + enabled
# ---------------------------------------------------------------------------

from app.services.backend_registry import get_builtin_onnx_model  # noqa: E402

_X4 = get_builtin_onnx_model("realesr-animevideov3-x4")


def test_select_model_file_prefers_fp16_on_gpu_when_present(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, _X4.filename)
    touch_builtin_onnx(engine.settings, _X4.fp16_filename)
    assert engine._select_model_file(_X4, "dml:0").name == _X4.fp16_filename


def test_select_model_file_uses_fp32_on_cpu_even_if_fp16_present(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, _X4.filename)
    touch_builtin_onnx(engine.settings, _X4.fp16_filename)
    assert engine._select_model_file(_X4, "cpu").name == _X4.filename


def test_select_model_file_falls_back_to_fp32_when_fp16_absent(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, _X4.filename)  # no fp16 sibling
    assert engine._select_model_file(_X4, "dml:0").name == _X4.filename


def test_select_model_file_honors_prefer_fp16_false(tmp_path: Path) -> None:
    engine = make_engine(tmp_path, ONNX_PREFER_FP16=False)
    touch_builtin_onnx(engine.settings, _X4.filename)
    touch_builtin_onnx(engine.settings, _X4.fp16_filename)
    assert engine._select_model_file(_X4, "dml:0").name == _X4.filename


def test_fp16_filename_matches_export_convention() -> None:
    assert _X4.fp16_filename == _X4.filename.replace(".onnx", "-fp16.onnx")


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


async def test_run_frames_streaming_writes_every_frame_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())

    frames_in = tmp_path / "frames-in"
    write_frames(frames_in, count=6, height=4, width=6)

    received: list[tuple[int, int, int]] = []
    first_pixels: list[int] = []

    def write_fn(frame_hwc) -> None:
        received.append(frame_hwc.shape)
        first_pixels.append(int(frame_hwc[0, 0, 0]))

    count = await engine.run_frames_streaming(frames_in, "realesr-animevideov3-x4", "cpu", write_fn)

    assert count == 6
    assert len(received) == 6
    assert all(shape == (8, 12, 3) for shape in received)  # 4x6 doubled


async def test_ordered_writer_reorders_by_frame_index() -> None:
    import queue as _queue
    from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler as _E

    save_q: _queue.Queue = _queue.Queue()
    # Deliver frames OUT of order; the writer must emit them in index order.
    for name, value in [("00000002.png", 2), ("00000001.png", 1), ("00000003.png", 3)]:
        save_q.put((name, np.full((1, 1, 1, 1), value, dtype=np.uint8)))
    save_q.put(None)
    emitted: list[int] = []
    _E._ordered_writer_loop(save_q, lambda f: emitted.append(int(f[0, 0, 0])), 3, [], threading.Event())
    assert emitted == [1, 2, 3]


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


from app.services.engines.frame_workers import derive_queue_maxsize


def test_derive_queue_maxsize_budget_decides_between_floor_and_ceiling() -> None:
    # 48MB/frame, presupuesto 150MB -> 150//48 = 3, estrictamente entre piso 2 y techo 4.
    assert derive_queue_maxsize(48 * 1024 * 1024, 150 * 1024 * 1024, 2, 4) == 3


def test_derive_queue_maxsize_floors_and_ceils() -> None:
    assert derive_queue_maxsize(48 * 1024 * 1024, 1 * 1024 * 1024, 5, 10) == 5  # piso
    assert derive_queue_maxsize(1024, 1024 * 1024 * 1024, 2, 16) == 16  # techo


def test_derive_queue_maxsize_nonpositive_frame_bytes_returns_ceiling() -> None:
    # Sin tamaño de frame conocido no hay presupuesto que aplicar: techo (el
    # caso "sin frames" que _save_queue_maxsize ya resolvía con el default).
    assert derive_queue_maxsize(0, 100, 2, 8) == 8
    assert derive_queue_maxsize(-1, 100, 2, 8) == 8


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


# ---------------------------------------------------------------------------
# FrameReadbackRing - anillo de K buffers CPU preasignados para el readback.
# K debe SUPERAR los frames en vuelo downstream: un buffer reusado mientras el
# frame anterior sigue encolado corrompe frames (regla dura, medida).
# ---------------------------------------------------------------------------


def test_readback_ring_rejects_capacity_below_two() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        FrameReadbackRing(1)


def test_readback_ring_reuses_the_same_buffers_after_a_full_cycle() -> None:
    ring = FrameReadbackRing(3)
    frames = [np.full((1, 4, 6, 3), value, dtype=np.uint8) for value in range(5)]
    outs = [ring.copy_in(frame) for frame in frames]
    # Dentro del ciclo son objetos distintos; al dar la vuelta se reusa el MISMO
    # objeto (esa identidad estable es la ausencia de allocación por frame).
    assert len({id(out) for out in outs[:3]}) == 3
    assert outs[3] is outs[0]
    assert outs[4] is outs[1]


def test_readback_ring_copy_does_not_clobber_in_flight_buffers() -> None:
    ring = FrameReadbackRing(3)
    first = ring.copy_in(np.full((1, 2, 2, 3), 7, dtype=np.uint8))
    ring.copy_in(np.full((1, 2, 2, 3), 9, dtype=np.uint8))
    assert np.array_equal(first, np.full((1, 2, 2, 3), 7, dtype=np.uint8))


def test_readback_ring_returns_its_buffer_not_the_input() -> None:
    ring = FrameReadbackRing(2)
    source = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    out = ring.copy_in(source)
    assert out is not source
    source[...] = 255
    assert out[0, 0, 0, 0] == 0  # el buffer no comparte memoria con el input


def test_readback_ring_reallocates_on_shape_change() -> None:
    ring = FrameReadbackRing(2)
    small = ring.copy_in(np.zeros((1, 2, 2, 3), dtype=np.uint8))
    big = ring.copy_in(np.ones((1, 4, 4, 3), dtype=np.uint8))
    assert small.shape == (1, 2, 2, 3)  # el buffer viejo queda intacto
    assert big.shape == (1, 4, 4, 3)
    # El anillo nuevo rota normalmente sobre el shape nuevo.
    big2 = ring.copy_in(np.ones((1, 4, 4, 3), dtype=np.uint8))
    big3 = ring.copy_in(np.ones((1, 4, 4, 3), dtype=np.uint8))
    assert big2 is not big
    assert big3 is big


def test_derive_readback_ring_capacity_exceeds_frames_in_flight() -> None:
    # Streaming: save_q(4) + writer(1); frames: save_q(4) + 2 savers.
    assert derive_readback_ring_capacity(4, 1) == 7
    assert derive_readback_ring_capacity(4, 2) == 8
    for slots, consumers in [(2, 1), (4, 2), (16, 1)]:
        frames_in_flight = 1 + slots + consumers  # 1 recién producido + cola + consumidores
        assert derive_readback_ring_capacity(slots, consumers) > frames_in_flight


def test_upscale_one_returns_ring_buffers_and_cycles_identity(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    session = Double2xUint8Session()
    ring = FrameReadbackRing(2)
    frame = np.random.default_rng(4).integers(0, 256, (1, 4, 6, 3), dtype=np.uint8)

    out1, _ = engine._upscale_one(session, frame, "cpu", False, ring)
    out2, _ = engine._upscale_one(session, frame, "cpu", False, ring)
    out3, _ = engine._upscale_one(session, frame, "cpu", False, ring)

    assert out1 is not out2
    assert out3 is out1  # K=2: el tercer frame reusa el primer buffer
    expected = np.repeat(np.repeat(frame, 2, axis=1), 2, axis=2)
    assert np.array_equal(out2, expected)


def test_upscale_one_without_ring_keeps_per_frame_output(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    session = Double2xUint8Session()
    frame = np.random.default_rng(5).integers(0, 256, (1, 4, 6, 3), dtype=np.uint8)
    out1, _ = engine._upscale_one(session, frame, "cpu")
    out2, _ = engine._upscale_one(session, frame, "cpu")
    assert out1 is not out2


def test_upscale_one_tiled_path_bypasses_the_ring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # El tiled ya aloca su canvas al blendear: el anillo solo cubre el readback
    # whole-frame, así que la salida tiled debe pasar tal cual.
    engine = make_engine(tmp_path, ONNX_WHOLE_FRAME_MAX_PIXELS=10)
    ring = FrameReadbackRing(2)
    sentinel = np.zeros((1, 8, 12, 3), dtype=np.uint8)
    monkeypatch.setattr(engine, "_infer_tiled", lambda s, f, d: sentinel)

    out, _ = engine._upscale_one(None, np.zeros((1, 8, 12, 3), dtype=np.uint8), "cpu", False, ring)

    assert out is sentinel


async def test_run_frames_builtin_ring_cycling_does_not_corrupt_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Con 1 saver el anillo queda en K=5 (save_q 2 + saver 1 + 2); 12 frames lo
    # hacen dar varias vueltas. La corrupción por reuso prematuro apareceria
    # como píxeles de OTRO frame en la salida, así que se verifica el contenido
    # exacto de cada frame, no solo el conteo.
    engine = make_engine(tmp_path, ONNX_VIDEO_LOAD_THREADS=1, ONNX_VIDEO_SAVE_THREADS=1)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_frames(frames_in, count=12, height=4, width=6)

    await engine.run_frames_builtin(frames_in, frames_out, "realesr-animevideov3-x4", "cpu")

    for path in sorted(frames_in.glob("*.png")):
        with Image.open(path) as image:
            source = np.asarray(image)
        expected = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
        with Image.open(frames_out / path.name) as image:
            produced = np.asarray(image)
        assert np.array_equal(produced, expected), path.name


async def test_run_frames_streaming_ring_cycling_keeps_frames_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Streaming: K=7 (save_q 4 + writer 1 + 2); 12 frames dan más de una vuelta.
    # write_frame recibe el buffer del anillo y debe consumirlo antes de
    # retornar (contrato del sink) — de ahí el .copy() al capturar.
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())
    frames_in = tmp_path / "frames-in"
    write_frames(frames_in, count=12, height=4, width=6)

    received: list[np.ndarray] = []
    count = await engine.run_frames_streaming(
        frames_in, "realesr-animevideov3-x4", "cpu", lambda frame: received.append(frame.copy())
    )

    assert count == 12
    assert len(received) == 12
    for index, path in enumerate(sorted(frames_in.glob("*.png"))):
        with Image.open(path) as image:
            source = np.asarray(image)
        expected = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
        assert np.array_equal(received[index], expected), path.name


def test_build_frame_upscaler_ring_capacity_cycles_buffers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # El modelo x2 coincide con el fake 2x: el anillo dimensiona sus buffers con
    # la escala declarada del modelo, y una salida de otra escala no entra en él.
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x2-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())
    upscale = engine.build_frame_upscaler("realesr-animevideov3-x2", "cpu", readback_ring_capacity=2)
    frame = np.random.default_rng(6).integers(0, 256, (1, 4, 6, 3), dtype=np.uint8)

    out1, out2, out3 = upscale(frame), upscale(frame), upscale(frame)

    assert out1 is not out2
    assert out3 is out1
    assert np.array_equal(out2, np.repeat(np.repeat(frame, 2, axis=1), 2, axis=2))


def test_build_frame_upscaler_defaults_to_no_ring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Sin capacidad explícita no hay anillo: el closure no conoce las colas del
    # FramePipeline del caller y un K menor a los frames en vuelo corrompe.
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())
    upscale = engine.build_frame_upscaler("realesr-animevideov3-x4", "cpu")
    frame = np.zeros((1, 4, 6, 3), dtype=np.uint8)

    assert upscale(frame) is not upscale(frame)


# ---------------------------------------------------------------------------
# Readback directo al anillo: la salida de ORT se bindea al buffer preasignado
# (bind_output con buffer_ptr) en vez de copy_outputs_to_cpu() + np.copyto.
# Sin GPU: la ruta DML se prueba con un io_binding falso y la ruta real con una
# sesión CPUExecutionProvider sobre un grafo Resize NHWC uint8.
# ---------------------------------------------------------------------------


def _doubled(frame: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(frame, 2, axis=1), 2, axis=2)


def _write_to_address(address: int, array: np.ndarray) -> None:
    target = np.ctypeslib.as_array((ctypes.c_uint8 * array.nbytes).from_address(address))
    target[:] = np.ascontiguousarray(array).reshape(-1).view(np.uint8)


class _FakeOrtValue:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array


class _FakeOrtModule:
    class OrtValue:
        @staticmethod
        def ortvalue_from_numpy(array: np.ndarray, device: str, device_id: int) -> _FakeOrtValue:
            return _FakeOrtValue(array)


class _RecordingIoBinding:
    def __init__(self, session: "IoBindingDouble2xSession") -> None:
        self._session = session
        self.input: np.ndarray | None = None
        self.output_buffer_ptr: int | None = None

    def bind_ortvalue_input(self, name: str, value: _FakeOrtValue) -> None:
        self.input = value.array

    def bind_cpu_input(self, name: str, array: np.ndarray) -> None:
        self.input = array

    def bind_output(
        self,
        name: str,
        device_type: str = "cpu",
        device_id: int = 0,
        element_type: Any = None,
        shape: Any = None,
        buffer_ptr: int | None = None,
    ) -> None:
        if device_type == "cpu" and self._session.fail_cpu_output_bind:
            raise RuntimeError("binding the output to CPU memory is not supported")
        self._session.bound_outputs.append((device_type, buffer_ptr))
        self.output_buffer_ptr = buffer_ptr

    def copy_outputs_to_cpu(self) -> list[np.ndarray]:
        self._session.copy_outputs_calls += 1
        assert self.input is not None
        return [_doubled(self.input)]


class IoBindingDouble2xSession(Double2xUint8Session):
    """Double2xUint8Session con io_binding(): si la salida está bindeada a un
    buffer CPU, run_with_iobinding escribe el resultado en esa dirección."""

    def __init__(self, fail_cpu_output_bind: bool = False, fail_cpu_output_run: bool = False) -> None:
        super().__init__()
        self.fail_cpu_output_bind = fail_cpu_output_bind
        self.fail_cpu_output_run = fail_cpu_output_run
        self.bound_outputs: list[tuple[str, int | None]] = []
        self.copy_outputs_calls = 0
        self.plain_run_calls = 0

    def io_binding(self) -> _RecordingIoBinding:
        return _RecordingIoBinding(self)

    def run_with_iobinding(self, binding: _RecordingIoBinding) -> None:
        assert binding.input is not None
        if binding.output_buffer_ptr is not None and self.fail_cpu_output_run:
            # Como ORT real: el buffer bindeado se rechaza recién al asignar la
            # salida del último nodo, después de ejecutar todo el grafo.
            raise RuntimeError("OrtValue shape verification failed")
        if binding.output_buffer_ptr is not None:
            _write_to_address(binding.output_buffer_ptr, _doubled(binding.input))

    def run(self, output_names: list[str], input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.plain_run_calls += 1
        return super().run(output_names, input_feed)


def make_resize_session(scale: int, trailing_identity: bool = False) -> Any:
    ort = pytest.importorskip("onnxruntime")
    onnx_helper = pytest.importorskip("onnx.helper")
    from onnx import TensorProto

    image = onnx_helper.make_tensor_value_info("image", TensorProto.UINT8, [1, None, None, 3])
    upscaled = onnx_helper.make_tensor_value_info("upscaled", TensorProto.UINT8, [1, None, None, 3])
    scales = onnx_helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, float(scale), float(scale), 1.0])
    resized = "resized" if trailing_identity else "upscaled"
    nodes = [onnx_helper.make_node("Resize", ["image", "", "scales"], [resized], mode="nearest")]
    if trailing_identity:
        # Con más de un nodo, ORT rechaza el buffer bindeado recién después de ejecutar el grafo.
        nodes.append(onnx_helper.make_node("Identity", [resized], ["upscaled"]))
    graph = onnx_helper.make_graph(nodes, "resize-nhwc-uint8", [image], [upscaled], initializer=[scales])
    model = onnx_helper.make_model(graph, opset_imports=[onnx_helper.make_opsetid("", 17)])
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def test_readback_ring_next_buffer_rotates_without_copying() -> None:
    ring = FrameReadbackRing(2)
    first = ring.next_buffer((1, 2, 2, 3), np.uint8)
    first[...] = 5
    second = ring.next_buffer((1, 2, 2, 3), np.uint8)
    third = ring.next_buffer((1, 2, 2, 3), np.uint8)

    assert second is not first
    assert third is first
    assert third[0, 0, 0, 0] == 5  # ni copia ni limpieza: escribe quien lo pide
    assert first.dtype == np.uint8
    assert first.flags.c_contiguous


def test_upscale_one_binds_the_dml_output_into_the_ring_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrtModule)
    engine = make_engine(tmp_path)
    session = IoBindingDouble2xSession()
    ring = FrameReadbackRing(2)
    frames = [np.random.default_rng(seed).integers(0, 256, (1, 4, 6, 3), dtype=np.uint8) for seed in (1, 2)]

    out1, _ = engine._upscale_one(session, frames[0], "dml:0", False, ring, 2)
    out2, _ = engine._upscale_one(session, frames[1], "dml:0", False, ring, 2)

    assert session.copy_outputs_calls == 0
    assert session.plain_run_calls == 0
    assert session.bound_outputs == [("cpu", out1.ctypes.data), ("cpu", out2.ctypes.data)]
    assert out1 is not out2
    assert np.array_equal(out1, _doubled(frames[0]))
    assert np.array_equal(out2, _doubled(frames[1]))


def test_upscale_one_ring_bind_on_a_real_cpu_session_is_bit_exact_and_rotates(tmp_path: Path) -> None:
    session = make_resize_session(4)
    engine = make_engine(tmp_path)
    capacity = 3
    ring = FrameReadbackRing(capacity)
    rng = np.random.default_rng(11)
    outs = []
    for _ in range(capacity + 2):
        frame = rng.integers(0, 256, (1, 5, 7, 3), dtype=np.uint8)
        out, _ = engine._upscale_one(session, frame, "cpu", False, ring, 4)
        assert np.array_equal(out, session.run(None, {"image": frame})[0])
        outs.append(out)

    assert engine._ring_bind_warned is False  # salió por el bind, no por el fallback
    assert len({id(out) for out in outs[:capacity]}) == capacity
    assert outs[capacity] is outs[0]
    assert outs[capacity + 1] is outs[1]


def test_upscale_one_ring_bind_failure_falls_back_to_the_dml_copy_path_and_logs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrtModule)
    engine = make_engine(tmp_path)
    session = IoBindingDouble2xSession(fail_cpu_output_bind=True)
    ring = FrameReadbackRing(2)
    frames = [np.random.default_rng(seed).integers(0, 256, (1, 4, 6, 3), dtype=np.uint8) for seed in (3, 4, 5)]

    outs = []
    with caplog.at_level(logging.WARNING, logger="app.services.engines.onnx_video_upscaler"):
        for frame in frames:
            out, _ = engine._upscale_one(session, frame, "dml:0", False, ring, 2)
            assert np.array_equal(out, _doubled(frame))  # antes de que el anillo lo reuse
            outs.append(out)

    assert session.copy_outputs_calls == 3  # camino actual: salida en "dml" + copy_outputs_to_cpu
    assert session.plain_run_calls == 0  # nunca al plain-run
    assert outs[2] is outs[0]  # el fallback sigue escribiendo en el anillo
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_upscale_one_ring_bind_rejected_at_run_time_is_not_retried_on_later_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrtModule)
    engine = make_engine(tmp_path)
    session = IoBindingDouble2xSession(fail_cpu_output_run=True)
    ring = FrameReadbackRing(2)
    frames = [np.random.default_rng(seed).integers(0, 256, (1, 4, 6, 3), dtype=np.uint8) for seed in (6, 7, 8)]

    with caplog.at_level(logging.WARNING, logger="app.services.engines.onnx_video_upscaler"):
        for frame in frames:
            out, _ = engine._upscale_one(session, frame, "dml:0", False, ring, 2)
            assert np.array_equal(out, _doubled(frame))

    cpu_binds = [bound for bound in session.bound_outputs if bound[0] == "cpu"]
    assert len(cpu_binds) == 1  # un solo intento: cada reintento costaría casi una inferencia entera
    assert session.copy_outputs_calls == 3
    assert session.plain_run_calls == 0
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_ring_bind_disabled_for_one_session_still_binds_for_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrtModule)
    engine = make_engine(tmp_path)
    rejecting = IoBindingDouble2xSession(fail_cpu_output_run=True)
    accepting = IoBindingDouble2xSession()
    frame = np.random.default_rng(9).integers(0, 256, (1, 4, 6, 3), dtype=np.uint8)

    engine._upscale_one(rejecting, frame, "dml:0", False, FrameReadbackRing(2), 2)
    out, _ = engine._upscale_one(accepting, frame, "dml:0", False, FrameReadbackRing(2), 2)

    assert accepting.bound_outputs == [("cpu", out.ctypes.data)]
    assert accepting.copy_outputs_calls == 0


def test_upscale_one_ring_with_a_wrong_declared_scale_returns_the_real_output_outside_the_ring(
    tmp_path: Path,
) -> None:
    # ORT rechaza el buffer de shape equivocado ("OrtValue shape verification
    # failed") en vez de escribir fuera de él; el frame sale por el fallback y,
    # como no entra en el anillo, se devuelve el array propio del run.
    session = make_resize_session(4)
    engine = make_engine(tmp_path)
    ring = FrameReadbackRing(2)
    frame = np.random.default_rng(13).integers(0, 256, (1, 5, 7, 3), dtype=np.uint8)

    out, _ = engine._upscale_one(session, frame, "cpu", False, ring, 2)

    assert engine._ring_bind_warned is True
    assert np.array_equal(out, session.run(None, {"image": frame})[0])
    assert out.shape == (1, 20, 28, 3)


class _CountingSession:
    def __init__(self, session: Any) -> None:
        self._session = session
        self.io_binding_calls = 0
        self.run_calls = 0

    def get_inputs(self) -> Any:
        return self._session.get_inputs()

    def get_outputs(self) -> Any:
        return self._session.get_outputs()

    def io_binding(self) -> Any:
        self.io_binding_calls += 1
        return self._session.io_binding()

    def run_with_iobinding(self, binding: Any) -> None:
        self._session.run_with_iobinding(binding)

    def run(self, output_names: Any, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.run_calls += 1
        return self._session.run(output_names, input_feed)


def test_ring_bind_rejected_by_a_real_multi_node_graph_is_tried_once_per_session(tmp_path: Path) -> None:
    real = make_resize_session(4, trailing_identity=True)
    session = _CountingSession(real)
    engine = make_engine(tmp_path)
    ring = FrameReadbackRing(2)
    rng = np.random.default_rng(14)

    for _ in range(3):
        frame = rng.integers(0, 256, (1, 5, 7, 3), dtype=np.uint8)
        out, _ = engine._upscale_one(session, frame, "cpu", False, ring, 2)
        assert np.array_equal(out, real.run(None, {"image": frame})[0])

    assert session.io_binding_calls == 1
    assert session.run_calls == 3


async def test_run_frames_streaming_binds_into_the_ring_with_the_model_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x2-uint8.onnx")
    session = make_resize_session(2)
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: session)
    frames_in = tmp_path / "frames-in"
    write_frames(frames_in, count=9, height=4, width=6)
    received: list[np.ndarray] = []

    await engine.run_frames_streaming(
        frames_in, "realesr-animevideov3-x2", "cpu", lambda frame: received.append(frame.copy())
    )

    assert engine._ring_bind_warned is False
    for index, path in enumerate(sorted(frames_in.glob("*.png"))):
        with Image.open(path) as image:
            source = np.asarray(image)
        assert np.array_equal(received[index], np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)), path.name


async def test_run_frames_builtin_binds_into_the_ring_with_the_model_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(tmp_path, ONNX_VIDEO_LOAD_THREADS=1, ONNX_VIDEO_SAVE_THREADS=1)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x2-uint8.onnx")
    session = make_resize_session(2)
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: session)
    frames_in = tmp_path / "frames-in"
    frames_out = tmp_path / "frames-out"
    write_frames(frames_in, count=9, height=4, width=6)

    await engine.run_frames_builtin(frames_in, frames_out, "realesr-animevideov3-x2", "cpu")

    assert engine._ring_bind_warned is False
    for path in sorted(frames_in.glob("*.png")):
        with Image.open(path) as image:
            source = np.asarray(image)
        with Image.open(frames_out / path.name) as image:
            produced = np.asarray(image)
        assert np.array_equal(produced, np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)), path.name


def test_build_frame_upscaler_binds_into_the_ring_with_the_model_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x2-uint8.onnx")
    session = make_resize_session(2)
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: session)
    upscale = engine.build_frame_upscaler("realesr-animevideov3-x2", "cpu", readback_ring_capacity=2)
    frame = np.random.default_rng(12).integers(0, 256, (1, 4, 6, 3), dtype=np.uint8)

    out1, out2, out3 = upscale(frame), upscale(frame), upscale(frame)

    assert engine._ring_bind_warned is False
    assert out1 is not out2
    assert out3 is out1
    assert np.array_equal(out2, _doubled(frame))


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
# GpuSessionCoordinator wiring (Fase 1 Task 4) - release_device evicts every
# cache entry for that device regardless of model (cache is keyed by
# (model_path, device), a single device can have several model entries,
# unlike the flat per-device caches in Tasks 2-3), and acquire() runs before
# any session is built. Same pattern as
# GmfssEngine/AudioSrRestorer/ApolloRestorer/OnnxUpscaler.
# ---------------------------------------------------------------------------


def test_release_device_clears_all_cached_sessions_for_that_device(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    engine._session_cache[("model-a.onnx", "dml:0")] = "fake-a"
    engine._session_cache[("model-b.onnx", "dml:0")] = "fake-b"
    engine._session_cache[("model-a.onnx", "dml:1")] = "fake-a-1"

    engine.release_device("dml:0")

    assert ("model-a.onnx", "dml:0") not in engine._session_cache
    assert ("model-b.onnx", "dml:0") not in engine._session_cache
    assert ("model-a.onnx", "dml:1") in engine._session_cache


def test_release_device_on_empty_cache_is_a_noop(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)

    engine.release_device("dml:0")  # no debe lanzar


def test_get_session_calls_coordinator_acquire_before_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    gpu_coordinator = GpuSessionCoordinator()
    engine = OnnxVideoUpscaler(settings, ModelRegistry(settings), DevicesService(settings), gpu_coordinator)
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(gpu_coordinator, "acquire", lambda device, owner: calls.append((device, owner)))
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: "fake-session")

    engine._get_session("/models/x4.onnx", "dml:0")

    assert calls == [("dml:0", engine)]


def test_build_frame_upscaler_returns_working_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())

    upscale = engine.build_frame_upscaler("realesr-animevideov3-x4", "cpu")
    frame = np.random.default_rng(3).integers(0, 256, (1, 4, 6, 3), dtype=np.uint8)
    out = upscale(frame)

    assert out.shape == (1, 8, 12, 3)
    assert out.dtype == np.uint8


def test_build_frame_upscaler_raises_for_unconfigured_model(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)
    with pytest.raises(RuntimeError, match="No ONNX export configured"):
        engine.build_frame_upscaler("does-not-exist", "cpu")


def test_build_frame_upscaler_raises_when_model_file_missing(tmp_path: Path) -> None:
    engine = make_engine(tmp_path)  # sin touch_builtin_onnx -> archivo ausente
    with pytest.raises(RuntimeError, match="ONNX model file not found"):
        engine.build_frame_upscaler("realesr-animevideov3-x4", "cpu")


def test_build_frame_upscaler_sticks_to_tiling_after_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mismo contrato sticky que _infer_loop: un OOM whole-frame degrada el RESTO
    # del run a tiling (el estado vive en el closure, un solo intento whole).
    engine = make_engine(tmp_path)
    touch_builtin_onnx(engine.settings, "realesr-animevideov3-x4-uint8.onnx")
    monkeypatch.setattr(engine, "_create_session", lambda model_path, device: Double2xUint8Session())
    calls = {"whole": 0, "tiled": 0}

    def whole_frame_oom(s, f, d):
        calls["whole"] += 1
        raise RuntimeError("Failed to allocate memory: out of memory (D3D12)")

    monkeypatch.setattr(engine, "_infer_frame", whole_frame_oom)
    monkeypatch.setattr(
        engine, "_infer_tiled", lambda s, f, d: (calls.__setitem__("tiled", calls["tiled"] + 1), f)[1]
    )

    upscale = engine.build_frame_upscaler("realesr-animevideov3-x4", "cpu")
    frame = np.zeros((1, 4, 6, 3), dtype=np.uint8)
    upscale(frame)
    upscale(frame)

    assert calls == {"whole": 1, "tiled": 2}  # el 2o frame va directo a tiling


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
    engine = OnnxVideoUpscaler(
        settings, ModelRegistry(settings), DevicesService(settings), GpuSessionCoordinator()
    )
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
    assert put_until_cancelled(q, "item", cancel, timeout=0.01) is True
    assert q.get_nowait() == "item"


def test_put_until_cancelled_returns_false_without_enqueue_when_cancelled_on_full_queue() -> None:
    q: queue.Queue = queue.Queue(maxsize=1)
    q.put("occupied")  # queue is now full
    cancel = threading.Event()
    cancel.set()
    assert put_until_cancelled(q, "item", cancel, timeout=0.01) is False
    assert q.qsize() == 1  # the blocked item was NOT enqueued


def test_put_until_cancelled_unblocks_when_cancel_set_from_another_thread() -> None:
    # The real deadlock shape: queue stays full, and cancel arrives later. The
    # put must return (False) rather than hang forever.
    q: queue.Queue = queue.Queue(maxsize=1)
    q.put("occupied")
    cancel = threading.Event()
    result: dict[str, bool] = {}

    def worker() -> None:
        result["value"] = put_until_cancelled(q, "item", cancel, timeout=0.02)

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
    drain_queue(q)
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
