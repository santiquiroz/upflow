from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.devices_service import DevicesService
from app.services.engines.ffmpeg_frame_sink import RawPipeEncoder
from app.services.engines.onnx_video_upscaler import OnnxVideoUpscaler
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.video_upscaler import (
    STREAM_MODE_FULL,
    STREAM_MODE_HYBRID,
    VideoUpscaler,
)

# ---------------------------------------------------------------------------
# Stream pipeline (spec 2026-07-25-stream-frame-pipeline-design.md) — ruteo de
# modos, tramo híbrido, pipeline completo y fallback clásico. Fakes calcados de
# tests/test_video_backend_dispatch.py; ningún binario real corre acá.
# ---------------------------------------------------------------------------


class FakeNcnnEngine:
    def available(self) -> bool:
        return True


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [{"codec_type": "video", "width": 1280, "height": 720, "avg_frame_rate": "24/1"}],
            "format": {"duration": "2.0"},
        }


class FakeDevicesService:
    def __init__(self, valid_ids: tuple[str, ...] = ("cpu", "dml:0")) -> None:
        self._valid_ids = valid_ids

    def list_devices(self) -> list[dict]:
        # "name" presente: _device_name lo indexa al resolver el encoder "auto".
        return [{"id": device_id, "name": "Fake GPU"} for device_id in self._valid_ids]

    def validate(self, device_id: str) -> dict:
        if device_id not in self._valid_ids:
            raise ValueError(f"Unknown device id: {device_id!r}")
        return {"id": device_id}


class FakeOnnxVideoEngine:
    """Stand-in de OnnxVideoUpscaler para el gate: solo responde los probes de
    capacidad que _resolve_builtin_backend consulta."""

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


def make_onnx_entry(**overrides: object) -> ModelEntry:
    defaults: dict[str, object] = {
        "id": "fake-onnx-2x",
        "name": "Fake ONNX 2x",
        "kind": ModelKind.onnx,
        "source": "https://huggingface.co/example/fake-onnx-2x",
        "size_bytes": 1_000,
        "scale": 2,
        "arch": "fake",
        "file_path": "onnx/fake-onnx-2x.onnx",
        "status": ModelStatus.installed,
    }
    defaults.update(overrides)
    return ModelEntry(**defaults)


def make_stream_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "RUNTIME_DIR": str(tmp_path / "runtime"),
        "BUILTIN_ONNX_DIR": str(tmp_path / "builtin-onnx"),
    }
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def make_stream_upscaler(
    tmp_path: Path,
    *,
    gmfss_engine: object | None = object(),
    onnx_video_engine: object | None = None,
    registry: ModelRegistry | None = None,
    **settings_overrides: object,
) -> VideoUpscaler:
    settings = make_stream_settings(tmp_path, **settings_overrides)
    engine = FakeOnnxVideoEngine() if onnx_video_engine is None else onnx_video_engine
    return VideoUpscaler(
        settings,
        FakeNcnnEngine(),  # type: ignore[arg-type]
        FakeMediaTools(),  # type: ignore[arg-type]
        gmfss_engine=gmfss_engine,  # type: ignore[arg-type]
        onnx_video_engine=engine,  # type: ignore[arg-type]
        model_registry=registry if registry is not None else ModelRegistry(settings),
        devices=FakeDevicesService(),  # type: ignore[arg-type]
    )


def make_stream_job(tmp_path: Path, **overrides: object) -> VideoUpscaleJob:
    fields: dict[str, object] = dict(
        source_path=tmp_path / "clip.mp4",
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x4",
        scale=4,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        model_id=None,
        device="cpu",
        backend="onnx",
    )
    fields.update(overrides)
    job = VideoUpscaleJob(**fields)
    # Metadata que _run_pipeline estampa en el probe, precondición del gate:
    # 1280x720 x4 => 14.7Mpx de salida, sobre el umbral raw_pipe_min_output_pixels.
    job.metadata.update({"sourceWidth": 1280, "sourceHeight": 720, "framesTotal": 48})
    return job


# ---------------------------------------------------------------------------
# Gate: _resolve_stream_pipeline_mode
# ---------------------------------------------------------------------------


def test_enable_stream_pipeline_defaults_to_true() -> None:
    assert Settings(_env_file=None).enable_stream_pipeline is True


async def test_mode_full_when_no_interpolation(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) == STREAM_MODE_FULL


async def test_mode_none_when_flag_off(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path, ENABLE_STREAM_PIPELINE=False)
    job = make_stream_job(tmp_path)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_none_without_onnx_video_engine(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path, onnx_video_engine=False)
    upscaler.onnx_video_engine = None
    job = make_stream_job(tmp_path)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_none_when_backend_resolves_ncnn(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, backend="ncnn")
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_none_for_hf_onnx_model(tmp_path: Path) -> None:
    settings_dir = tmp_path
    registry = ModelRegistry(make_stream_settings(settings_dir))
    registry.register(make_onnx_entry())
    upscaler = make_stream_upscaler(tmp_path, registry=registry)
    job = make_stream_job(tmp_path, model_name="fake-onnx-2x", model_id="fake-onnx-2x", scale=2)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_none_below_min_output_pixels(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path)
    job.metadata.update({"sourceWidth": 64, "sourceHeight": 64})  # 64x64x4x = chico
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_none_when_job_needs_source_input(tmp_path: Path) -> None:
    # Pistas de audio extra / subtítulos: solo _build_encode_command (camino PNG)
    # sabe mapearlos desde el source original — misma restricción que el raw-pipe.
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, keep_subtitles=True)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) is None


async def test_mode_hybrid_for_rife_interpolation(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, fps_multiplier=2)
    assert await upscaler._resolve_stream_pipeline_mode(job, 2) == STREAM_MODE_HYBRID


async def test_mode_full_for_gmfss_with_known_frames_total(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2)
    assert await upscaler._resolve_stream_pipeline_mode(job, 2) == STREAM_MODE_FULL


async def test_mode_none_for_gmfss_without_frames_total(tmp_path: Path) -> None:
    # VFR/indeterminable: sin conteo honesto no hay plan GMFSS posible.
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2)
    job.metadata["framesTotal"] = None
    assert await upscaler._resolve_stream_pipeline_mode(job, 2) is None


@pytest.mark.parametrize("frames_total", [0, -1])
async def test_mode_none_for_gmfss_with_non_positive_frames_total(
    tmp_path: Path, frames_total: int
) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2)
    job.metadata["framesTotal"] = frames_total
    assert await upscaler._resolve_stream_pipeline_mode(job, 2) is None


async def test_mode_none_for_gmfss_without_engine(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path, gmfss_engine=None)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2)
    assert await upscaler._resolve_stream_pipeline_mode(job, 2) is None


SOURCE_H, SOURCE_W = 8, 12


def write_source_frames(directory: Path, count: int) -> None:
    # Primer píxel R = offset del frame (i*17): el orden queda verificable en
    # los bytes crudos que recibe el encoder fake.
    directory.mkdir(parents=True, exist_ok=True)
    row = np.arange(SOURCE_H, dtype=np.int32).reshape(SOURCE_H, 1)
    col = np.arange(SOURCE_W, dtype=np.int32).reshape(1, SOURCE_W)
    for index in range(count):
        offset = (index * 17) % 256
        frame = np.empty((SOURCE_H, SOURCE_W, 3), dtype=np.uint8)
        frame[:, :, 2] = (row * 0 + offset) % 256  # canal R en BGR de cv2
        frame[:, :, 1] = (col * 13 + offset * 2) % 256
        frame[:, :, 0] = (row + col * 5 + offset * 3) % 256
        assert cv2.imwrite(str(directory / f"{index + 1:08d}.png"), frame)


class _IoInfo:
    def __init__(self, name: str) -> None:
        self.name = name


class Double2xUint8Session:
    """Fake de sesión ONNX uint8: dobla H/W — copiado de tests/test_onnx_video_upscaler.py."""

    def __init__(self) -> None:
        self._input = _IoInfo("image")
        self._output = _IoInfo("upscaled")

    def get_inputs(self) -> list[_IoInfo]:
        return [self._input]

    def get_outputs(self) -> list[_IoInfo]:
        return [self._output]

    def run(self, output_names, input_feed):
        array = input_feed[self._input.name]
        assert array.dtype == np.uint8
        return [np.repeat(np.repeat(array, 2, axis=1), 2, axis=2)]


class FakeStdin(io.BytesIO):
    def close(self) -> None:  # type: ignore[override]
        pass  # el buffer sigue legible para inspeccionar los bytes escritos


class FakeEncodeProc:
    def __init__(self, returncode: int = 0) -> None:
        self.stdin = FakeStdin()
        self.stderr = io.BytesIO(b"")
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.killed = False

    def wait(self, timeout=None) -> int:
        self.returncode = self._final_returncode
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self._final_returncode = -9


def make_streaming_upscaler_with_real_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> VideoUpscaler:
    """VideoUpscaler con un OnnxVideoUpscaler REAL (sesión fake Double2x) — el
    FramePipeline y las etapas corren de verdad; solo sesión y procesos son fake."""
    settings = make_stream_settings(tmp_path)
    settings.builtin_onnx_path.mkdir(parents=True, exist_ok=True)
    (settings.builtin_onnx_path / "realesr-animevideov3-x4-uint8.onnx").write_bytes(b"fake")
    onnx_video = OnnxVideoUpscaler(
        settings, ModelRegistry(settings), DevicesService(settings), GpuSessionCoordinator()
    )
    monkeypatch.setattr(onnx_video, "_create_session", lambda model_path, device: Double2xUint8Session())
    return VideoUpscaler(
        settings,
        FakeNcnnEngine(),  # type: ignore[arg-type]
        FakeMediaTools(),  # type: ignore[arg-type]
        gmfss_engine=None,
        onnx_video_engine=onnx_video,
        model_registry=ModelRegistry(settings),
        devices=DevicesService(settings),
    )


# ---------------------------------------------------------------------------
# Tramo híbrido: ruteo + fallback + integración con FramePipeline real
# ---------------------------------------------------------------------------


async def test_hybrid_mode_streams_from_interp_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, fps_multiplier=2)
    frames_in = tmp_path / "frames-in"
    frames_in.mkdir(parents=True)
    interp_dir = tmp_path / "frames-interp"
    calls: dict = {}

    async def fake_interp(job_arg, frames_dir, fps, mult, target_fps=None):
        interp_dir.mkdir(parents=True, exist_ok=True)
        return interp_dir, "48/1"

    async def fake_from_dir(job_arg, frames_dir, output_path, encode_fps, mux, codec_args):
        calls["dir"] = frames_dir
        calls["fps"] = encode_fps
        output_path.write_bytes(b"fake-video")
        return True

    monkeypatch.setattr(upscaler, "_maybe_interpolate", fake_interp)
    monkeypatch.setattr(upscaler, "_try_stream_pipeline_from_dir", fake_from_dir)

    encode_dir, encode_fps = await upscaler._interpolate_and_upscale(
        job, frames_in, tmp_path / "frames-out", "24/1", 2, tmp_path / "out.mp4", None, [], STREAM_MODE_HYBRID
    )

    assert encode_dir is None  # el caller NO encodea: el tramo ya produjo el output
    assert encode_fps == "48/1"
    assert calls["dir"] == interp_dir


async def test_hybrid_fallback_uses_classic_png_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, fps_multiplier=2)
    frames_in = tmp_path / "frames-in"
    frames_in.mkdir(parents=True)
    frames_out = tmp_path / "frames-out"
    interp_dir = tmp_path / "frames-interp"

    async def fake_interp(job_arg, frames_dir, fps, mult, target_fps=None):
        interp_dir.mkdir(parents=True, exist_ok=True)
        (interp_dir / "00000001.png").write_bytes(b"png")
        return interp_dir, "48/1"

    async def failing_from_dir(job_arg, frames_dir, output_path, encode_fps, mux, codec_args):
        job_arg.metadata["streamPipelineFallback"] = "boom"
        return False

    upscaled = {"n": 0}

    async def fake_upscale(job_arg, src, dst):
        upscaled["n"] += 1
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "00000001.png").write_bytes(b"png")

    monkeypatch.setattr(upscaler, "_maybe_interpolate", fake_interp)
    monkeypatch.setattr(upscaler, "_try_stream_pipeline_from_dir", failing_from_dir)
    monkeypatch.setattr(upscaler, "_upscale_frames", fake_upscale)

    encode_dir, encode_fps = await upscaler._interpolate_and_upscale(
        job, frames_in, frames_out, "24/1", 2, tmp_path / "out.mp4", None, [], STREAM_MODE_HYBRID
    )

    assert encode_dir == frames_out  # el caller encodea por el camino PNG clásico
    assert upscaled["n"] == 1
    assert job.metadata["streamPipelineFallback"] == "boom"


async def test_stream_pipeline_from_dir_streams_all_frames_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Integración real (spec "Testing/Integración", conteo 1→1): FramePipeline +
    # etapa de upscale reales; solo la sesión ONNX y el proceso ffmpeg son fake.
    upscaler = make_streaming_upscaler_with_real_engines(tmp_path, monkeypatch)
    frames_dir = tmp_path / "frames-interp"
    write_source_frames(frames_dir, 5)
    fake_proc = FakeEncodeProc()
    monkeypatch.setattr(RawPipeEncoder, "_spawn", lambda self, command: fake_proc)
    job = make_stream_job(tmp_path, device="cpu")

    ok = await upscaler._try_stream_pipeline_from_dir(
        job, frames_dir, tmp_path / "out.mp4", "24/1", None, []
    )

    assert ok is True
    assert job.metadata["streamPipeline"] is True
    assert job.metadata["framesTotal"] == 5  # denominador honesto del tramo
    data = fake_proc.stdin.getvalue()
    # La sesión fake dobla (no 4x como job.scale): el tamaño real escrito manda.
    frame_bytes = (SOURCE_H * 2) * (SOURCE_W * 2) * 3
    assert len(data) == 5 * frame_bytes
    # Orden 1..5 por el primer byte de cada frame (R de (0,0) = offset i*17).
    assert [data[i * frame_bytes] for i in range(5)] == [(i * 17) % 256 for i in range(5)]


async def test_stream_pipeline_from_dir_falls_back_on_encoder_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_streaming_upscaler_with_real_engines(tmp_path, monkeypatch)
    frames_dir = tmp_path / "frames-interp"
    write_source_frames(frames_dir, 2)
    fake_proc = FakeEncodeProc(returncode=1)  # ffmpeg "falla" al cerrar
    monkeypatch.setattr(RawPipeEncoder, "_spawn", lambda self, command: fake_proc)
    job = make_stream_job(tmp_path, device="cpu")
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"parcial")

    ok = await upscaler._try_stream_pipeline_from_dir(job, frames_dir, output_path, "24/1", None, [])

    assert ok is False
    assert "streamPipelineFallback" in job.metadata
    assert not output_path.exists()  # el output parcial se borra antes del fallback
