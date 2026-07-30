from __future__ import annotations

import asyncio
import io
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.devices_service import DevicesService
from app.services.engines.ffmpeg_frame_source import FfmpegFrameSource
from app.services.engines.ffmpeg_frame_sink import RawPipeEncoder
from app.services.engines.gmfss.assets import GRAPH_NAMES
from app.services.engines.gmfss_engine import GmfssEngine
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

    def builtin_onnx_available(self, engine_model_name: str, scale: int | None = None) -> bool:
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


async def test_mode_full_when_job_needs_source_input(tmp_path: Path) -> None:
    # Pistas de audio extra / subtítulos ya NO excluyen del streaming: el
    # comando raw-pipe mapea el source como input adicional.
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, keep_subtitles=True)
    assert await upscaler._resolve_stream_pipeline_mode(job, 1) == STREAM_MODE_FULL


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
FULL_H, FULL_W = 16, 24  # resolución "padded" GMFSS de juguete (no cuadrada a propósito)


def make_combined_settings(tmp_path: Path) -> Settings:
    """Settings que satisfacen al GMFSS engine (model dir + ENABLE_GMFSS) y al
    motor ONNX de video (builtin dir aislado) — patrón del viejo test de fusión."""
    gmfss_dir = tmp_path / "gmfss"
    gmfss_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "resolution": {"fixed_padded_hw": [FULL_H, FULL_W]},
        "required_files": ["manifest.json"] + [f"{name}.onnx" for name in GRAPH_NAMES],
    }
    (gmfss_dir / "manifest.json").write_text(json.dumps(manifest))
    for name in GRAPH_NAMES:
        (gmfss_dir / f"{name}.onnx").write_bytes(b"fake")
    return make_stream_settings(tmp_path, ENABLE_GMFSS=True, GMFSS_MODEL_DIR=str(gmfss_dir))


class FakeGmfssSession:
    """Sesión fake determinista de los 4 grafos GMFSS — copiada de
    tests/test_gmfss_engine.py (FakeSession)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, _outputs, feeds):
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


def fake_gmfss_sessions(_device: str):
    return {name: FakeGmfssSession(name) for name in GRAPH_NAMES}


def make_gmfss_streaming_upscaler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VideoUpscaler:
    settings = make_combined_settings(tmp_path)
    settings.builtin_onnx_path.mkdir(parents=True, exist_ok=True)
    (settings.builtin_onnx_path / "realesr-animevideov3-x4-uint8.onnx").write_bytes(b"fake")
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
        self.command: list[str] | None = None

    def wait(self, timeout=None) -> int:
        self.returncode = self._final_returncode
        # Un ffmpeg real deja el archivo de salida al terminar bien; el fake
        # tambien, para que la validacion de output del pipeline vea lo mismo
        # que en produccion (un exit 0 sin archivo debe caer al clasico).
        if self.returncode == 0 and self.command:
            Path(self.command[-1]).write_bytes(b"fake-mp4-bytes")
        return self.returncode

    def poll(self):
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self._final_returncode = -9


def fake_encoder_spawn(fake: FakeEncodeProc):
    """_spawn monkeypatcheado que le pasa el comando al fake, para que sepa
    donde "escribir" el archivo de salida (ultimo argumento del comando)."""

    def _spawn(self, command: list[str]) -> FakeEncodeProc:
        fake.command = command
        return fake

    return _spawn


class FakeDecodeProc:
    """Popen fake de decode: stdout con frames rgb24 crudos — copiado de
    tests/test_ffmpeg_frame_source.py."""

    def __init__(self, stdout_bytes: bytes, returncode: int = 0) -> None:
        self.stdout = io.BytesIO(stdout_bytes)
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


def raw_source_frames(count: int) -> bytes:
    # Frame i uniforme en (i*17)%256 a resolución SOURCE_H x SOURCE_W.
    return b"".join(bytes([(i * 17) % 256]) * (SOURCE_H * SOURCE_W * 3) for i in range(count))


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


class RecordingRunProcessUpscaler(VideoUpscaler):
    """Registra los comandos de _run_process y fakea sus efectos (extract PNG /
    encode escribe el output) — patrón de StageTrackingVideoUpscaler de
    tests/test_pipeline_stage_order.py."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.commands: list[list[str]] = []

    async def _run_process(self, command: list[str]) -> None:
        self.commands.append(command)
        if "-fps_mode" in command:
            frames_dir = Path(command[-1]).parent
            frames_dir.mkdir(parents=True, exist_ok=True)
            (frames_dir / "00000001.png").write_bytes(b"png")
        elif "-framerate" in command:
            output = Path(command[-1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"fake-output-video")


def make_recording_upscaler(tmp_path: Path, **settings_overrides: object) -> RecordingRunProcessUpscaler:
    settings = make_stream_settings(tmp_path, **settings_overrides)
    return RecordingRunProcessUpscaler(
        settings,
        FakeNcnnEngine(),  # type: ignore[arg-type]
        FakeMediaTools(),  # type: ignore[arg-type]
        gmfss_engine=object(),  # type: ignore[arg-type]
        onnx_video_engine=FakeOnnxVideoEngine(),  # type: ignore[arg-type]
        model_registry=ModelRegistry(settings),
        devices=FakeDevicesService(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Pipeline completo (modo full): saltea la extracción PNG; fallback desde cero
# ---------------------------------------------------------------------------


async def test_full_mode_skips_png_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upscaler = make_recording_upscaler(tmp_path)
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    job = make_stream_job(tmp_path, source_path=source_path)
    full_calls: dict = {}

    async def fake_full(job_arg, fps_multiplier, output_path, fps, mux, codec_args, frames_exact=False):
        full_calls["fps"] = fps
        output_path.write_bytes(b"fake-video")
        return True

    monkeypatch.setattr(upscaler, "_try_stream_pipeline_full", fake_full)

    output = await upscaler.run(job)

    assert output.exists()
    assert full_calls["fps"] == "24/1"  # el fps del probe de FakeMediaTools
    assert all("-fps_mode" not in command for command in upscaler.commands), "corrió extracción PNG en modo full"
    assert job.metadata["progress"] == 1.0


async def test_full_mode_falls_back_to_classic_from_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_recording_upscaler(tmp_path)
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    job = make_stream_job(tmp_path, source_path=source_path)

    async def failing_full(job_arg, fps_multiplier, output_path, fps, mux, codec_args, frames_exact=False):
        job_arg.metadata["streamPipelineFallback"] = "boom"
        return False

    async def fake_upscale(job_arg, src, dst):
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "00000001.png").write_bytes(b"png")

    monkeypatch.setattr(upscaler, "_try_stream_pipeline_full", failing_full)
    monkeypatch.setattr(upscaler, "_upscale_frames", fake_upscale)

    output = await upscaler.run(job)

    assert output.exists()
    assert job.metadata["streamPipelineFallback"] == "boom"
    # El camino clásico corrió DESDE CERO: extracción PNG + encode PNG.
    assert any("-fps_mode" in command for command in upscaler.commands)
    assert any("-framerate" in command for command in upscaler.commands)


async def test_full_mode_gmfss_fallback_restores_source_frames_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_recording_upscaler(tmp_path)
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    job = make_stream_job(
        tmp_path,
        source_path=source_path,
        interp_engine="gmfss",
        fps_multiplier=2,
    )

    async def failing_stream(*args, **kwargs):
        raise RuntimeError("boom")

    async def fake_interp(job_arg, frames_dir, fps, mult, target_fps=None):
        return frames_dir, fps

    async def fake_upscale(job_arg, src, dst):
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "00000001.png").write_bytes(b"png")

    monkeypatch.setattr(upscaler, "_run_stream_pipeline", failing_stream)
    monkeypatch.setattr(upscaler, "_maybe_interpolate", fake_interp)
    monkeypatch.setattr(upscaler, "_upscale_frames", fake_upscale)

    output = await upscaler.run(job, fps_multiplier=2)

    assert output.exists()
    assert job.metadata["streamPipelineFallback"] == "boom"
    assert job.metadata["framesTotal"] == 48
    assert any("-fps_mode" in command for command in upscaler.commands)
    assert any("-framerate" in command for command in upscaler.commands)


async def test_stream_pipeline_full_integration_no_interp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Integración real (spec "Testing/Integración", conteo 1→1 y orden):
    # FramePipeline + FfmpegFrameSource + etapa de upscale reales; sesión ONNX
    # y ambos procesos ffmpeg son fakes.
    upscaler = make_streaming_upscaler_with_real_engines(tmp_path, monkeypatch)
    job = make_stream_job(tmp_path, device="cpu")
    job.metadata.update({"sourceWidth": SOURCE_W, "sourceHeight": SOURCE_H, "framesTotal": 3})
    decode_proc = FakeDecodeProc(raw_source_frames(3))
    sink_proc = FakeEncodeProc()
    monkeypatch.setattr(FfmpegFrameSource, "_spawn", lambda self, command: decode_proc)
    monkeypatch.setattr(RawPipeEncoder, "_spawn", fake_encoder_spawn(sink_proc))

    ok = await upscaler._try_stream_pipeline_full(job, 1, tmp_path / "out.mp4", "24/1", None, [])

    assert ok is True
    assert job.metadata["streamPipeline"] is True
    data = sink_proc.stdin.getvalue()
    frame_bytes = (SOURCE_H * 2) * (SOURCE_W * 2) * 3  # la sesión fake dobla
    assert len(data) == 3 * frame_bytes
    assert [data[i * frame_bytes] for i in range(3)] == [0, 17, 34]  # orden fuente


# ---------------------------------------------------------------------------
# GMFSS dentro del pipeline completo: conteo 1→2x, orden y fallback
# ---------------------------------------------------------------------------


async def test_stream_pipeline_full_with_gmfss_doubles_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_gmfss_streaming_upscaler(tmp_path, monkeypatch)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2, device="cpu")
    job.metadata.update({"sourceWidth": SOURCE_W, "sourceHeight": SOURCE_H, "framesTotal": 3})
    decode_proc = FakeDecodeProc(raw_source_frames(3))
    sink_proc = FakeEncodeProc()
    monkeypatch.setattr(FfmpegFrameSource, "_spawn", lambda self, command: decode_proc)
    monkeypatch.setattr(RawPipeEncoder, "_spawn", fake_encoder_spawn(sink_proc))

    ok = await upscaler._try_stream_pipeline_full(job, 2, tmp_path / "out.mp4", "24/1", None, [])

    assert ok is True
    assert job.metadata["streamPipeline"] is True
    assert job.metadata["framesTotal"] == 6  # denominador honesto: salida interpolada
    assert job.metadata["outputFps"] == "48/1"
    data = sink_proc.stdin.getvalue()
    frame_bytes = (SOURCE_H * 2) * (SOURCE_W * 2) * 3  # la sesión fake dobla
    assert len(data) == 6 * frame_bytes  # conteo 1→2x (3 fuente -> 6 salida)
    # Orden: s0, interp, s1, interp, interp, s2 (plan(3→6) = [1, 2]); los frames
    # fuente pasan verbatim y doblados conservan su primer byte uniforme.
    assert data[0 * frame_bytes] == 0
    assert data[2 * frame_bytes] == 17
    assert data[5 * frame_bytes] == 34


async def test_stream_pipeline_full_gmfss_falls_back_on_source_count_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # El decode entrega 2 frames pero el probe prometió 3: el plan GMFSS no
    # cierra, la etapa revienta en flush() y el job cae al clásico (no falla).
    upscaler = make_gmfss_streaming_upscaler(tmp_path, monkeypatch)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2, device="cpu")
    job.metadata.update({"sourceWidth": SOURCE_W, "sourceHeight": SOURCE_H, "framesTotal": 3})
    decode_proc = FakeDecodeProc(raw_source_frames(2))
    sink_proc = FakeEncodeProc()
    monkeypatch.setattr(FfmpegFrameSource, "_spawn", lambda self, command: decode_proc)
    monkeypatch.setattr(RawPipeEncoder, "_spawn", fake_encoder_spawn(sink_proc))
    output_path = tmp_path / "out.mp4"

    ok = await upscaler._try_stream_pipeline_full(job, 2, output_path, "24/1", None, [])

    assert ok is False
    assert "streamPipelineFallback" in job.metadata
    assert not output_path.exists()


async def test_run_routes_gmfss_job_through_full_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upscaler = make_recording_upscaler(tmp_path)
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    job = make_stream_job(tmp_path, source_path=source_path, interp_engine="gmfss", fps_multiplier=2)
    seen: dict = {}

    async def fake_full(job_arg, fps_multiplier, output_path, fps, mux, codec_args, frames_exact=False):
        seen["fps_multiplier"] = fps_multiplier
        output_path.write_bytes(b"fake-video")
        return True

    monkeypatch.setattr(upscaler, "_try_stream_pipeline_full", fake_full)

    output = await upscaler.run(job, fps_multiplier=2)

    assert output.exists()
    assert seen["fps_multiplier"] == 2
    assert all("-fps_mode" not in command for command in upscaler.commands), "GMFSS full no debe extraer PNGs"


async def test_full_pipeline_cancel_waits_for_worker_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mismo contrato shield+await del raw-pipe/motores: al propagar el cancel,
    # el worker YA terminó — la limpieza del work-dir no corre contra threads vivos.
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path)
    finished = threading.Event()

    def blocking(job_arg, source_factory, stage_factory, device, width, height, expected, command, counter, cancel_event):
        cancel_event.wait(timeout=10)
        time.sleep(0.2)  # simula un teardown no interrumpible en vuelo
        finished.set()

    monkeypatch.setattr(upscaler, "_run_stream_pipeline_blocking", blocking)

    task = asyncio.create_task(
        upscaler._try_stream_pipeline_full(job, 1, tmp_path / "out.mp4", "24/1", None, [])
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set(), "el cancel propagó antes de que el worker terminara"


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
    monkeypatch.setattr(RawPipeEncoder, "_spawn", fake_encoder_spawn(fake_proc))
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
    monkeypatch.setattr(RawPipeEncoder, "_spawn", fake_encoder_spawn(fake_proc))
    job = make_stream_job(tmp_path, device="cpu")
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"parcial")

    ok = await upscaler._try_stream_pipeline_from_dir(job, frames_dir, output_path, "24/1", None, [])

    assert ok is False
    assert "streamPipelineFallback" in job.metadata
    assert not output_path.exists()  # el output parcial se borra antes del fallback


# ---------------------------------------------------------------------------
# Diagnóstico de fallos de ffmpeg: el resumen de error debe conservar la línea
# que NOMBRA el input/output culpable, no quedarse con el último renglón (que
# en un fallo de encode real es una estadística del codec, no un error).
# ---------------------------------------------------------------------------


def test_summarize_keeps_the_line_naming_the_failing_input(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    stderr = (
        b"[in#0 @ 000001] Error opening input: Invalid argument\n"
        b"Error opening input file C:/work/job/frames-interp/%08d.png.\n"
        b"Error opening input files: Invalid argument\n"
    )

    summary = upscaler._summarize_process_error(stderr, b"")

    assert "frames-interp" in summary, "se perdió la ruta del input que falló"
    assert "Invalid argument" in summary


def test_summarize_prefers_error_lines_over_trailing_codec_stats(tmp_path: Path) -> None:
    # Caso real: a verbosidad por defecto libx264 escupe estadísticas DESPUÉS
    # del error, así que lines[-1] devolvía "kb/s:52.80" como mensaje de error.
    upscaler = make_stream_upscaler(tmp_path)
    stderr = (
        b"[libx264 @ 0001] Error: invalid parameter\n"
        b"[libx264 @ 0001] i16 v,h,dc,p: 100%  0%  0%  0%\n"
        b"[libx264 @ 0001] kb/s:52.80\n"
    )

    summary = upscaler._summarize_process_error(stderr, b"")

    assert "invalid parameter" in summary
    assert "kb/s" not in summary


def test_summarize_falls_back_to_last_line_without_error_markers(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    assert upscaler._summarize_process_error(b"algo raro\nultima linea\n", b"") == "ultima linea"


def test_summarize_keeps_the_curated_x265_message(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    stderr = b"[libx265 @ 0001] cannot open libx265\nError opening output files: Invalid argument\n"
    assert "H.265/libx265" in upscaler._summarize_process_error(stderr, b"")


# ---------------------------------------------------------------------------
# Guard de espacio en disco: el camino clásico materializa TODOS los frames
# upscaleados en PNG. Un 4x de 2h llega a terabytes; hay que fallar en 1s con
# un mensaje claro, no a las 6h con un error críptico de ffmpeg.
# ---------------------------------------------------------------------------


def test_estimate_png_workdir_bytes_dominated_by_upscaled_frames() -> None:
    from app.services.video_upscaler import estimate_png_workdir_bytes

    # 100 frames fuente 100x100, x4 => 400x400 (16x los píxeles), sin interp.
    kwargs = dict(
        source_width=100, source_height=100, scale=4,
        source_frame_count=100, output_frame_count=100,
        writes_interpolated_frames=False,
    )
    with_upscaled = estimate_png_workdir_bytes(**kwargs, writes_upscaled_frames=True)
    only_source = estimate_png_workdir_bytes(**kwargs, writes_upscaled_frames=False)

    # El set upscaleado es el que vuelve inviables los jobs largos: tiene que
    # dominar el total, no ser un detalle sobre los frames de entrada.
    assert with_upscaled > only_source * 2


def test_estimate_png_workdir_bytes_zero_when_no_png_is_written() -> None:
    from app.services.video_upscaler import estimate_png_workdir_bytes

    estimate = estimate_png_workdir_bytes(
        source_width=1920,
        source_height=1080,
        scale=4,
        source_frame_count=1000,
        output_frame_count=1000,
        writes_upscaled_frames=False,
        writes_interpolated_frames=False,
    )
    # Solo frames-in a resolución fuente: nunca puede superar el crudo RGB8.
    assert 0 < estimate < 1000 * 1920 * 1080 * 3


@pytest.mark.asyncio
async def test_run_fails_fast_when_the_png_path_does_not_fit_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil as shutil_module

    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, device="cpu")
    job.metadata["framesTotal"] = 400_000
    job.metadata["sourceWidth"] = 1920
    job.metadata["sourceHeight"] = 1080

    monkeypatch.setattr(shutil_module, "disk_usage", lambda path: _FakeDiskUsage())

    with pytest.raises(RuntimeError, match="espacio"):
        upscaler._ensure_disk_room_for_png_path(job, tmp_path, fps_multiplier=1, stream_mode=None)


def test_disk_guard_passes_when_there_is_room(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, device="cpu")
    job.metadata["framesTotal"] = 10
    job.metadata["sourceWidth"] = 64
    job.metadata["sourceHeight"] = 64

    upscaler._ensure_disk_room_for_png_path(job, tmp_path, fps_multiplier=1, stream_mode=None)


def test_disk_guard_skips_when_frame_count_is_unknown(tmp_path: Path) -> None:
    # framesTotal None (VFR) es honesto: sin conteo no hay estimación honesta.
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, device="cpu")
    job.metadata["framesTotal"] = None
    job.metadata["sourceWidth"] = 1920
    job.metadata["sourceHeight"] = 1080

    upscaler._ensure_disk_room_for_png_path(job, tmp_path, fps_multiplier=1, stream_mode=None)


class _FakeDiskUsage:
    total = 4_000_000_000_000
    used = 3_900_000_000_000
    free = 100_000_000_000  # 100 GB libres


# ---------------------------------------------------------------------------
# Orden de argumentos del encode: en ffmpeg -map es opcion de SALIDA. Emitir
# -map entre dos -i hace que ffmpeg las lea como opciones del input siguiente y
# rechace el comando entero ("Option map cannot be applied to input url ...").
# Se dispara solo con audio preparado + fuente extra (subtitulos o pistas
# adicionales), que es justo el caso que ningun test ejecutaba contra ffmpeg.
# ---------------------------------------------------------------------------


def _map_flags_come_after_every_input(cmd: list[str]) -> bool:
    last_input = max(i for i, token in enumerate(cmd) if token == "-i")
    first_map = next((i for i, token in enumerate(cmd) if token == "-map"), None)
    return first_map is None or first_map > last_input


def test_encode_command_emits_every_input_before_any_map_with_subtitles(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, keep_subtitles=True)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames", "24/1", tmp_path / "audio.m4a", ["-c:a", "aac"],
        tmp_path / "out.mkv", "libx264",
    )

    assert _map_flags_come_after_every_input(cmd), f"-map antes de un -i: {cmd}"
    assert "-map" in cmd and str(job.source_path) in cmd


def test_encode_command_emits_every_input_before_any_map_with_extra_audio(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, keep_audio=True, audio_track_indices=[1, 2, 3])

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames", "24/1", tmp_path / "audio.m4a", ["-c:a", "aac"],
        tmp_path / "out.mkv", "libx264",
    )

    assert _map_flags_come_after_every_input(cmd), f"-map antes de un -i: {cmd}"


def test_encode_command_keeps_map_order_video_then_primary_audio_then_extras(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, keep_audio=True, audio_track_indices=[1, 2], keep_subtitles=True)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames", "24/1", tmp_path / "audio.m4a", ["-c:a", "aac"],
        tmp_path / "out.mkv", "libx264",
    )

    maps = [cmd[i + 1] for i, token in enumerate(cmd) if token == "-map"]
    assert maps[0] == "0:v:0"
    assert maps[1] == "1:a:0"       # audio primario ya procesado
    assert maps[-1].endswith(":s?")  # subtitulos al final
    assert "2:2" in maps             # pista extra por indice absoluto de la fuente


def test_encode_command_without_source_input_is_unchanged(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames", "24/1", tmp_path / "audio.m4a", ["-c:a", "aac"],
        tmp_path / "out.mkv", "libx264",
    )

    assert _map_flags_come_after_every_input(cmd)
    assert str(job.source_path) not in cmd
    assert [cmd[i + 1] for i, t in enumerate(cmd) if t == "-map"] == ["0:v:0", "1:a:0"]


# ---------------------------------------------------------------------------
# Pistas extra / subtítulos EN EL STREAM PIPELINE. Antes obligaban al camino
# PNG clásico, que a 4x materializa cientos de GB de frames intermedios. El
# comando raw-pipe puede tomar el archivo fuente como input adicional igual que
# el camino PNG (verificado contra ffmpeg real), así que ya no hay motivo para
# excluirlos del streaming.
# ---------------------------------------------------------------------------


def test_rawpipe_command_maps_extra_audio_and_subtitles_from_source(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, keep_audio=True, audio_track_indices=[1, 2], keep_subtitles=True)

    cmd = upscaler._build_rawpipe_command(
        3840, 2880, "24/1", tmp_path / "audio.m4a", ["-c:a", "aac"],
        tmp_path / "out.mkv", job, "libx264",
    )

    assert str(job.source_path) in cmd, "la fuente no se agregó como input"
    maps = [cmd[i + 1] for i, token in enumerate(cmd) if token == "-map"]
    assert maps[0] == "0:v:0"      # el pipe de frames crudos
    assert maps[1] == "1:a:0"      # audio primario ya procesado
    assert "2:2" in maps           # pista extra por índice absoluto
    assert maps[-1].endswith(":s?")
    assert "-c:s" in cmd


def test_rawpipe_command_emits_every_input_before_any_map(tmp_path: Path) -> None:
    # Mismo invariante que el camino PNG: -map es opción de salida.
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, keep_audio=True, audio_track_indices=[1, 2], keep_subtitles=True)

    cmd = upscaler._build_rawpipe_command(
        3840, 2880, "24/1", tmp_path / "audio.m4a", ["-c:a", "aac"],
        tmp_path / "out.mkv", job, "libx264",
    )

    assert _map_flags_come_after_every_input(cmd), f"-map antes de un -i: {cmd}"


def test_rawpipe_command_without_extras_is_unchanged(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path)

    cmd = upscaler._build_rawpipe_command(
        3840, 2880, "24/1", tmp_path / "audio.m4a", ["-c:a", "aac"],
        tmp_path / "out.mkv", job, "libx264",
    )

    assert str(job.source_path) not in cmd
    assert [cmd[i + 1] for i, t in enumerate(cmd) if t == "-map"] == ["0:v:0", "1:a:0"]


@pytest.mark.asyncio
async def test_gate_routes_subtitle_job_through_the_stream_pipeline(tmp_path: Path) -> None:
    # El caso que antes caía al clásico y quemaba cientos de GB de PNG.
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, keep_subtitles=True)

    assert await upscaler._resolve_stream_pipeline_mode(job, 1) == STREAM_MODE_FULL


@pytest.mark.asyncio
async def test_gate_routes_multi_audio_job_through_the_stream_pipeline(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path)
    job = make_stream_job(tmp_path, keep_audio=True, audio_track_indices=[1, 2, 3])

    assert await upscaler._resolve_stream_pipeline_mode(job, 1) == STREAM_MODE_FULL
