from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
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


async def test_mode_none_for_gmfss_without_engine(tmp_path: Path) -> None:
    upscaler = make_stream_upscaler(tmp_path, gmfss_engine=None)
    job = make_stream_job(tmp_path, interp_engine="gmfss", fps_multiplier=2)
    assert await upscaler._resolve_stream_pipeline_mode(job, 2) is None
