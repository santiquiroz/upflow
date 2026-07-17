from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from app.api.routes import create_video_job
from app.config import (
    UPSCALE_BACKEND_NCNN,
    UPSCALE_BACKEND_ONNX,
    Settings,
)
from app.models import JobStatus, VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager
from app.services.video_upscaler import VideoUpscaler

# ---------------------------------------------------------------------------
# SP11 Task 2 - VideoUpscaler dispatches a BUILTIN model's frames to the ncnn
# subprocess or the optimized onnx video engine per the resolved backend
# (mock both, assert exactly one runs). Plus route-level honoring/validation
# of the per-job `backend` override.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


class FakeNcnnEngine:
    def available(self) -> bool:
        return True


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [{"codec_type": "video", "width": 4, "height": 4, "avg_frame_rate": "30/1"}],
            "format": {"duration": "1.0"},
        }


class FakeDevicesService:
    def __init__(self, valid_ids: tuple[str, ...] = ("cpu", "dml:0")) -> None:
        self._valid_ids = valid_ids

    def list_devices(self) -> list[dict]:
        return [{"id": device_id} for device_id in self._valid_ids]

    def resolve_default(self, devices: list[dict] | None = None) -> dict:
        return {"id": self._valid_ids[-1]}

    def validate(self, device_id: str) -> dict:
        if device_id not in self._valid_ids:
            raise ValueError(f"Unknown device id: {device_id!r}")
        return {"id": device_id}


class FakeHfOnnxEngine:
    """Stands in for OnnxUpscaler (HF-installed onnx models)."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, str, str]] = []

    async def run_frames(self, frames_in: Path, frames_out: Path, model_id: str, device: str) -> Path:
        self.calls.append((frames_in, frames_out, model_id, device))
        frames_out.mkdir(parents=True, exist_ok=True)
        for frame in sorted(frames_in.glob("*.png")):
            (frames_out / frame.name).write_bytes(b"hf-onnx-frame")
        return frames_out


class FakeOnnxVideoEngine:
    def __init__(self, *, available: bool = True, gpu_ep: bool = True, builtin_available: bool = True) -> None:
        self._available = available
        self._gpu_ep = gpu_ep
        self._builtin_available = builtin_available
        self.calls: list[tuple[Path, Path, str, str]] = []

    def available(self) -> bool:
        return self._available

    def has_gpu_execution_provider(self) -> bool:
        return self._gpu_ep

    def builtin_onnx_available(self, engine_model_name: str) -> bool:
        return self._builtin_available

    async def run_frames_builtin(
        self, frames_in: Path, frames_out: Path, engine_model_name: str, device: str
    ) -> Path:
        self.calls.append((frames_in, frames_out, engine_model_name, device))
        frames_out.mkdir(parents=True, exist_ok=True)
        for frame in sorted(frames_in.glob("*.png")):
            (frames_out / frame.name).write_bytes(b"onnx-frame")
        return frames_out


class RecordingVideoUpscaler(VideoUpscaler):
    """Records ncnn dispatch without needing a real Real-ESRGAN binary."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.ncnn_calls = 0

    async def _upscale_frames_ncnn(self, job: VideoUpscaleJob, frames_in: Path, frames_out: Path) -> None:
        self.ncnn_calls += 1
        frames_out.mkdir(parents=True, exist_ok=True)
        for frame in sorted(frames_in.glob("*.png")):
            (frames_out / frame.name).write_bytes(b"ncnn-frame")


def make_upscaler(
    settings: Settings,
    *,
    onnx_video_engine: FakeOnnxVideoEngine | None = None,
    onnx_engine: FakeHfOnnxEngine | None = None,
    registry: ModelRegistry | None = None,
) -> RecordingVideoUpscaler:
    return RecordingVideoUpscaler(
        settings,
        FakeNcnnEngine(),
        FakeMediaTools(),
        rife_engine=None,
        audio_enhancers=None,
        onnx_engine=onnx_engine,
        model_registry=registry if registry is not None else ModelRegistry(settings),
        onnx_video_engine=onnx_video_engine,
    )


def make_video_job(source_path: Path, **overrides: object) -> VideoUpscaleJob:
    fields = dict(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesr-animevideov3-x4",
        model_id="realesr-animevideov3-x4",
        scale=4,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        device="dml:0",
    )
    fields.update(overrides)
    return VideoUpscaleJob(**fields)


def make_frames_in(tmp_path: Path, count: int = 2) -> Path:
    frames_in = tmp_path / "frames-in"
    frames_in.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        (frames_in / f"{index:08d}.png").write_bytes(b"frame-bytes")
    return frames_in


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


# ---------------------------------------------------------------------------
# Dispatch: builtin model -> ncnn vs optimized onnx engine
# ---------------------------------------------------------------------------


async def test_auto_dispatches_builtin_to_onnx_when_available_and_gpu(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)  # UPSCALE_BACKEND defaults to auto
    onnx_video = FakeOnnxVideoEngine(available=True, gpu_ep=True, builtin_available=True)
    upscaler = make_upscaler(settings, onnx_video_engine=onnx_video)
    frames_in = make_frames_in(tmp_path)
    frames_out = tmp_path / "frames-out"

    await upscaler._upscale_frames(make_video_job(tmp_path / "clip.mp4"), frames_in, frames_out)

    assert len(onnx_video.calls) == 1
    assert upscaler.ncnn_calls == 0
    assert onnx_video.calls[0][2] == "realesr-animevideov3-x4"


async def test_auto_dispatches_builtin_to_ncnn_when_onnx_engine_unavailable(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    onnx_video = FakeOnnxVideoEngine(available=False, gpu_ep=True, builtin_available=True)
    upscaler = make_upscaler(settings, onnx_video_engine=onnx_video)
    frames_in = make_frames_in(tmp_path)

    await upscaler._upscale_frames(make_video_job(tmp_path / "clip.mp4"), frames_in, tmp_path / "frames-out")

    assert onnx_video.calls == []
    assert upscaler.ncnn_calls == 1


async def test_auto_dispatches_builtin_to_ncnn_when_no_vendored_onnx(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    onnx_video = FakeOnnxVideoEngine(available=True, gpu_ep=True, builtin_available=False)
    upscaler = make_upscaler(settings, onnx_video_engine=onnx_video)
    frames_in = make_frames_in(tmp_path)

    await upscaler._upscale_frames(make_video_job(tmp_path / "clip.mp4"), frames_in, tmp_path / "frames-out")

    assert onnx_video.calls == []
    assert upscaler.ncnn_calls == 1


async def test_no_onnx_video_engine_configured_falls_back_to_ncnn(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    upscaler = make_upscaler(settings, onnx_video_engine=None)
    frames_in = make_frames_in(tmp_path)

    await upscaler._upscale_frames(make_video_job(tmp_path / "clip.mp4"), frames_in, tmp_path / "frames-out")

    assert upscaler.ncnn_calls == 1


async def test_setting_ncnn_forces_ncnn_even_when_onnx_available(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, UPSCALE_BACKEND=UPSCALE_BACKEND_NCNN)
    onnx_video = FakeOnnxVideoEngine(available=True, gpu_ep=True, builtin_available=True)
    upscaler = make_upscaler(settings, onnx_video_engine=onnx_video)
    frames_in = make_frames_in(tmp_path)

    await upscaler._upscale_frames(make_video_job(tmp_path / "clip.mp4"), frames_in, tmp_path / "frames-out")

    assert onnx_video.calls == []
    assert upscaler.ncnn_calls == 1


async def test_job_backend_override_onnx_beats_setting_ncnn(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, UPSCALE_BACKEND=UPSCALE_BACKEND_NCNN)
    onnx_video = FakeOnnxVideoEngine(available=True, gpu_ep=True, builtin_available=True)
    upscaler = make_upscaler(settings, onnx_video_engine=onnx_video)
    frames_in = make_frames_in(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4", backend=UPSCALE_BACKEND_ONNX)

    await upscaler._upscale_frames(job, frames_in, tmp_path / "frames-out")

    assert len(onnx_video.calls) == 1
    assert upscaler.ncnn_calls == 0


async def test_hf_onnx_model_routes_to_existing_onnx_engine_not_builtin(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())
    hf_onnx = FakeHfOnnxEngine()
    onnx_video = FakeOnnxVideoEngine(available=True, gpu_ep=True, builtin_available=True)
    upscaler = make_upscaler(settings, onnx_video_engine=onnx_video, onnx_engine=hf_onnx, registry=registry)
    frames_in = make_frames_in(tmp_path)
    job = make_video_job(tmp_path / "clip.mp4", model_name="fake-onnx-2x", model_id="fake-onnx-2x", device="cpu")

    await upscaler._upscale_frames(job, frames_in, tmp_path / "frames-out")

    assert len(hf_onnx.calls) == 1
    assert onnx_video.calls == []
    assert upscaler.ncnn_calls == 0


# ---------------------------------------------------------------------------
# Route: per-job backend override honored / validated
# ---------------------------------------------------------------------------


class FakeSimpleVideoUpscaler:
    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        return job.source_path


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


def make_video_manager(settings: Settings) -> VideoJobManager:
    return VideoJobManager(
        settings,
        FakeSimpleVideoUpscaler(),
        FakeMediaTools(),
        DeviceSemaphores(settings),
        registry=ModelRegistry(settings),
        devices=FakeDevicesService(),
    )


async def _create_video_job_route(settings: Settings, manager: VideoJobManager, backend: str | None):
    return await create_video_job(
        request=None,
        file=make_upload("clip.mp4", b"fake-video-bytes"),
        profile_key="anime-balanced-2x",
        model_name=None,
        scale=None,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
        fps_multiplier=None,
        target_fps=None,
        audio_enhance=None,
        audio_restore=None,
        model_id=None,
        device="dml:0",
        backend=backend,
        video_jobs=manager,
        storage=StorageService(settings),
        settings=settings,
        devices=FakeDevicesService(),
    )


async def test_route_honors_valid_backend_override(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_manager(settings)

    await manager.start()
    try:
        response = await _create_video_job_route(settings, manager, backend=UPSCALE_BACKEND_ONNX)
        await manager.queue.join()
    finally:
        await manager.stop()

    job = manager.get_job(response.job_id)
    assert job is not None
    assert job.backend == UPSCALE_BACKEND_ONNX


async def test_route_rejects_invalid_backend_with_400(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_manager(settings)

    await manager.start()
    try:
        with pytest.raises(HTTPException) as exc_info:
            await _create_video_job_route(settings, manager, backend="cuda")
    finally:
        await manager.stop()

    assert exc_info.value.status_code == 400
    assert "backend must be one of" in str(exc_info.value.detail)


def test_video_job_response_exposes_backend() -> None:
    from app.api.routes import video_job_to_response

    job = make_video_job(Path("clip.mp4"), backend=UPSCALE_BACKEND_ONNX)
    response = video_job_to_response(job)

    assert response.backend == UPSCALE_BACKEND_ONNX
    assert response.model_dump(by_alias=True)["backend"] == UPSCALE_BACKEND_ONNX
