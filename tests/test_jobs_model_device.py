from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from app.api.routes import create_job, create_video_job, job_to_response, video_job_to_response
from app.config import Settings
from app.models import JobStatus, UpscaleJob, VideoUpscaleJob
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager
from app.services.video_upscaler import VideoUpscaler

# ---------------------------------------------------------------------------
# SP1 Task 7 - routing image/video jobs by model_id + device.
#
# No real onnxruntime/binaries here: FakeNcnnEngine/FakeOnnxEngine (image) and
# FakeVideoNcnnEngine/FakeOnnxRunFramesEngine (video, via a StageTracking
# VideoUpscaler that fakes _run_process the same way tests/test_pipeline_
# stage_order.py does) stand in for the real engines. FakeDevicesService
# avoids touching real hardware enumeration.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def make_png_bytes(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def write_source_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_png_bytes())


def write_source_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-video-bytes")


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


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


class FakeDevicesService:
    """Minimal stand-in covering the surface JobManager/VideoJobManager/routes touch."""

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


class ConcurrencyTracker:
    def __init__(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)

    async def exit(self) -> None:
        async with self._lock:
            self.in_flight -= 1


# ---------------------------------------------------------------------------
# Image fakes
# ---------------------------------------------------------------------------


class FakeNcnnEngine(UpscaleEngine):
    def __init__(self) -> None:
        self.calls: list[UpscaleJob] = []

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        self.calls.append(job)
        output_path = job.source_path.parent / f"{job.id}-ncnn-out.png"
        output_path.write_bytes(b"fake-ncnn-output")
        return output_path


class FakeOnnxEngine(UpscaleEngine):
    def __init__(self) -> None:
        self.calls: list[UpscaleJob] = []

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        self.calls.append(job)
        output_path = job.source_path.parent / f"{job.id}-onnx-out.png"
        output_path.write_bytes(b"fake-onnx-output")
        return output_path


# ---------------------------------------------------------------------------
# Video fakes
# ---------------------------------------------------------------------------


class FakeVideoNcnnEngine:
    def available(self) -> bool:
        return True


class FakeVideoMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [{"codec_type": "video", "width": 4, "height": 4, "avg_frame_rate": "30/1"}],
            "format": {"duration": "1.0"},
        }


class FakeSimpleVideoUpscaler:
    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        return job.source_path


class FakeOnnxRunFramesEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, str, str]] = []

    async def run_frames(self, frames_in: Path, frames_out: Path, model_id: str, device: str) -> Path:
        self.calls.append((frames_in, frames_out, model_id, device))
        frames_out.mkdir(parents=True, exist_ok=True)
        for frame in sorted(frames_in.glob("*.png")):
            (frames_out / frame.name).write_bytes(b"fake-onnx-frame")
        return frames_out


class StageTrackingVideoUpscaler(VideoUpscaler):
    """Fakes _run_process so no real ffmpeg/ncnn binary runs; records stage order."""

    def __init__(self, *args: object, events: list[str], **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.events = events

    async def _run_process(self, command: list[str]) -> None:
        if "-vsync" in command:
            self.events.append("extract")
            self._write_dummy_frame(command)
        elif command[0] == str(self.settings.engine_binary_path):
            self.events.append("upscale_ncnn")
            self._write_dummy_upscaled_frame(command)
        elif "-framerate" in command:
            self.events.append("encode")
            self._write_dummy_output(command)

    @staticmethod
    def _write_dummy_frame(command: list[str]) -> None:
        frames_in_dir = Path(command[-1]).parent
        frames_in_dir.mkdir(parents=True, exist_ok=True)
        (frames_in_dir / "00000001.png").write_bytes(b"fake-frame-in")

    @staticmethod
    def _write_dummy_upscaled_frame(command: list[str]) -> None:
        frames_out_dir = Path(command[command.index("-o") + 1])
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        (frames_out_dir / "00000001.png").write_bytes(b"fake-frame-out")

    @staticmethod
    def _write_dummy_output(command: list[str]) -> None:
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-output-video")


def make_video_upscaler(
    settings: Settings, events: list[str], registry: ModelRegistry, onnx_engine: object | None
) -> StageTrackingVideoUpscaler:
    return StageTrackingVideoUpscaler(
        settings,
        FakeVideoNcnnEngine(),
        FakeVideoMediaTools(),
        None,
        None,
        onnx_engine=onnx_engine,
        model_registry=registry,
        events=events,
    )


def make_video_job(source_path: Path, **overrides: object) -> VideoUpscaleJob:
    fields = dict(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )
    fields.update(overrides)
    return VideoUpscaleJob(**fields)


# ---------------------------------------------------------------------------
# Image routing: builtin -> ncnn engine
# ---------------------------------------------------------------------------


async def test_job_manager_routes_builtin_model_to_ncnn_engine(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    ncnn_engine = FakeNcnnEngine()
    onnx_engine = FakeOnnxEngine()
    manager = JobManager(
        settings,
        ncnn_engine,
        asyncio.Semaphore(1),
        onnx_engine=onnx_engine,
        registry=ModelRegistry(settings),
        devices=FakeDevicesService(),
    )
    source_path = settings.uploads_path / "photo.png"
    write_source_image(source_path)

    await manager.start()
    try:
        job = await manager.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
            device="dml:0",
        )
        await manager.queue.join()
    finally:
        await manager.stop()

    assert len(ncnn_engine.calls) == 1
    assert onnx_engine.calls == []
    assert job.status == JobStatus.completed
    assert job.model_id == "realesrgan-x4plus"
    assert job.device == "dml:0"


async def test_job_manager_model_name_back_compat_maps_to_model_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeNcnnEngine(), asyncio.Semaphore(1), devices=FakeDevicesService())
    source_path = settings.uploads_path / "photo.png"
    write_source_image(source_path)

    job = await manager.create_job(
        source_path=source_path,
        original_filename="photo.png",
        model_name="realesrgan-x4plus-anime",
        scale=4,
        output_format="png",
        device="dml:0",
    )

    assert job.model_id == "realesrgan-x4plus-anime"
    assert job.model_name == "realesrgan-x4plus-anime"


# ---------------------------------------------------------------------------
# Image routing: onnx -> OnnxUpscaler
# ---------------------------------------------------------------------------


async def test_job_manager_routes_onnx_model_id_to_onnx_engine(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())
    ncnn_engine = FakeNcnnEngine()
    onnx_engine = FakeOnnxEngine()
    manager = JobManager(
        settings,
        ncnn_engine,
        asyncio.Semaphore(1),
        onnx_engine=onnx_engine,
        registry=registry,
        devices=FakeDevicesService(),
    )
    source_path = settings.uploads_path / "photo.png"
    write_source_image(source_path)

    await manager.start()
    try:
        job = await manager.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            model_id="fake-onnx-2x",
            scale=2,
            output_format="png",
            device="cpu",
        )
        await manager.queue.join()
    finally:
        await manager.stop()

    assert len(onnx_engine.calls) == 1
    assert ncnn_engine.calls == []
    assert job.status == JobStatus.completed
    assert job.model_id == "fake-onnx-2x"
    assert job.device == "cpu"


async def test_job_manager_onnx_model_accepts_cpu_device(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())
    manager = JobManager(
        settings,
        FakeNcnnEngine(),
        asyncio.Semaphore(1),
        onnx_engine=FakeOnnxEngine(),
        registry=registry,
        devices=FakeDevicesService(),
    )
    source_path = settings.uploads_path / "photo.png"
    write_source_image(source_path)

    job = await manager.create_job(
        source_path=source_path,
        original_filename="photo.png",
        model_name="realesrgan-x4plus",
        model_id="fake-onnx-2x",
        scale=2,
        output_format="png",
        device="cpu",
    )

    assert job.model_id == "fake-onnx-2x"
    assert job.device == "cpu"


# ---------------------------------------------------------------------------
# Image: validation errors
# ---------------------------------------------------------------------------


async def test_job_manager_create_job_rejects_unknown_model_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(
        settings,
        FakeNcnnEngine(),
        asyncio.Semaphore(1),
        registry=ModelRegistry(settings),
        devices=FakeDevicesService(),
    )
    source_path = settings.uploads_path / "photo.png"
    write_source_image(source_path)

    with pytest.raises(ValueError, match="Unknown model id"):
        await manager.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            model_id="does-not-exist",
            scale=4,
            output_format="png",
            device="dml:0",
        )


async def test_job_manager_rejects_cpu_device_for_builtin_ncnn_model(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeNcnnEngine(), asyncio.Semaphore(1), devices=FakeDevicesService())
    source_path = settings.uploads_path / "photo.png"
    write_source_image(source_path)

    with pytest.raises(ValueError, match="(?i)cpu.*not supported"):
        await manager.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
            device="cpu",
        )


async def test_job_manager_create_job_rejects_unknown_device(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = JobManager(settings, FakeNcnnEngine(), asyncio.Semaphore(1), devices=FakeDevicesService())
    source_path = settings.uploads_path / "photo.png"
    write_source_image(source_path)

    with pytest.raises(ValueError, match="Unknown device id"):
        await manager.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
            device="totally-fake-device",
        )


# ---------------------------------------------------------------------------
# Image: route-level wiring end to end
# ---------------------------------------------------------------------------


async def test_create_job_route_routes_explicit_model_id_to_onnx_engine(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())
    onnx_engine = FakeOnnxEngine()
    manager = JobManager(
        settings,
        FakeNcnnEngine(),
        asyncio.Semaphore(1),
        onnx_engine=onnx_engine,
        registry=registry,
        devices=FakeDevicesService(),
    )

    response = await create_job(
        request=None,
        file=make_upload("photo.png", make_png_bytes()),
        model_name="realesrgan-x4plus",
        model_id="fake-onnx-2x",
        device=None,
        scale=2,
        output_format="png",
        jobs=manager,
        storage=storage,
        settings=settings,
        devices=FakeDevicesService(),
    )

    job = manager.get_job(response.job_id)
    assert job is not None
    assert job.model_id == "fake-onnx-2x"
    assert job.device == "dml:0"


# ---------------------------------------------------------------------------
# Image: response exposes modelId/device
# ---------------------------------------------------------------------------


def test_job_to_response_exposes_model_id_and_device(tmp_path: Path) -> None:
    job = UpscaleJob(
        source_path=tmp_path / "a.png",
        original_filename="a.png",
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
        model_id="realesrgan-x4plus",
        device="dml:0",
    )

    response = job_to_response(job)

    assert response.model_id == "realesrgan-x4plus"
    assert response.device == "dml:0"
    serialized = response.model_dump(by_alias=True)
    assert serialized["modelId"] == "realesrgan-x4plus"
    assert serialized["device"] == "dml:0"


# ---------------------------------------------------------------------------
# Video routing: builtin -> ncnn subprocess stage
# ---------------------------------------------------------------------------


async def test_video_upscaler_routes_builtin_model_to_ncnn_subprocess(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    events: list[str] = []
    onnx_engine = FakeOnnxRunFramesEngine()
    upscaler = make_video_upscaler(settings, events, ModelRegistry(settings), onnx_engine)
    source_path = settings.uploads_path / "clip.mp4"
    write_source_video(source_path)
    job = make_video_job(source_path)

    await upscaler.run(job)

    assert events == ["extract", "upscale_ncnn", "encode"]
    assert onnx_engine.calls == []


# ---------------------------------------------------------------------------
# Video routing: onnx -> OnnxUpscaler.run_frames
# ---------------------------------------------------------------------------


async def test_video_upscaler_routes_onnx_model_to_run_frames(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())
    events: list[str] = []
    onnx_engine = FakeOnnxRunFramesEngine()
    upscaler = make_video_upscaler(settings, events, registry, onnx_engine)
    source_path = settings.uploads_path / "clip.mp4"
    write_source_video(source_path)
    job = make_video_job(source_path, model_name="fake-onnx-2x", model_id="fake-onnx-2x", device="cpu")

    output_path = await upscaler.run(job)

    assert events == ["extract", "encode"]
    assert "upscale_ncnn" not in events
    assert len(onnx_engine.calls) == 1
    _frames_in, _frames_out, model_id_arg, device_arg = onnx_engine.calls[0]
    assert model_id_arg == "fake-onnx-2x"
    assert device_arg == "cpu"
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Video: route-level wiring end to end (create_video_job -> VideoJobManager ->
# VideoUpscaler -> run_frames). Analogous to the image
# test_create_job_route_routes_explicit_model_id_to_onnx_engine, but this one
# runs the worker so it exercises the subtlety unique to video: the route's
# resolve_video_job_fields computes model_name from the profile default
# (realesr-animevideov3-x2, a builtin) INDEPENDENTLY of model_id, and only
# VideoJobManager.create_job's own resolution overrides it with the onnx
# model_id. A regression there would silently route to the ncnn subprocess
# (or 400 on cpu), which the assertions below catch RED.
# ---------------------------------------------------------------------------


async def test_create_video_job_route_routes_explicit_model_id_to_onnx_run_frames(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())
    events: list[str] = []
    onnx_engine = FakeOnnxRunFramesEngine()
    upscaler = make_video_upscaler(settings, events, registry, onnx_engine)
    video_jobs = VideoJobManager(
        settings,
        upscaler,
        FakeVideoMediaTools(),
        asyncio.Semaphore(1),
        registry=registry,
        devices=FakeDevicesService(),
    )

    await video_jobs.start()
    try:
        response = await create_video_job(
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
            model_id="fake-onnx-2x",
            device="cpu",
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        )
        await video_jobs.queue.join()
    finally:
        await video_jobs.stop()

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert job.status == JobStatus.completed
    assert job.model_id == "fake-onnx-2x"
    assert job.device == "cpu"
    # Routed to onnx run_frames, NOT the ncnn subprocess (the profile default
    # realesr-animevideov3-x2 must not win over the explicit onnx model_id).
    assert len(onnx_engine.calls) == 1
    assert "upscale_ncnn" not in events
    _frames_in, _frames_out, model_id_arg, device_arg = onnx_engine.calls[0]
    assert model_id_arg == "fake-onnx-2x"
    assert device_arg == "cpu"
    # Response (GET shape) exposes modelId/device.
    serialized = video_job_to_response(job).model_dump(by_alias=True)
    assert serialized["modelId"] == "fake-onnx-2x"
    assert serialized["device"] == "cpu"


# ---------------------------------------------------------------------------
# Video: VideoJobManager model resolution / validation
# ---------------------------------------------------------------------------


async def test_video_job_manager_model_name_back_compat_maps_to_model_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = VideoJobManager(
        settings,
        FakeSimpleVideoUpscaler(),
        FakeVideoMediaTools(),
        asyncio.Semaphore(1),
        registry=ModelRegistry(settings),
        devices=FakeDevicesService(),
    )
    source_path = settings.uploads_path / "clip.mp4"
    write_source_video(source_path)

    job = await manager.create_job(
        source_path=source_path,
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        device="dml:0",
    )

    assert job.model_id == "realesr-animevideov3-x2"


async def test_video_job_manager_rejects_unknown_model_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = VideoJobManager(
        settings,
        FakeSimpleVideoUpscaler(),
        FakeVideoMediaTools(),
        asyncio.Semaphore(1),
        registry=ModelRegistry(settings),
        devices=FakeDevicesService(),
    )
    source_path = settings.uploads_path / "clip.mp4"
    write_source_video(source_path)

    with pytest.raises(ValueError, match="Unknown model id"):
        await manager.create_job(
            source_path=source_path,
            original_filename="clip.mp4",
            model_name="realesr-animevideov3-x2",
            model_id="does-not-exist",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=False,
            device="dml:0",
        )


async def test_video_job_manager_rejects_cpu_device_for_builtin_ncnn_model(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = VideoJobManager(
        settings,
        FakeSimpleVideoUpscaler(),
        FakeVideoMediaTools(),
        asyncio.Semaphore(1),
        devices=FakeDevicesService(),
    )
    source_path = settings.uploads_path / "clip.mp4"
    write_source_video(source_path)

    with pytest.raises(ValueError, match="(?i)cpu.*not supported"):
        await manager.create_job(
            source_path=source_path,
            original_filename="clip.mp4",
            model_name="realesr-animevideov3-x2",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=False,
            device="cpu",
        )


async def test_video_job_manager_rejects_unknown_device(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = VideoJobManager(
        settings,
        FakeSimpleVideoUpscaler(),
        FakeVideoMediaTools(),
        asyncio.Semaphore(1),
        devices=FakeDevicesService(),
    )
    source_path = settings.uploads_path / "clip.mp4"
    write_source_video(source_path)

    with pytest.raises(ValueError, match="Unknown device id"):
        await manager.create_job(
            source_path=source_path,
            original_filename="clip.mp4",
            model_name="realesr-animevideov3-x2",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=False,
            device="totally-fake-device",
        )


# ---------------------------------------------------------------------------
# Video: response exposes modelId/device
# ---------------------------------------------------------------------------


def test_video_job_to_response_exposes_model_id_and_device() -> None:
    job = make_video_job(Path("clip.mp4"), model_id="realesr-animevideov3-x2", device="dml:0")

    response = video_job_to_response(job)

    assert response.model_id == "realesr-animevideov3-x2"
    assert response.device == "dml:0"
    serialized = response.model_dump(by_alias=True)
    assert serialized["modelId"] == "realesr-animevideov3-x2"
    assert serialized["device"] == "dml:0"


# ---------------------------------------------------------------------------
# Shared GPU semaphore must cover the onnx routing path too, for both
# managers at once (mirrors tests/test_concurrency.py's cross-manager test).
# ---------------------------------------------------------------------------


async def test_shared_semaphore_gates_onnx_routed_image_and_video_jobs_together(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, GPU_CONCURRENCY=1)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())
    tracker = ConcurrencyTracker()

    class TrackingOnnxEngine(UpscaleEngine):
        def available(self) -> bool:
            return True

        async def run(self, job: UpscaleJob) -> Path:
            await tracker.enter()
            try:
                await asyncio.sleep(0.05)
                output_path = settings.outputs_path / f"{job.id}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"fake")
                return output_path
            finally:
                await tracker.exit()

    class TrackingOnnxRunFramesEngine:
        async def run_frames(self, frames_in: Path, frames_out: Path, model_id: str, device: str) -> Path:
            await tracker.enter()
            try:
                await asyncio.sleep(0.05)
                frames_out.mkdir(parents=True, exist_ok=True)
                for frame in sorted(frames_in.glob("*.png")):
                    (frames_out / frame.name).write_bytes(b"fake")
                return frames_out
            finally:
                await tracker.exit()

    events: list[str] = []
    gpu_semaphore = asyncio.Semaphore(1)

    image_manager = JobManager(
        settings,
        FakeNcnnEngine(),
        gpu_semaphore,
        onnx_engine=TrackingOnnxEngine(),
        registry=registry,
        devices=FakeDevicesService(),
    )
    video_upscaler = make_video_upscaler(settings, events, registry, TrackingOnnxRunFramesEngine())
    video_manager = VideoJobManager(
        settings, video_upscaler, FakeVideoMediaTools(), gpu_semaphore, registry=registry, devices=FakeDevicesService()
    )

    image_source = settings.uploads_path / "photo.png"
    write_source_image(image_source)
    video_source = settings.uploads_path / "clip.mp4"
    write_source_video(video_source)

    await image_manager.start()
    await video_manager.start()
    try:
        await asyncio.gather(
            image_manager.create_job(
                source_path=image_source,
                original_filename="photo.png",
                model_name="realesrgan-x4plus",
                model_id="fake-onnx-2x",
                scale=2,
                output_format="png",
                device="cpu",
            ),
            video_manager.create_job(
                source_path=video_source,
                original_filename="clip.mp4",
                model_name="realesr-animevideov3-x2",
                model_id="fake-onnx-2x",
                scale=2,
                output_container="mp4",
                video_codec="libx264",
                video_preset="medium",
                crf=18,
                keep_audio=False,
                device="cpu",
            ),
        )
        await image_manager.queue.join()
        await video_manager.queue.join()
    finally:
        await image_manager.stop()
        await video_manager.stop()

    assert tracker.max_in_flight == 1, "onnx-routed image and video jobs overlapped despite a shared semaphore of 1"
