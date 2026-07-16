from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from app.api.routes import create_job, resolve_request_device
from app.config import Settings
from app.models import JobStatus, UpscaleJob, VideoUpscaleJob
from app.services.device_router import DeviceRouter
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DeviceInfo
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

# ---------------------------------------------------------------------------
# SP7 Task 2 - the optional auto-router, exercised through JobManager /
# VideoJobManager / the create_job route, the same way SP7 T1's
# tests/test_multigpu_concurrency.py exercises DeviceSemaphores end to end
# with a fake engine instead of real ncnn/onnxruntime binaries.
# ---------------------------------------------------------------------------

CPU: DeviceInfo = {"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"}
GPU0: DeviceInfo = {"id": "dml:0", "kind": "gpu", "name": "GPU 0", "backend": "directml"}
GPU1: DeviceInfo = {"id": "dml:1", "kind": "gpu", "name": "GPU 1", "backend": "directml"}

HOLD_SECONDS = 0.08


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path)}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def make_png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buffer, format="PNG")
    return buffer.getvalue()


def write_source_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(make_png_bytes())


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
    """Minimal stand-in exposing real DeviceInfo shapes (id + kind), needed
    by device_router's compatibility filtering -- unlike the lighter fake in
    tests/test_jobs_model_device.py, which only carries "id"."""

    def __init__(self, devices: list[DeviceInfo]) -> None:
        self._devices = devices

    def list_devices(self) -> list[DeviceInfo]:
        return list(self._devices)

    def resolve_default(self, devices: list[DeviceInfo] | None = None) -> DeviceInfo:
        return (devices if devices is not None else self._devices)[-1]

    def validate(self, device_id: str) -> DeviceInfo:
        for device in self._devices:
            if device["id"] == device_id:
                return device
        raise ValueError(f"Unknown device id: {device_id!r}")


class DeviceTimestampRecorder:
    def __init__(self) -> None:
        self.intervals: list[tuple[str | None, float, float]] = []
        self._lock = asyncio.Lock()

    async def record(self, device_id: str | None, hold_seconds: float = HOLD_SECONDS) -> None:
        start = time.monotonic()
        await asyncio.sleep(hold_seconds)
        end = time.monotonic()
        async with self._lock:
            self.intervals.append((device_id, start, end))

    def devices_used(self) -> list[str | None]:
        return [device_id for device_id, _, _ in self.intervals]

    def overlaps(self, device_a: str | None, device_b: str | None) -> bool:
        (start_a, end_a) = next((s, e) for d, s, e in self.intervals if d == device_a)
        (start_b, end_b) = next((s, e) for d, s, e in self.intervals if d == device_b)
        return start_a < end_b and start_b < end_a


class RecordingNcnnEngine(UpscaleEngine):
    def __init__(self, recorder: DeviceTimestampRecorder, hold_seconds: float = HOLD_SECONDS) -> None:
        self.recorder = recorder
        self.hold_seconds = hold_seconds

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        await self.recorder.record(job.device, self.hold_seconds)
        output_path = job.source_path.parent / f"{job.id}-out.png"
        output_path.write_bytes(b"fake-output")
        return output_path


class RecordingOnnxEngine(UpscaleEngine):
    def __init__(self, recorder: DeviceTimestampRecorder, hold_seconds: float = HOLD_SECONDS) -> None:
        self.recorder = recorder
        self.hold_seconds = hold_seconds

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        await self.recorder.record(job.device, self.hold_seconds)
        output_path = job.source_path.parent / f"{job.id}-onnx-out.png"
        output_path.write_bytes(b"fake-onnx-output")
        return output_path


class FakeSimpleVideoUpscaler:
    def __init__(self, recorder: DeviceTimestampRecorder, hold_seconds: float = HOLD_SECONDS) -> None:
        self.recorder = recorder
        self.hold_seconds = hold_seconds

    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        await self.recorder.record(job.device, self.hold_seconds)
        return job.source_path


class FakeVideoMediaTools:
    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video", "avg_frame_rate": "30/1"}]}


def make_image_source(settings: Settings, name: str) -> Path:
    source_path = settings.uploads_path / name
    write_source_image(source_path)
    return source_path


# ---------------------------------------------------------------------------
# create_job validation: "auto" accepted / rejected at request time
# ---------------------------------------------------------------------------


async def test_job_manager_accepts_auto_device_when_compatible_device_exists(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    recorder = DeviceTimestampRecorder()
    manager = JobManager(
        settings,
        RecordingNcnnEngine(recorder),
        DeviceSemaphores(settings),
        devices=FakeDevicesService([CPU, GPU0]),
    )
    source_path = make_image_source(settings, "photo.png")

    job = await manager.create_job(
        source_path=source_path,
        original_filename="photo.png",
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
        device=AUTO_DEVICE_ID,
    )

    assert job.device == AUTO_DEVICE_ID, "device stays the sentinel until the worker resolves it at dequeue"


async def test_job_manager_rejects_auto_device_when_ncnn_and_only_cpu_available(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    recorder = DeviceTimestampRecorder()
    manager = JobManager(
        settings,
        RecordingNcnnEngine(recorder),
        DeviceSemaphores(settings),
        devices=FakeDevicesService([CPU]),
    )
    source_path = make_image_source(settings, "photo.png")

    with pytest.raises(ValueError, match="No compatible device"):
        await manager.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
            device=AUTO_DEVICE_ID,
        )


# ---------------------------------------------------------------------------
# Worker resolution: compatibility + least-loaded pick
# ---------------------------------------------------------------------------


async def test_auto_ncnn_job_never_resolves_to_cpu(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    recorder = DeviceTimestampRecorder()
    manager = JobManager(
        settings,
        RecordingNcnnEngine(recorder),
        DeviceSemaphores(settings),
        devices=FakeDevicesService([CPU, GPU0]),
    )
    source_path = make_image_source(settings, "photo.png")

    await manager.start()
    try:
        job = await manager.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
            device=AUTO_DEVICE_ID,
        )
        await manager.queue.join()
    finally:
        await manager.stop()

    assert job.status == JobStatus.completed
    assert job.device == "dml:0"


async def test_auto_onnx_job_can_resolve_to_cpu_when_its_the_only_device(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())
    recorder = DeviceTimestampRecorder()
    manager = JobManager(
        settings,
        RecordingNcnnEngine(recorder),
        DeviceSemaphores(settings),
        onnx_engine=RecordingOnnxEngine(recorder),
        registry=registry,
        devices=FakeDevicesService([CPU]),
    )
    source_path = make_image_source(settings, "photo.png")

    await manager.start()
    try:
        job = await manager.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            model_id="fake-onnx-2x",
            scale=2,
            output_format="png",
            device=AUTO_DEVICE_ID,
        )
        await manager.queue.join()
    finally:
        await manager.stop()

    assert job.status == JobStatus.completed
    assert job.device == "cpu"


async def test_auto_picks_the_free_gpu_when_the_other_is_busy(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MAX_CONCURRENT_JOBS=4)
    recorder = DeviceTimestampRecorder()
    manager = JobManager(
        settings,
        RecordingNcnnEngine(recorder, hold_seconds=0.12),
        DeviceSemaphores(settings),
        devices=FakeDevicesService([GPU0, GPU1]),
    )
    pinned_source = make_image_source(settings, "pinned.png")
    auto_source = make_image_source(settings, "auto.png")

    await manager.start()
    try:
        await manager.create_job(
            source_path=pinned_source,
            original_filename="pinned.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
            device="dml:0",
        )
        await asyncio.sleep(0.02)  # let the pinned job grab dml:0 first
        auto_job = await manager.create_job(
            source_path=auto_source,
            original_filename="auto.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
            device=AUTO_DEVICE_ID,
        )
        await manager.queue.join()
    finally:
        await manager.stop()

    assert auto_job.device == "dml:1", "dml:0 was busy; auto must route to the idle dml:1"
    assert recorder.overlaps("dml:0", "dml:1"), "the auto job must not wait behind the pinned dml:0 job"


async def test_two_concurrent_auto_jobs_distribute_across_two_free_gpus(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MAX_CONCURRENT_JOBS=4)
    recorder = DeviceTimestampRecorder()
    manager = JobManager(
        settings,
        RecordingNcnnEngine(recorder),
        DeviceSemaphores(settings),
        devices=FakeDevicesService([GPU0, GPU1]),
    )
    source_a = make_image_source(settings, "photo-a.png")
    source_b = make_image_source(settings, "photo-b.png")

    await manager.start()
    try:
        await asyncio.gather(
            manager.create_job(
                source_path=source_a,
                original_filename="photo-a.png",
                model_name="realesrgan-x4plus",
                scale=4,
                output_format="png",
                device=AUTO_DEVICE_ID,
            ),
            manager.create_job(
                source_path=source_b,
                original_filename="photo-b.png",
                model_name="realesrgan-x4plus",
                scale=4,
                output_format="png",
                device=AUTO_DEVICE_ID,
            ),
        )
        await manager.queue.join()
    finally:
        await manager.stop()

    assert sorted(recorder.devices_used()) == ["dml:0", "dml:1"], (
        f"two auto jobs with two free GPUs must not both land on the same device, got {recorder.devices_used()}"
    )
    assert recorder.overlaps("dml:0", "dml:1"), "the two auto jobs should also run in parallel"


async def test_router_off_respects_the_pinned_device_on_the_job(tmp_path: Path) -> None:
    """Regression guard: a job that pins a concrete device (not "auto") must
    be completely unaffected by the router/DeviceRouter machinery."""
    settings = make_settings(tmp_path)
    recorder = DeviceTimestampRecorder()
    manager = JobManager(
        settings,
        RecordingNcnnEngine(recorder),
        DeviceSemaphores(settings),
        devices=FakeDevicesService([CPU, GPU0, GPU1]),
    )
    source_path = make_image_source(settings, "photo.png")

    await manager.start()
    try:
        job = await manager.create_job(
            source_path=source_path,
            original_filename="photo.png",
            model_name="realesrgan-x4plus",
            scale=4,
            output_format="png",
            device="dml:1",
        )
        await manager.queue.join()
    finally:
        await manager.stop()

    assert job.status == JobStatus.completed
    assert job.device == "dml:1"


# ---------------------------------------------------------------------------
# VideoJobManager mirrors the same worker-level auto resolution
# ---------------------------------------------------------------------------


async def test_video_job_manager_auto_ncnn_job_never_resolves_to_cpu(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    recorder = DeviceTimestampRecorder()
    manager = VideoJobManager(
        settings,
        FakeSimpleVideoUpscaler(recorder),
        FakeVideoMediaTools(),
        DeviceSemaphores(settings),
        devices=FakeDevicesService([CPU, GPU0]),
    )
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    await manager.start()
    try:
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
            device=AUTO_DEVICE_ID,
        )
        await manager.queue.join()
    finally:
        await manager.stop()

    assert job.status == JobStatus.completed
    assert job.device == "dml:0"


async def test_video_job_manager_rejects_auto_device_when_no_compatible_device(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    recorder = DeviceTimestampRecorder()
    manager = VideoJobManager(
        settings,
        FakeSimpleVideoUpscaler(recorder),
        FakeVideoMediaTools(),
        DeviceSemaphores(settings),
        devices=FakeDevicesService([CPU]),
    )
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    with pytest.raises(ValueError, match="No compatible device"):
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
            device=AUTO_DEVICE_ID,
        )


# ---------------------------------------------------------------------------
# routes.py: resolve_request_device + ENABLE_AUTO_ROUTE toggle
# ---------------------------------------------------------------------------


async def test_resolve_request_device_passes_through_explicit_auto_regardless_of_toggle(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ENABLE_AUTO_ROUTE=False)
    devices = FakeDevicesService([CPU, GPU0])

    resolved = await resolve_request_device(AUTO_DEVICE_ID, devices, settings)

    assert resolved == AUTO_DEVICE_ID


async def test_resolve_request_device_defaults_normally_when_toggle_off_and_device_unset(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ENABLE_AUTO_ROUTE=False, DEFAULT_DEVICE="dml:0")
    devices = FakeDevicesService([CPU, GPU0])

    resolved = await resolve_request_device(None, devices, settings)

    assert resolved == "dml:0"


async def test_resolve_request_device_returns_auto_sentinel_when_toggle_on_and_device_unset(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ENABLE_AUTO_ROUTE=True, DEFAULT_DEVICE="dml:0")
    devices = FakeDevicesService([CPU, GPU0])

    resolved = await resolve_request_device(None, devices, settings)

    assert resolved == AUTO_DEVICE_ID


async def test_create_job_route_rejects_auto_device_with_no_compatible_device(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    recorder = DeviceTimestampRecorder()
    manager = JobManager(
        settings,
        RecordingNcnnEngine(recorder),
        DeviceSemaphores(settings),
        devices=FakeDevicesService([CPU]),
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_job(
            request=None,
            file=make_upload("photo.png", make_png_bytes()),
            model_name="realesrgan-x4plus",
            model_id=None,
            device=AUTO_DEVICE_ID,
            scale=4,
            output_format="png",
            jobs=manager,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService([CPU]),
        )

    assert exc_info.value.status_code == 400
    assert "compatible device" in str(exc_info.value.detail)


async def test_create_job_route_accepts_explicit_auto_device_end_to_end(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    recorder = DeviceTimestampRecorder()
    manager = JobManager(
        settings,
        RecordingNcnnEngine(recorder),
        DeviceSemaphores(settings),
        devices=FakeDevicesService([CPU, GPU0]),
    )

    await manager.start()
    try:
        response = await create_job(
            request=None,
            file=make_upload("photo.png", make_png_bytes()),
            model_name="realesrgan-x4plus",
            model_id=None,
            device=AUTO_DEVICE_ID,
            scale=4,
            output_format="png",
            jobs=manager,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService([CPU, GPU0]),
        )
        await manager.queue.join()
    finally:
        await manager.stop()

    job = manager.get_job(response.job_id)
    assert job is not None
    assert job.status == JobStatus.completed
    assert job.device == "dml:0"


# ---------------------------------------------------------------------------
# Self-review guard: a DeviceRouter shared across two managers (as wired in
# app/main.py) must still atomically distribute auto picks, not just a
# router private to a single manager.
# ---------------------------------------------------------------------------


async def test_shared_device_router_distributes_across_image_and_video_managers(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MAX_CONCURRENT_JOBS=4)
    recorder = DeviceTimestampRecorder()
    device_semaphores = DeviceSemaphores(settings)
    shared_router = DeviceRouter(device_semaphores)
    devices = FakeDevicesService([GPU0, GPU1])

    image_manager = JobManager(
        settings,
        RecordingNcnnEngine(recorder),
        device_semaphores,
        devices=devices,
        device_router=shared_router,
    )
    video_manager = VideoJobManager(
        settings,
        FakeSimpleVideoUpscaler(recorder),
        FakeVideoMediaTools(),
        device_semaphores,
        devices=devices,
        device_router=shared_router,
    )
    image_source = make_image_source(settings, "photo.png")
    video_source = settings.uploads_path / "clip.mp4"
    video_source.write_bytes(b"fake-video-bytes")

    await image_manager.start()
    await video_manager.start()
    try:
        await asyncio.gather(
            image_manager.create_job(
                source_path=image_source,
                original_filename="photo.png",
                model_name="realesrgan-x4plus",
                scale=4,
                output_format="png",
                device=AUTO_DEVICE_ID,
            ),
            video_manager.create_job(
                source_path=video_source,
                original_filename="clip.mp4",
                model_name="realesr-animevideov3-x2",
                scale=2,
                output_container="mp4",
                video_codec="libx264",
                video_preset="medium",
                crf=18,
                keep_audio=False,
                device=AUTO_DEVICE_ID,
            ),
        )
        await image_manager.queue.join()
        await video_manager.queue.join()
    finally:
        await image_manager.stop()
        await video_manager.stop()

    assert sorted(recorder.devices_used()) == ["dml:0", "dml:1"], (
        f"cross-manager auto jobs sharing one DeviceRouter must still land on different devices, "
        f"got {recorder.devices_used()}"
    )
