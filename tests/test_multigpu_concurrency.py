from __future__ import annotations

import asyncio
import io
import time
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.models import UpscaleJob, VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

# ---------------------------------------------------------------------------
# SP7 Task 1 - the reported bug: an image queued on the iGPU (dml:1) sat
# behind a video running on the dGPU (dml:0) because ONE shared
# asyncio.Semaphore gated every GPU job across both managers. DeviceSemaphores
# replaces that with a semaphore PER device_id, so this file proves -- with
# real overlapping in-flight timestamps recorded by a fake engine -- that:
#   * two jobs on DIFFERENT devices now run in parallel
#   * two jobs on the SAME device (per_device capacity=1) still serialize
#   * cpu and gpu jobs run in parallel (independent semaphores)
#   * the exact cross-manager scenario from the bug report (video on dml:0 +
#     image on dml:1) is fixed
# ---------------------------------------------------------------------------

HOLD_SECONDS = 0.1


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path)}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def make_png_bytes(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def make_image_source(settings: Settings, name: str) -> Path:
    source_path = settings.uploads_path / name
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(make_png_bytes())
    return source_path


def make_video_source(settings: Settings, name: str) -> Path:
    source_path = settings.uploads_path / name
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    return source_path


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


class DeviceTimestampRecorder:
    """Records (device_id, start, end) monotonic intervals for every fake
    engine.run() call, so tests can assert overlap (parallel) or
    non-overlap (serial) between specific device_ids."""

    def __init__(self) -> None:
        self.intervals: list[tuple[str | None, float, float]] = []
        self._lock = asyncio.Lock()

    async def record(self, device_id: str | None) -> None:
        start = time.monotonic()
        await asyncio.sleep(HOLD_SECONDS)
        end = time.monotonic()
        async with self._lock:
            self.intervals.append((device_id, start, end))

    def interval_for(self, device_id: str | None) -> tuple[float, float]:
        matches = [(start, end) for recorded_device, start, end in self.intervals if recorded_device == device_id]
        assert matches, f"no recorded interval for device {device_id!r}"
        assert len(matches) == 1, f"expected exactly one interval for device {device_id!r}, got {len(matches)}"
        return matches[0]

    def overlaps(self, device_a: str | None, device_b: str | None) -> bool:
        start_a, end_a = self.interval_for(device_a)
        start_b, end_b = self.interval_for(device_b)
        return start_a < end_b and start_b < end_a


class TrackingImageEngine(UpscaleEngine):
    def __init__(self, settings: Settings, recorder: DeviceTimestampRecorder) -> None:
        self.settings = settings
        self.recorder = recorder

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        await self.recorder.record(job.device)
        output_path = self.settings.outputs_path / f"{job.id}.{job.output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-output-image")
        return output_path


class TrackingVideoUpscaler:
    def __init__(self, recorder: DeviceTimestampRecorder) -> None:
        self.recorder = recorder

    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        await self.recorder.record(job.device)
        return job.source_path


class FakeMediaTools:
    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video"}]}


async def submit_image_job(jobs: JobManager, source_path: Path, device: str, **overrides: object) -> None:
    fields: dict[str, object] = dict(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
        device=device,
    )
    fields.update(overrides)
    await jobs.create_job(**fields)


async def submit_video_job(video_jobs: VideoJobManager, source_path: Path, device: str, **overrides: object) -> None:
    fields: dict[str, object] = dict(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        device=device,
    )
    fields.update(overrides)
    await video_jobs.create_job(**fields)


def make_image_manager(
    settings: Settings, recorder: DeviceTimestampRecorder, device_semaphores: DeviceSemaphores
) -> JobManager:
    engine = TrackingImageEngine(settings, recorder)
    return JobManager(settings, engine, device_semaphores, onnx_engine=engine)


def make_video_manager(
    settings: Settings, recorder: DeviceTimestampRecorder, device_semaphores: DeviceSemaphores
) -> VideoJobManager:
    return VideoJobManager(settings, TrackingVideoUpscaler(recorder), FakeMediaTools(), device_semaphores)


async def test_two_image_jobs_on_different_gpus_run_in_parallel(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MAX_CONCURRENT_JOBS=4)
    StorageService(settings)
    recorder = DeviceTimestampRecorder()
    device_semaphores = DeviceSemaphores(settings)
    jobs = make_image_manager(settings, recorder, device_semaphores)

    source_a = make_image_source(settings, "photo-a.png")
    source_b = make_image_source(settings, "photo-b.png")

    await jobs.start()
    try:
        await asyncio.gather(
            submit_image_job(jobs, source_a, "dml:0"),
            submit_image_job(jobs, source_b, "dml:1"),
        )
        await jobs.queue.join()
    finally:
        await jobs.stop()

    assert recorder.overlaps("dml:0", "dml:1"), "jobs on different GPUs (dml:0, dml:1) must run in parallel"


async def test_two_image_jobs_on_same_gpu_serialize_with_per_device_one(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MAX_CONCURRENT_JOBS=4)
    StorageService(settings)
    recorder = DeviceTimestampRecorder()
    device_semaphores = DeviceSemaphores(settings)
    jobs = make_image_manager(settings, recorder, device_semaphores)

    source_a = make_image_source(settings, "photo-a.png")
    source_b = make_image_source(settings, "photo-b.png")

    await jobs.start()
    try:
        await asyncio.gather(
            submit_image_job(jobs, source_a, "dml:0"),
            submit_image_job(jobs, source_b, "dml:0"),
        )
        await jobs.queue.join()
    finally:
        await jobs.stop()

    same_device_intervals = [
        (start, end) for device_id, start, end in recorder.intervals if device_id == "dml:0"
    ]
    assert len(same_device_intervals) == 2
    (start_a, end_a), (start_b, end_b) = same_device_intervals
    assert not (start_a < end_b and start_b < end_a), (
        "two jobs on the SAME device (dml:0) with per_device capacity=1 must not overlap"
    )


async def test_cpu_and_gpu_jobs_run_in_parallel(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, CPU_CONCURRENCY=1, MAX_CONCURRENT_JOBS=4)
    StorageService(settings)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())
    recorder = DeviceTimestampRecorder()
    device_semaphores = DeviceSemaphores(settings)
    engine = TrackingImageEngine(settings, recorder)
    jobs = JobManager(settings, engine, device_semaphores, onnx_engine=engine, registry=registry)

    gpu_source = make_image_source(settings, "gpu-photo.png")
    cpu_source = make_image_source(settings, "cpu-photo.png")

    await jobs.start()
    try:
        await asyncio.gather(
            submit_image_job(jobs, gpu_source, "dml:0"),
            submit_image_job(
                jobs, cpu_source, "cpu", model_id="fake-onnx-2x", model_name="fake-onnx-2x", scale=2
            ),
        )
        await jobs.queue.join()
    finally:
        await jobs.stop()

    assert recorder.overlaps("dml:0", "cpu"), "a cpu job must not serialize behind a gpu job (independent semaphores)"


async def test_cross_manager_video_on_dml0_and_image_on_dml1_run_in_parallel(tmp_path: Path) -> None:
    """The exact bug report: an image queued on the iGPU (dml:1) must not sit
    behind a video running on the dGPU (dml:0) -- both managers share ONE
    DeviceSemaphores instance, exactly like app/main.py wires it."""
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MAX_CONCURRENT_JOBS=4)
    StorageService(settings)
    recorder = DeviceTimestampRecorder()
    device_semaphores = DeviceSemaphores(settings)
    jobs = make_image_manager(settings, recorder, device_semaphores)
    video_jobs = make_video_manager(settings, recorder, device_semaphores)

    image_source = make_image_source(settings, "photo.png")
    video_source = make_video_source(settings, "clip.mp4")

    await jobs.start()
    await video_jobs.start()
    try:
        await asyncio.gather(
            submit_video_job(video_jobs, video_source, "dml:0"),
            submit_image_job(jobs, image_source, "dml:1"),
        )
        await jobs.queue.join()
        await video_jobs.queue.join()
    finally:
        await jobs.stop()
        await video_jobs.stop()

    assert recorder.overlaps("dml:0", "dml:1"), (
        "video on dml:0 and image on dml:1 must run in parallel across managers (the reported bug)"
    )
