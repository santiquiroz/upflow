from __future__ import annotations

import asyncio
import io
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.models import UpscaleJob, VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import JobManager
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

HOLD_SECONDS = 0.15


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path)}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def make_png_bytes(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class ConcurrencyTracker:
    """Records how many fake GPU jobs were in flight at once, across both managers."""

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


class TrackingImageEngine(UpscaleEngine):
    def __init__(self, settings: Settings, tracker: ConcurrencyTracker) -> None:
        self.settings = settings
        self.tracker = tracker

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        await self.tracker.enter()
        try:
            await asyncio.sleep(HOLD_SECONDS)
            output_path = self.settings.outputs_path / f"{job.id}.{job.output_format}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake-output-image")
            return output_path
        finally:
            await self.tracker.exit()


class TrackingVideoUpscaler:
    def __init__(self, tracker: ConcurrencyTracker) -> None:
        self.tracker = tracker

    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        await self.tracker.enter()
        try:
            await asyncio.sleep(HOLD_SECONDS)
            return job.source_path
        finally:
            await self.tracker.exit()


class FakeMediaTools:
    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video"}]}


def make_image_source(settings: Settings, name: str) -> Path:
    source_path = settings.uploads_path / name
    source_path.write_bytes(make_png_bytes())
    return source_path


def make_video_source(settings: Settings, name: str) -> Path:
    source_path = settings.uploads_path / name
    source_path.write_bytes(b"fake-video-bytes")
    return source_path


async def submit_image_job(jobs: JobManager, source_path: Path, device: str) -> None:
    await jobs.create_job(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
        device=device,
    )


async def submit_video_job(video_jobs: VideoJobManager, source_path: Path, device: str) -> None:
    await video_jobs.create_job(
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


def make_shared_managers(
    settings: Settings, tracker: ConcurrencyTracker
) -> tuple[JobManager, VideoJobManager]:
    device_semaphores = DeviceSemaphores(settings)
    jobs = JobManager(settings, TrackingImageEngine(settings, tracker), device_semaphores)
    video_jobs = VideoJobManager(settings, TrackingVideoUpscaler(tracker), FakeMediaTools(), device_semaphores)
    return jobs, video_jobs


# ---------------------------------------------------------------------------
# SP7 Task 1: these two tests used to prove a single shared asyncio.Semaphore
# gated image+video jobs together across both managers. That semaphore is now
# per-device_id (DeviceSemaphores) -- to keep proving the SAME-device gating
# intent, every job below explicitly targets device="dml:0" so they all
# compete for that one device's semaphore. Cross-device parallelism (the
# actual SP7 fix) is proven separately in tests/test_multigpu_concurrency.py.
# ---------------------------------------------------------------------------


async def test_same_device_concurrency_one_prevents_image_and_video_overlap(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MAX_CONCURRENT_JOBS=4)
    StorageService(settings)
    tracker = ConcurrencyTracker()
    jobs, video_jobs = make_shared_managers(settings, tracker)

    image_source = make_image_source(settings, "photo.png")
    video_source = make_video_source(settings, "clip.mp4")

    await jobs.start()
    await video_jobs.start()
    try:
        await asyncio.gather(
            submit_image_job(jobs, image_source, "dml:0"),
            submit_video_job(video_jobs, video_source, "dml:0"),
        )
        await jobs.queue.join()
        await video_jobs.queue.join()
    finally:
        await jobs.stop()
        await video_jobs.stop()

    assert tracker.max_in_flight == 1, "image and video jobs on the same device overlapped despite capacity 1"


async def test_same_device_concurrency_two_allows_two_jobs_at_once(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=2, MAX_CONCURRENT_JOBS=4)
    StorageService(settings)
    tracker = ConcurrencyTracker()
    jobs, video_jobs = make_shared_managers(settings, tracker)

    image_sources = [make_image_source(settings, f"photo-{i}.png") for i in range(2)]
    video_sources = [make_video_source(settings, f"clip-{i}.mp4") for i in range(2)]

    await jobs.start()
    await video_jobs.start()
    try:
        await asyncio.gather(
            *(submit_image_job(jobs, source, "dml:0") for source in image_sources),
            *(submit_video_job(video_jobs, source, "dml:0") for source in video_sources),
        )
        await jobs.queue.join()
        await video_jobs.queue.join()
    finally:
        await jobs.stop()
        await video_jobs.stop()

    assert tracker.max_in_flight == 2, "device capacity of 2 was not reached or was exceeded across managers"


async def test_job_manager_start_spawns_configured_worker_count_and_stop_cancels_all(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, MAX_CONCURRENT_JOBS=3)
    StorageService(settings)
    tracker = ConcurrencyTracker()
    jobs, _ = make_shared_managers(settings, tracker)

    await jobs.start()
    try:
        assert len(jobs.worker_tasks) == 3
        assert all(not task.done() for task in jobs.worker_tasks)
    finally:
        await jobs.stop()

    assert jobs.worker_tasks == []


async def test_video_job_manager_start_spawns_configured_worker_count_and_stop_cancels_all(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path, MAX_CONCURRENT_JOBS=3)
    StorageService(settings)
    tracker = ConcurrencyTracker()
    _, video_jobs = make_shared_managers(settings, tracker)

    await video_jobs.start()
    try:
        assert len(video_jobs.worker_tasks) == 3
        assert all(not task.done() for task in video_jobs.worker_tasks)
    finally:
        await video_jobs.stop()

    assert video_jobs.worker_tasks == []
