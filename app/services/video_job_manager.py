from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from app.config import Settings
from app.exceptions import QueueFullError
from app.models import JobStatus, VideoUpscaleJob, utc_now
from app.services.media_tools import MediaTools
from app.services.video_upscaler import VideoUpscaler


class VideoJobManager:
    def __init__(
        self,
        settings: Settings,
        upscaler: VideoUpscaler,
        media_tools: MediaTools,
        gpu_semaphore: asyncio.Semaphore,
    ) -> None:
        self.settings = settings
        self.upscaler = upscaler
        self.media_tools = media_tools
        self.jobs: dict[str, VideoUpscaleJob] = {}
        self.queue: asyncio.Queue[VideoUpscaleJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self.gpu_semaphore = gpu_semaphore
        self.worker_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        if self.worker_tasks:
            return
        self.worker_tasks = [
            asyncio.create_task(self._worker(), name=f"video-upscale-worker-{i}")
            for i in range(self.settings.gpu_concurrency)
        ]

    async def stop(self) -> None:
        for task in self.worker_tasks:
            task.cancel()
        for task in self.worker_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.worker_tasks = []

    def queue_depth(self) -> int:
        return self.queue.qsize()

    async def create_job(
        self,
        *,
        source_path: Path,
        original_filename: str,
        model_name: str,
        scale: int,
        output_container: str,
        video_codec: str,
        video_preset: str,
        crf: int,
        keep_audio: bool,
        fps_multiplier: int = 1,
        job_id: str | None = None,
    ) -> VideoUpscaleJob:
        await self._validate_video(source_path)
        self._validate_request(model_name, scale, output_container, video_codec, video_preset, crf, fps_multiplier)

        job = VideoUpscaleJob(
            source_path=source_path,
            original_filename=original_filename,
            model_name=self.settings.resolve_engine_model_name(model_name, scale),
            scale=scale,
            output_container=output_container,
            video_codec=video_codec,
            video_preset=video_preset,
            crf=crf,
            keep_audio=keep_audio,
            fps_multiplier=fps_multiplier,
        )
        if job_id is not None:
            job.id = job_id
        self._enqueue(job)
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> VideoUpscaleJob | None:
        return self.jobs.get(job_id)

    def _enqueue(self, job: VideoUpscaleJob) -> None:
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError("Video job queue is full; try again later") from exc

    async def _validate_video(self, source_path: Path) -> None:
        try:
            probe = await self.media_tools.ffprobe_json(source_path)
        except subprocess.CalledProcessError as exc:
            raise ValueError("Uploaded file is not a valid video") from exc
        streams = probe.get("streams", [])
        if not any(stream.get("codec_type") == "video" for stream in streams):
            raise ValueError("Uploaded file is not a valid video")

    def _validate_request(
        self,
        model_name: str,
        scale: int,
        output_container: str,
        video_codec: str,
        video_preset: str,
        crf: int,
        fps_multiplier: int,
    ) -> None:
        if model_name not in self.settings.model_keys:
            raise ValueError(f"Model must be one of {sorted(self.settings.model_keys)}")
        option = self.settings.get_model_option(model_name)
        if option and scale not in option["scales"]:
            raise ValueError(f"Model {model_name} supports only scales {option['scales']}")
        if output_container not in {"mp4", "mkv"}:
            raise ValueError("Output container must be mp4 or mkv")
        if video_codec not in {"libx264", "libx265"}:
            raise ValueError("Video codec must be libx264 or libx265")
        if video_preset not in {"medium", "slow", "veryslow"}:
            raise ValueError("Video preset must be medium, slow, or veryslow")
        if crf < 10 or crf > 28:
            raise ValueError("CRF must be between 10 and 28")
        self._validate_fps_multiplier(fps_multiplier)

    def _validate_fps_multiplier(self, fps_multiplier: int) -> None:
        if fps_multiplier <= 0:
            raise ValueError("fps_multiplier must be a positive integer")
        allowed_multipliers = {1, *self.settings.allowed_fps_multiplier_values}
        if fps_multiplier not in allowed_multipliers:
            raise ValueError(
                f"fps_multiplier must be 1 (off) or one of {sorted(allowed_multipliers - {1})}"
            )
        if fps_multiplier > 1:
            self._validate_interpolation_enabled()

    def _validate_interpolation_enabled(self) -> None:
        if not self.settings.enable_interpolation:
            raise ValueError(
                "Frame interpolation is disabled by configuration (set ENABLE_INTERPOLATION=true)"
            )
        if not self.settings.interpolation_available():
            raise ValueError(
                "Frame interpolation requested but RIFE is not installed "
                "(run scripts/download-rife.ps1)"
            )

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            async with self.gpu_semaphore:
                job.status = JobStatus.running
                job.started_at = utc_now()
                try:
                    job.output_path = await self.upscaler.run(job, fps_multiplier=job.fps_multiplier)
                    job.status = JobStatus.completed
                except asyncio.CancelledError:
                    job.status = JobStatus.failed
                    job.error = "Job cancelled"
                    raise
                except Exception as exc:  # noqa: BLE001
                    job.status = JobStatus.failed
                    job.error = str(exc)
                finally:
                    job.finished_at = utc_now()
                    job.source_path.unlink(missing_ok=True)
                    self.queue.task_done()
