from __future__ import annotations

import asyncio
import logging
import subprocess
from fractions import Fraction
from pathlib import Path

from app.config import Settings
from app.exceptions import QueueFullError
from app.models import JobStatus, VideoUpscaleJob, utc_now
from app.services.media_tools import MediaTools, parse_fps_fraction, resolve_video_fps
from app.services.video_upscaler import VideoUpscaler

logger = logging.getLogger(__name__)

MAX_TARGET_FPS = 240


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
        target_fps: str | None = None,
        job_id: str | None = None,
    ) -> VideoUpscaleJob:
        source_fps = await self._validate_video(source_path)
        self._validate_request(
            model_name,
            scale,
            output_container,
            video_codec,
            video_preset,
            crf,
            fps_multiplier,
            target_fps,
            source_fps,
        )

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
            target_fps=target_fps,
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

    async def _validate_video(self, source_path: Path) -> Fraction:
        try:
            probe = await self.media_tools.ffprobe_json(source_path)
        except subprocess.CalledProcessError as exc:
            raise ValueError("Uploaded file is not a valid video") from exc
        streams = probe.get("streams", [])
        video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        if video_stream is None:
            raise ValueError("Uploaded file is not a valid video")
        return resolve_video_fps(video_stream.get("avg_frame_rate"), video_stream.get("r_frame_rate"))

    def _validate_request(
        self,
        model_name: str,
        scale: int,
        output_container: str,
        video_codec: str,
        video_preset: str,
        crf: int,
        fps_multiplier: int,
        target_fps: str | None,
        source_fps: Fraction,
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
        self._validate_fps_mode(fps_multiplier, target_fps, source_fps)

    def _validate_fps_mode(self, fps_multiplier: int, target_fps: str | None, source_fps: Fraction) -> None:
        if target_fps is not None and fps_multiplier > 1:
            raise ValueError("target_fps and fps_multiplier are mutually exclusive; provide only one")
        self._validate_fps_multiplier(fps_multiplier)
        if target_fps is not None:
            self._validate_target_fps(target_fps, source_fps)

    def _validate_target_fps(self, target_fps: str, source_fps: Fraction) -> None:
        target_fraction = parse_fps_fraction(target_fps)
        if target_fraction is None:
            raise ValueError(
                f"target_fps must be a positive fraction (e.g. '60' or '60000/1001'), got {target_fps!r}"
            )
        if target_fraction > MAX_TARGET_FPS:
            raise ValueError(f"target_fps must not exceed {MAX_TARGET_FPS}")
        if target_fraction <= source_fps:
            raise ValueError(
                f"target_fps ({target_fraction}) must be greater than the source video fps ({source_fps})"
            )
        self._validate_interpolation_enabled()

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
                    self._unlink_source_safely(job.source_path)
                    self.queue.task_done()

    @staticmethod
    def _unlink_source_safely(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete source upload %s", source_path)
