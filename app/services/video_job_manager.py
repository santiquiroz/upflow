from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.models import JobStatus, VideoUpscaleJob, utc_now
from app.services.media_tools import MediaTools
from app.services.video_upscaler import VideoUpscaler


class VideoJobManager:
    def __init__(self, settings: Settings, upscaler: VideoUpscaler, media_tools: MediaTools) -> None:
        self.settings = settings
        self.upscaler = upscaler
        self.media_tools = media_tools
        self.jobs: dict[str, VideoUpscaleJob] = {}
        self.queue: asyncio.Queue[VideoUpscaleJob] = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(1)
        self.worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker(), name="video-upscale-worker")

    async def stop(self) -> None:
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None

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
    ) -> VideoUpscaleJob:
        self._validate_video(source_path)
        self._validate_request(model_name, scale, output_container, video_codec, video_preset, crf)

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
        )
        self.jobs[job.id] = job
        await self.queue.put(job)
        return job

    def get_job(self, job_id: str) -> VideoUpscaleJob | None:
        return self.jobs.get(job_id)

    def _validate_video(self, source_path: Path) -> None:
        probe = self.media_tools.ffprobe_json(source_path)
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

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            async with self.semaphore:
                job.status = JobStatus.running
                job.started_at = utc_now()
                try:
                    job.output_path = await self.upscaler.run(job)
                    job.status = JobStatus.completed
                except Exception as exc:  # noqa: BLE001
                    job.status = JobStatus.failed
                    job.error = str(exc)
                finally:
                    job.finished_at = utc_now()
                    self.queue.task_done()
