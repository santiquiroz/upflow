from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta

from app.config import Settings
from app.models import JobStatus, UpscaleJob, VideoUpscaleJob, utc_now
from app.services.job_manager import JobManager
from app.services.video_job_manager import VideoJobManager

SWEEP_INTERVAL_SECONDS = 3600


class RetentionSweeper:
    def __init__(self, settings: Settings, job_manager: JobManager, video_job_manager: VideoJobManager) -> None:
        self.settings = settings
        self.job_manager = job_manager
        self.video_job_manager = video_job_manager
        self.sweep_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self.sweep_task is None:
            self.sweep_task = asyncio.create_task(self._run(), name="retention-sweeper")

    async def stop(self) -> None:
        if self.sweep_task:
            self.sweep_task.cancel()
            try:
                await self.sweep_task
            except asyncio.CancelledError:
                pass
            self.sweep_task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
            self.sweep_once()

    def sweep_once(self) -> None:
        self._delete_expired_outputs()
        self._prune_finished_jobs(self.job_manager.jobs)
        self._prune_finished_jobs(self.video_job_manager.jobs)

    def _delete_expired_outputs(self) -> None:
        if not self.settings.outputs_path.exists():
            return
        cutoff = time.time() - self.settings.output_ttl_hours * 3600
        for output_file in self.settings.outputs_path.iterdir():
            if output_file.is_file() and output_file.stat().st_mtime < cutoff:
                output_file.unlink(missing_ok=True)

    def _prune_finished_jobs(self, jobs: dict[str, UpscaleJob] | dict[str, VideoUpscaleJob]) -> None:
        cutoff = utc_now() - timedelta(hours=self.settings.output_ttl_hours)
        expired_ids = [job_id for job_id, job in jobs.items() if self._is_expired(job, cutoff)]
        for job_id in expired_ids:
            del jobs[job_id]

    @staticmethod
    def _is_expired(job: UpscaleJob | VideoUpscaleJob, cutoff: datetime) -> bool:
        if job.status not in (JobStatus.completed, JobStatus.failed):
            return False
        return job.finished_at is not None and job.finished_at < cutoff
