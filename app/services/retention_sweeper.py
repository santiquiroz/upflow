from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from app.config import Settings
from app.models import AudioJob, JobStatus, TERMINAL_JOB_STATUSES, UpscaleJob, VideoUpscaleJob, utc_now
from app.services.audio_job_manager import AudioJobManager
from app.services.job_manager import JobManager
from app.services.video_job_manager import VideoJobManager

SWEEP_INTERVAL_SECONDS = 3600

logger = logging.getLogger(__name__)


class RetentionSweeper:
    def __init__(
        self,
        settings: Settings,
        job_manager: JobManager,
        video_job_manager: VideoJobManager,
        audio_job_manager: AudioJobManager | None = None,
    ) -> None:
        self.settings = settings
        self.job_manager = job_manager
        self.video_job_manager = video_job_manager
        self.audio_job_manager = audio_job_manager
        self.sweep_task: asyncio.Task | None = None
        self._stop_event = threading.Event()

    async def start(self) -> None:
        if self.sweep_task is None:
            self._stop_event.clear()
            self.sweep_task = asyncio.create_task(self._run(), name="retention-sweeper")

    async def stop(self) -> None:
        if self.sweep_task:
            # Signal the worker thread FIRST so an in-flight sweep bails between
            # phases, then cancel + await: cancel alone can't stop a to_thread
            # sweep already running, so we must wait for it (see _run) instead of
            # leaving a detached thread deleting files after stop() returns.
            self._stop_event.set()
            self.sweep_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.sweep_task
            self.sweep_task = None

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            # The sweep does blocking disk I/O (iterdir/stat/unlink/rmtree over
            # potentially thousands of files); run it off the event loop so it
            # never freezes concurrent job progress polling. _prune_finished_jobs
            # snapshots jobs.items() and uses pop(), so running in a worker thread
            # cannot hit "dict changed size" if the loop adds a job concurrently.
            # Shield + await-the-worker on cancel so stop() genuinely waits for an
            # in-flight sweep to finish (the _stop_event makes it bail fast).
            worker = asyncio.ensure_future(asyncio.to_thread(self._sweep_safely))
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                self._stop_event.set()
                with contextlib.suppress(BaseException):
                    await worker
                raise
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)

    def _sweep_safely(self) -> None:
        try:
            self.sweep_once()
        except Exception:  # noqa: BLE001
            logger.exception("Retention sweep failed; retrying on next interval")

    def sweep_once(self) -> None:
        active_source_paths = self._active_source_paths()
        active_video_work_ids = self._active_video_work_ids()
        # Bail between phases if stop() was requested, so a big sweep doesn't keep
        # the worker thread alive (and stop()'s await) longer than one phase.
        self._delete_expired_outputs()
        if self._stop_event.is_set():
            return
        self._delete_expired_uploads(active_source_paths)
        if self._stop_event.is_set():
            return
        self._delete_expired_work_dirs(active_video_work_ids)
        if self._stop_event.is_set():
            return
        self._prune_finished_jobs(self.job_manager.jobs)
        self._prune_finished_jobs(self.video_job_manager.jobs)
        if self.audio_job_manager is not None:
            self._prune_finished_jobs(self.audio_job_manager.jobs)

    def _audio_jobs(self) -> list[AudioJob]:
        if self.audio_job_manager is None:
            return []
        return list(self.audio_job_manager.jobs.values())

    def _active_source_paths(self) -> set[Path]:
        all_jobs = (
            list(self.job_manager.jobs.values())
            + list(self.video_job_manager.jobs.values())
            + self._audio_jobs()
        )
        return {job.source_path for job in all_jobs if not self._is_finished(job)}

    def _active_video_work_ids(self) -> set[str]:
        return {
            job.id for job in self.video_job_manager.jobs.values() if not self._is_finished(job)
        }

    @staticmethod
    def _is_finished(job: UpscaleJob | VideoUpscaleJob | AudioJob) -> bool:
        return job.status in TERMINAL_JOB_STATUSES

    def _delete_expired_outputs(self) -> None:
        if not self.settings.outputs_path.exists():
            return
        cutoff = time.time() - self.settings.output_ttl_hours * 3600
        for output_file in self.settings.outputs_path.iterdir():
            if output_file.is_file() and output_file.stat().st_mtime < cutoff:
                output_file.unlink(missing_ok=True)

    def _delete_expired_uploads(self, active_source_paths: set[Path]) -> None:
        if not self.settings.uploads_path.exists():
            return
        cutoff = time.time() - self.settings.output_ttl_hours * 3600
        for upload_file in self.settings.uploads_path.iterdir():
            if not upload_file.is_file() or upload_file in active_source_paths:
                continue
            if upload_file.stat().st_mtime < cutoff:
                upload_file.unlink(missing_ok=True)

    def _delete_expired_work_dirs(self, active_video_work_ids: set[str]) -> None:
        if not self.settings.video_work_path.exists():
            return
        cutoff = time.time() - self.settings.output_ttl_hours * 3600
        for work_dir in self.settings.video_work_path.iterdir():
            if not work_dir.is_dir() or work_dir.name in active_video_work_ids:
                continue
            if work_dir.stat().st_mtime < cutoff:
                shutil.rmtree(work_dir, ignore_errors=True)

    def _prune_finished_jobs(
        self, jobs: dict[str, UpscaleJob] | dict[str, VideoUpscaleJob] | dict[str, AudioJob]
    ) -> None:
        cutoff = utc_now() - timedelta(hours=self.settings.output_ttl_hours)
        # Snapshot before iterating + pop() instead of del: this runs in a worker
        # thread while the event loop may add/remove jobs concurrently.
        expired_ids = [job_id for job_id, job in list(jobs.items()) if self._is_expired(job, cutoff)]
        for job_id in expired_ids:
            jobs.pop(job_id, None)

    @classmethod
    def _is_expired(cls, job: UpscaleJob | VideoUpscaleJob | AudioJob, cutoff: datetime) -> bool:
        if not cls._is_finished(job):
            return False
        return job.finished_at is not None and job.finished_at < cutoff
