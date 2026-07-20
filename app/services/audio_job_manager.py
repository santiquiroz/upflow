from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import AUDIO_ENHANCE_MODES, AUDIO_RESTORE_MODES, Settings
from app.exceptions import QueueFullError
from app.models import TERMINAL_JOB_STATUSES, AudioJob, JobStatus, utc_now
from app.services.audio_pipeline import AudioPipeline
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.restorer_registry import validate_restore_mode_ready

logger = logging.getLogger(__name__)


class AudioJobManager:
    """Standalone audio job manager. Mirrors JobManager/VideoJobManager:
    bounded queue, N workers, shared device_semaphores (restore uses the GPU),
    CancelledError-safe execution, source unlink + task_done in finally.

    Audio jobs do NOT participate in auto-routing: restore is experimental and
    device-pinned, so the `auto` sentinel is rejected here rather than resolved.
    """

    def __init__(
        self,
        settings: Settings,
        pipeline: AudioPipeline,
        device_semaphores: DeviceSemaphores,
        *,
        devices: DevicesService | None = None,
    ) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self.devices = devices
        self.jobs: dict[str, AudioJob] = {}
        self.queue: asyncio.Queue[AudioJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self.device_semaphores = device_semaphores
        self.worker_tasks: list[asyncio.Task] = []
        self._active: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        if self.worker_tasks:
            return
        self.worker_tasks = [
            asyncio.create_task(self._worker(), name=f"audio-worker-{i}")
            for i in range(self.settings.max_concurrent_jobs)
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
        denoise: str | None = None,
        restore: str | None = None,
        device: str | None = None,
        job_id: str | None = None,
    ) -> AudioJob:
        self._validate_modes(denoise, restore)
        await self._validate_device(device)

        job = AudioJob(
            source_path=source_path,
            original_filename=original_filename,
            denoise=denoise,
            restore=restore,
            device=device,
        )
        if job_id is not None:
            job.id = job_id
        self._enqueue(job)
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> AudioJob | None:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job.status in TERMINAL_JOB_STATUSES:
            return False
        if job.status == JobStatus.queued:
            # Still in the queue: mark it so the worker skips it on dequeue.
            job.status = JobStatus.cancelled
            job.finished_at = utc_now()
            return True
        task = self._active.get(job_id)
        if task is not None:
            task.cancel()
        return True

    def _enqueue(self, job: AudioJob) -> None:
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError("Audio job queue is full; try again later") from exc

    def _validate_modes(self, denoise: str | None, restore: str | None) -> None:
        if denoise is None and restore is None:
            raise ValueError("At least one of denoise or restore must be requested")
        if denoise is not None:
            self._validate_denoise(denoise)
        if restore is not None:
            self._validate_restore(restore)

    def _validate_denoise(self, denoise: str) -> None:
        if denoise not in AUDIO_ENHANCE_MODES:
            raise ValueError(f"denoise must be one of {sorted(AUDIO_ENHANCE_MODES)}")
        if not self.settings.audio_enhance_available(denoise):
            raise ValueError(
                f"denoise mode {denoise!r} requested but not installed "
                "(run scripts/download-deepfilternet.ps1)"
            )

    def _validate_restore(self, restore: str) -> None:
        if restore not in AUDIO_RESTORE_MODES:
            raise ValueError(f"restore must be one of {sorted(AUDIO_RESTORE_MODES)}")
        validate_restore_mode_ready(self.settings, restore)

    async def _validate_device(self, device: str | None) -> None:
        if device is None:
            return
        if device == AUTO_DEVICE_ID:
            raise ValueError("device 'auto' is not supported for audio jobs; pin a concrete device (cpu|dml:N)")
        if self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            if job.status == JobStatus.cancelled:
                # Cancelled while waiting in the queue: skip without processing.
                self._unlink_source_safely(job.source_path)
                self.queue.task_done()
                continue
            await self._run_job(job)

    async def _run_job(self, job: AudioJob) -> None:
        async with self.device_semaphores.acquire(job.device):
            await self._execute_job(job)

    async def _execute_job(self, job: AudioJob) -> None:
        job.status = JobStatus.running
        job.started_at = utc_now()
        run_task = asyncio.ensure_future(self._run_engine(job))
        self._active[job.id] = run_task
        try:
            await run_task
            job.status = JobStatus.completed
        except asyncio.CancelledError:
            run_task.cancel()
            if asyncio.current_task().cancelling() > 0:
                # The WORKER task itself was cancelled (shutdown via stop()):
                # fail the job and re-raise so the worker actually dies.
                job.status = JobStatus.failed
                job.error = "Job cancelled"
                raise
            # Only the child engine task was cancelled (per-job cancel_job):
            # mark cancelled and let the worker live on for other jobs.
            job.status = JobStatus.cancelled
            job.error = None
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            self._active.pop(job.id, None)
            job.finished_at = utc_now()
            self._unlink_source_safely(job.source_path)
            self.queue.task_done()

    async def _run_engine(self, job: AudioJob) -> None:
        job.output_path = await self.pipeline.run(job)

    @staticmethod
    def _unlink_source_safely(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete source upload %s", source_path)
