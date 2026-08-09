"""Base común de los managers de jobs con cola.

Cinco managers (imagen, video, audio, generación, transcripción) repetían
byte-a-byte el ciclo de vida cola/worker/cancel/cuota — el fix de la ventana
del semáforo (2026-08-08) tuvo que copiarse cinco veces. Esta base lo deja en
UN lugar; cada manager solo define cómo admite (permiso de device), cómo corre
su motor y cómo limpia su fuente.

Quedan afuera a propósito: DownloadJobManager (cancel cooperativo por
threading.Event, no Task.cancel) y Shape3dJobManager (hilo de CPU no
interrumpible, sin permiso de device) — sus ciclos de vida son genuinamente
distintos y forzarlos acá los complicaría en vez de simplificarlos.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncContextManager, Generic, TypeVar

from app.config import Settings
from app.exceptions import QueueFullError
from app.models import TERMINAL_JOB_STATUSES, JobStatus, utc_now
from app.services.auth.quotas import QuotaService

JobT = TypeVar("JobT")


class QueuedJobManager(Generic[JobT]):
    queue_full_message = "Job queue is full; try again later"
    worker_name_prefix: str | None = None

    def __init__(
        self,
        settings: Settings,
        *,
        quota_service: QuotaService | None = None,
        worker_count: int = 1,
    ) -> None:
        self.settings = settings
        self.quota_service = quota_service
        self.jobs: dict[str, JobT] = {}
        self.queue: asyncio.Queue[JobT] = asyncio.Queue(maxsize=settings.max_queue_size)
        self.worker_tasks: list[asyncio.Task] = []
        self._active: dict[str, asyncio.Task] = {}
        self._worker_count = worker_count

    # ------------------------------------------------------------ lifecycle

    async def start(self) -> None:
        if self.worker_tasks:
            return
        self.worker_tasks = [
            asyncio.create_task(self._worker(), name=self._worker_name(index))
            for index in range(self._worker_count)
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

    def _worker_name(self, index: int) -> str | None:
        if self.worker_name_prefix is None:
            return None
        return f"{self.worker_name_prefix}-{index}"

    def queue_depth(self) -> int:
        return self.queue.qsize()

    def get_job(self, job_id: str) -> JobT | None:
        return self.jobs.get(job_id)

    # ------------------------------------------------------------ cancel

    def cancel_job(self, job_id: str) -> bool:
        job: Any = self.jobs.get(job_id)
        if job is None:
            return False
        if job.status in TERMINAL_JOB_STATUSES:
            return False
        if job.status == JobStatus.queued:
            # Still in the queue (or dequeued and waiting for a device permit —
            # _execute_job re-checks): mark it so it is skipped, never run.
            job.status = JobStatus.cancelled
            job.finished_at = utc_now()
            return True
        task = self._active.get(job_id)
        if task is not None:
            task.cancel()
        return True

    # ------------------------------------------------------------ queue

    def _enqueue(self, job: Any) -> None:
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError(self.queue_full_message) from exc

    async def _worker(self) -> None:
        while True:
            job: Any = await self.queue.get()
            if job.status == JobStatus.cancelled:
                # Cancelled while waiting in the queue: skip without processing.
                self._cleanup_source(job)
                self.queue.task_done()
                continue
            await self._dispatch(job)

    async def _process_next(self) -> bool:
        """Un solo job, para los tests: el worker real corre para siempre."""
        if self.queue.empty():
            return False
        job: Any = await self.queue.get()
        if job.status == JobStatus.cancelled:
            self._cleanup_source(job)
            self.queue.task_done()
            return True
        await self._dispatch(job)
        return True

    async def _dispatch(self, job: JobT) -> None:
        async with self._admit(job):
            await self._execute_job(job)

    # ------------------------------------------------------------ execution

    async def _execute_job(self, job: Any) -> None:
        if job.status == JobStatus.cancelled:
            # Cancelled while this worker waited for a device permit: the job
            # was already out of the queue, so the dequeue-side skip in
            # _worker can't catch it. Without this re-check the job would
            # silently resurrect and run to completion.
            self._cleanup_source(job)
            self.queue.task_done()
            return
        job.status = JobStatus.running
        job.started_at = utc_now()
        self._on_running(job)
        run_task = asyncio.ensure_future(self._run_engine(job))
        self._active[job.id] = run_task
        try:
            await run_task
            job.status = JobStatus.completed
            self._on_completed(job)
        except asyncio.CancelledError:
            run_task.cancel()
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0:
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
            self._cleanup_source(job)
            self.queue.task_done()
            self._record_quota_usage(job)

    def _fail_dequeued_job(self, job: Any, error: str) -> None:
        job.status = JobStatus.failed
        job.error = error
        job.finished_at = utc_now()
        self._cleanup_source(job)
        self.queue.task_done()

    # ------------------------------------------------------------ hooks

    def _admit(self, job: JobT) -> AsyncContextManager[Any]:
        """Permiso para correr (semáforo de device). Default: sin gate."""
        return contextlib.nullcontext()

    async def _run_engine(self, job: JobT) -> None:
        raise NotImplementedError

    def _cleanup_source(self, job: JobT) -> None:
        """Limpieza del archivo fuente al terminar/saltear. Default: nada."""

    def _on_running(self, job: JobT) -> None:
        """Hook tras status=running (p.ej. avanzar stage). Default: nada."""

    def _on_completed(self, job: JobT) -> None:
        """Hook tras status=completed. Default: nada."""

    def _record_quota_usage(self, job: Any) -> None:
        if self.quota_service is None or job.started_at is None or job.finished_at is None:
            return
        duration = (job.finished_at - job.started_at).total_seconds()
        self.quota_service.record_usage(job.owner_id, duration)
