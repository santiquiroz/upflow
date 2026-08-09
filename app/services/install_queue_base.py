from __future__ import annotations

import asyncio
import contextlib
from typing import Generic, Protocol, TypeVar

# ---------------------------------------------------------------------------
# Scaffold comun de los installers/provisioners (asr, vulkan, packs, modelos,
# generacion, conversion): cola en memoria, UN worker asyncio, seam
# _process_next para tests, y el patron "el job reporta el error, el worker
# nunca muere".
#
# NO confundir con job_manager_base.QueuedJobManager (managers de MEDIA):
# aquel tiene N workers, semaforos de device, maquina de estados de cancel y
# hooks de cuota. Los installers tienen un ciclo de vida mas chico: encolar,
# procesar de a uno, reportar estado con el enum propio de cada familia.
#
# Los enums de estado se quedan en cada familia (AsrInstallStatus,
# VulkanInstallStatus, ProvisionStatus, InstallStatus, JobStatus): la base solo
# necesita saber cual es el valor de error (_error_status) para _fail_job.
# ---------------------------------------------------------------------------


class QueuedInstallJob(Protocol):
    id: str
    status: object
    error: str | None


JobT = TypeVar("JobT", bound=QueuedInstallJob)


class SingleWorkerJobQueue(Generic[JobT]):
    # Nombre de la task del worker (diagnostico); None deja el autonombre.
    _worker_name: str | None = None
    # Valor de error del enum de la familia; lo asigna _fail_job.
    _error_status: object = None

    def __init__(self) -> None:
        self._jobs: dict[str, JobT] = {}
        self._queue: asyncio.Queue[JobT] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(
                self._worker(), name=self._worker_name
            )

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task
        self._worker_task = None

    def status(self, job_id: str) -> JobT | None:
        return self._jobs.get(job_id)

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def _enqueue(self, job: JobT) -> str:
        self._jobs[job.id] = job
        await self._queue.put(job)
        return job.id

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._run(job)
            finally:
                self._queue.task_done()

    async def _process_next(self) -> bool:
        """Seam de tests: procesa exactamente un job encolado, sin worker vivo,
        para assertar el estado final sin sondear una asyncio.Task de fondo."""
        try:
            job = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return False
        try:
            await self._run(job)
        finally:
            self._queue.task_done()
        return True

    async def _run(self, job: JobT) -> None:
        raise NotImplementedError

    def _fail_job(self, job: JobT, exc: Exception) -> None:
        job.status = self._error_status
        job.error = str(exc) or type(exc).__name__
