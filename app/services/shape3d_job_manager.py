"""Cola para generar mallas 3D: dos minutos de CPU no entran en una petición.

Medido: 116-137 s por malla en esta máquina. Eso no puede ser una petición
sincrónica — el navegador cortaría antes y el usuario no tendría a qué volver.

Lo que hace distinto a este manager: la malla generada NO se entrega cruda. Pasa
obligatoriamente por el banco (`print_check`), y el veredicto viaja con el
resultado. Una malla generada que no cierra no es una pieza, y entregarla sin
decirlo sería mandar a imprimir con confianza prestada.

No toma el semáforo de GPU: esto es CPU pura, y colgarse un cupo de GPU durante
dos minutos estancaría la cola de escalado, que es el trabajo principal de Upflow.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import Settings
from app.exceptions import QueueFullError
from app.models import JobStatus, Shape3dJob, utc_now
from app.services.auth.identity import AuthenticatedUser
from app.services.engines.shape3d import Shape3dEngine, Shape3dUnavailable
from app.services.mesh_fit import PRINTER_BEDS
from app.services.print_check import check_stl_for_printing
from app.services.stl_writer import write_stl

logger = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 400


class Shape3dJobManager:
    def __init__(
        self,
        settings: Settings,
        engine: Shape3dEngine,
        *,
        quota_service=None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.quota_service = quota_service
        self.jobs: dict[str, Shape3dJob] = {}
        self.queue: asyncio.Queue[Shape3dJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    def queue_depth(self) -> int:
        return self.queue.qsize()

    async def create_job(
        self,
        *,
        prompt: str,
        printer: str = "ender-3",
        target_mm: float | None = None,
        owner: AuthenticatedUser | None = None,
        job_id: str | None = None,
    ) -> Shape3dJob:
        if not prompt.strip():
            raise ValueError("Hace falta una descripcion de la pieza.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"La descripcion tiene {len(prompt)} caracteres y el maximo son "
                f"{MAX_PROMPT_CHARS}. Una descripcion mas larga no mejora la malla."
            )
        if printer not in PRINTER_BEDS:
            raise ValueError(
                f"No conozco la impresora {printer!r}. Las que si: "
                f"{', '.join(sorted(PRINTER_BEDS))}."
            )
        if target_mm is not None and target_mm <= 0:
            raise ValueError("La medida pedida tiene que ser mayor que cero.")
        if not self.engine.available():
            raise Shape3dUnavailable(
                "El modelo 3D no esta instalado. Bajalo con scripts/download-shap-e.ps1"
            )
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)

        job = Shape3dJob(
            prompt=prompt.strip(),
            printer=printer,
            target_mm=target_mm,
            owner_id=owner.id if owner is not None else None,
        )
        if job_id is not None:
            job.id = job_id
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError("La cola de generacion 3D esta llena; proba de nuevo") from exc
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Shape3dJob | None:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status in (
            JobStatus.completed,
            JobStatus.failed,
            JobStatus.cancelled,
        ):
            return False
        job.status = JobStatus.cancelled
        job.finished_at = utc_now()
        return True

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            if job.status != JobStatus.cancelled:
                await self._run_job(job)
            self.queue.task_done()

    async def _process_next(self) -> None:
        """Un solo job, para los tests: el worker corre para siempre."""
        job = await self.queue.get()
        if job.status != JobStatus.cancelled:
            await self._run_job(job)
        self.queue.task_done()

    async def _run_job(self, job: Shape3dJob) -> None:
        job.status = JobStatus.running
        job.started_at = utc_now()
        try:
            # `to_thread` y no el loop: son dos minutos de CPU, y bloquear el loop
            # dejaria la app entera sin responder mientras tanto.
            triangulos = await asyncio.to_thread(self.engine.generate_from_text, job.prompt)
            destino = self.settings.outputs_path / f"{job.id}.stl"
            await asyncio.to_thread(write_stl, destino, triangulos)

            reporte = await asyncio.to_thread(
                check_stl_for_printing,
                destino,
                printer=job.printer,
                target_axis="x" if job.target_mm else None,
                target_mm=job.target_mm,
            )
            job.output_path = destino
            job.can_print = reporte.can_print
            job.size_mm = reporte.size
            job.triangle_count = reporte.mesh.triangle_count
            job.blockers = list(reporte.blockers)
            job.advice = list(reporte.advice)
            job.status = JobStatus.completed
        except Shape3dUnavailable as exc:
            job.status = JobStatus.failed
            job.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - el job reporta cualquier fallo
            logger.exception("Fallo generando la malla 3D")
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            job.finished_at = utc_now()
