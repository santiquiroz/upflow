from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from urllib.parse import urlparse

from app.config import Settings
from app.exceptions import QueueFullError
from app.models import DownloadJob, JobStatus, TERMINAL_JOB_STATUSES, utc_now
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
from app.services.fetch import engine as fetch_engine
from app.services.fetch.options import (
    FetchRequest,
    build_plan,
    validate_max_height,
    validate_playlist_limit,
)

logger = logging.getLogger(__name__)

# Solo http/https. Un file:// o un data: no son "descargar de la web", son leer disco
# local con la URL como disfraz: exactamente la forma de un SSRF/LFI.
ALLOWED_SCHEMES = frozenset({"http", "https"})


class DownloadJobManager:
    def __init__(
        self,
        settings: Settings,
        *,
        quota_service: QuotaService | None = None,
    ) -> None:
        self.settings = settings
        self.quota_service = quota_service
        self.jobs: dict[str, DownloadJob] = {}
        self.queue: asyncio.Queue[DownloadJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self._worker_task: asyncio.Task[None] | None = None
        self._cancel_events: dict[str, threading.Event] = {}

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
        url: str,
        max_height: int = 1080,
        audio_only: bool = False,
        include_playlist: bool = False,
        playlist_limit: int = 10,
        subtitle_languages: list[str] | None = None,
        owner: AuthenticatedUser | None = None,
    ) -> DownloadJob:
        validate_url(url)
        validate_max_height(max_height)
        if include_playlist:
            validate_playlist_limit(playlist_limit)
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)

        job = DownloadJob(
            url=url,
            max_height=max_height,
            audio_only=audio_only,
            include_playlist=include_playlist,
            playlist_limit=playlist_limit,
            subtitle_languages=list(subtitle_languages or []),
            owner_id=owner.id if owner is not None else None,
        )
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError("La cola de descargas esta llena; proba de nuevo") from exc
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> DownloadJob | None:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status in TERMINAL_JOB_STATUSES:
            return False
        if job.status == JobStatus.queued:
            job.status = JobStatus.cancelled
            job.finished_at = utc_now()
            return True
        event = self._cancel_events.get(job_id)
        if event is not None:
            # El corte real ocurre en el hook de progreso del motor, que lanza.
            event.set()
        return True

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            if job.status == JobStatus.cancelled:
                self.queue.task_done()
                continue
            await self._run_job(job)
            self.queue.task_done()

    async def _run_job(self, job: DownloadJob) -> None:
        job.status = JobStatus.running
        job.started_at = utc_now()
        cancel_event = threading.Event()
        self._cancel_events[job.id] = cancel_event
        try:
            await asyncio.to_thread(self._download_blocking, job, cancel_event)
            job.status = JobStatus.completed
        except fetch_engine.FetchCancelled:
            # No es un fallo: lo pidio el usuario. Reportarlo como error mostraria un
            # rojo por algo intencional.
            job.status = JobStatus.cancelled
        except fetch_engine.FetchUnavailable as exc:
            job.status = JobStatus.failed
            job.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - el motivo va al usuario
            job.status = JobStatus.failed
            job.error = describe_failure(exc)
            logger.exception("download job %s failed", job.id)
        finally:
            job.finished_at = utc_now()
            self._cancel_events.pop(job.id, None)

    def _download_blocking(self, job: DownloadJob, cancel_event: threading.Event) -> None:
        info = fetch_engine.probe(job.url)
        job.media_title = info.title
        job.media_uploader = info.uploader
        job.extractor = info.extractor

        request = FetchRequest(
            url=job.url,
            # Aterriza en uploads para que la salida sea la MISMA entrada que un archivo
            # subido a mano: asi el pipeline de mejora no necesita saber que vino de una
            # descarga.
            output_dir=self.settings.uploads_path,
            max_height=job.max_height,
            audio_only=job.audio_only,
            include_playlist=job.include_playlist,
            playlist_limit=job.playlist_limit,
            subtitle_languages=tuple(job.subtitle_languages),
        )
        plan = build_plan(request, self.settings.ffmpeg_binary_path.parent)

        def on_progress(progress: fetch_engine.FetchProgress) -> None:
            job.downloaded_bytes = progress.downloaded_bytes
            job.total_bytes = progress.total_bytes
            fraction = progress.fraction
            job.progress_pct = None if fraction is None else round(fraction * 100, 1)

        job.output_paths = fetch_engine.download(
            plan, job.url, on_progress=on_progress, cancel_event=cancel_event
        )


def validate_url(url: str) -> None:
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Solo se aceptan URLs http o https; se recibio {parsed.scheme or 'ninguno'!r}"
        )
    if not parsed.netloc:
        raise ValueError("La URL no tiene host")


def describe_failure(exc: Exception) -> str:
    """Un motivo que le sirva a una persona, no un stacktrace.

    Es el punto mas importante de la UX de un descargador: los sitios cambian y la
    extraccion se rompe seguido, asi que la diferencia entre util e inutil es decir
    QUE paso. yt-dlp ya escribe mensajes legibles, pero les pega un prefijo 'ERROR: '
    y a veces el nombre del extractor.
    """
    message = str(exc).strip()
    for prefix in ("ERROR: ", "error: "):
        if message.startswith(prefix):
            message = message[len(prefix):]
    return message or exc.__class__.__name__
