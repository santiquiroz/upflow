from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import threading
from pathlib import Path
from urllib.parse import urlparse

from app.config import Settings
from app.exceptions import QueueFullError
from app.models import DownloadJob, JobStatus, TERMINAL_JOB_STATUSES, utc_now
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
from fetchflow import engine as fetch_engine
# describe_failure vive en la libreria: limpiar la decoracion de terminal de yt-dlp y
# traducir un rate limit le sirve a cualquiera que la use, no solo a Upflow.
from fetchflow.errors import describe_failure
from fetchflow.options import (
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
    host = (parsed.hostname or "").strip().rstrip(".")
    if not host:
        raise ValueError("La URL no tiene host")
    _reject_internal_target(host)


def _addresses_for(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Todas las direcciones a las que resuelve el host.

    TODAS y no la primera: un DNS con round-robin puede devolver una publica y una
    privada, y quedarse con la primera dejaria pasar la privada la mitad de las veces.
    """
    try:
        # Un host que YA es una IP no necesita DNS. Resolverlo igual seria una consulta
        # al vicio, y en un equipo sin red haria fallar la validacion de una direccion
        # que se puede evaluar sola.
        return [ipaddress.ip_address(host.strip("[]"))]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise ValueError(f"No se pudo resolver el host {host!r}") from exc
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def _reject_internal_target(host: str) -> None:
    """Rechaza lo que apunte a la red interna o al propio equipo.

    Es la guarda contra SSRF: sin ella, cualquiera que pueda crear un job hace que el
    SERVIDOR pida la URL, y con el extractor generico de yt-dlp eso alcanza para sondear
    servicios internos, el endpoint de metadata de la nube (169.254.169.254) o la propia
    app en localhost.

    LIMITE CONOCIDO, que conviene decir en vez de aparentar que no existe: la validacion
    resuelve el nombre ACA y yt-dlp lo resuelve de nuevo al descargar, asi que un DNS
    hostil que cambie la respuesta entre los dos momentos (rebinding) se escapa. Cerrarlo
    del todo pide fijar la IP en el socket, que yt-dlp no expone. Esto sube el costo del
    ataque de trivial a no trivial; no lo vuelve imposible.
    """
    for address in _addresses_for(host):
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ValueError(
                f"El host {host!r} apunta a una direccion interna ({address}). "
                "Solo se pueden descargar URLs publicas."
            )


