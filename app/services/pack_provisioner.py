from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from app.config import Settings, resolve_against_project_root
from app.services.capabilities import CATALOG, PathRequirement
from app.services.process_runner import run_guarded_process

logger = logging.getLogger(__name__)

# Los paquetes vendorizados se bajaban a mano corriendo estos scripts. La app
# ahora los corre por el usuario, y la regla que mantiene esto acotado es REUSAR
# el script, no reimplementar la descarga: el riesgo de regresion esta en tocar
# codigo que funciona desde muchas versiones, no en el boton.
PACK_SCRIPTS: dict[str, str] = {
    "realesrgan": "download-realesrgan.ps1",
    "realesrgan-onnx": "download-realesrgan-onnx.ps1",
    "rife": "download-rife.ps1",
    "gmfss": "download-gmfss-onnx.ps1",
    "deepfilternet": "download-deepfilternet.ps1",
    "apollo": "download-apollo.ps1",
    "audiosr": "download-audiosr-onnx.ps1",
    "ffmpeg": "download-ffmpeg.ps1",
    "mobilesam": "download-mobilesam.ps1",
    "migan": "download-migan.ps1",
}

# Los scripts descargan cientos de MB desde GitHub releases. El techo es un
# limite de seguridad contra un proceso colgado, no una expectativa de duracion.
PROVISION_TIMEOUT_SECONDS = 45 * 60

SCRIPTS_DIRNAME = "scripts"


class ProvisionStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


@dataclass(slots=True, kw_only=True)
class ProvisionJob:
    id: str
    pack: str
    status: ProvisionStatus = ProvisionStatus.queued
    error: str | None = None


class UnknownPackError(ValueError):
    pass


Platform = Literal["win32", "other"]


def packs_required_by_catalog() -> frozenset[str]:
    return frozenset(
        requirement.pack
        for capability in CATALOG
        for requirement in capability.requirements
        if isinstance(requirement, PathRequirement)
    )


def script_for(pack: str) -> str:
    script = PACK_SCRIPTS.get(pack)
    if script is None:
        raise UnknownPackError(f"No hay script de descarga para el paquete {pack!r}.")
    return script


def script_path(pack: str) -> Path:
    return resolve_against_project_root(SCRIPTS_DIRNAME) / script_for(pack)


def provisioning_supported(platform: str = sys.platform) -> bool:
    """Los scripts de descarga son PowerShell.

    En otras plataformas la capacidad queda en needs_setup con un motivo
    explicito, en vez de fallar de forma rara a mitad de una descarga.
    """
    return platform == "win32"


def build_command(pack: str) -> list[str]:
    return [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path(pack)),
    ]


def _tail(raw: bytes, limit: int = 600) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    return text[-limit:] if len(text) > limit else text


class PackProvisioner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jobs: dict[str, ProvisionJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        # Un paquete a la vez: dos descargas del mismo destino se pisarian los
        # archivos a medio escribir.
        self._locks: dict[str, asyncio.Lock] = {}

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

    async def provision(self, pack: str) -> str:
        # Se valida ANTES de encolar: un pack desconocido tiene que fallar la
        # request, no aparecer como un job que despues se muere solo.
        script_for(pack)
        job = ProvisionJob(id=uuid4().hex, pack=pack)
        self._jobs[job.id] = job
        await self._queue.put(job.id)
        return job.id

    def status(self, job_id: str) -> ProvisionJob | None:
        return self._jobs.get(job_id)

    def _lock_for(self, pack: str) -> asyncio.Lock:
        if pack not in self._locks:
            self._locks[pack] = asyncio.Lock()
        return self._locks[pack]

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            await self._run(self._jobs[job_id])

    async def _process_next(self) -> bool:
        if self._queue.empty():
            return False
        job_id = await self._queue.get()
        await self._run(self._jobs[job_id])
        return True

    async def _run(self, job: ProvisionJob) -> None:
        async with self._lock_for(job.pack):
            try:
                await self._execute(job)
            except Exception as exc:  # noqa: BLE001 - el job reporta, no propaga
                logger.warning("pack provisioning failed", extra={"pack": job.pack})
                job.status = ProvisionStatus.error
                job.error = str(exc) or type(exc).__name__

    async def _execute(self, job: ProvisionJob) -> None:
        if not provisioning_supported():
            raise RuntimeError(
                "La descarga automatica de paquetes solo esta disponible en Windows: "
                "los scripts son PowerShell. Corre el script a mano en esta plataforma."
            )

        path = script_path(job.pack)
        if not path.exists():
            raise FileNotFoundError(f"No existe el script de descarga {path}.")

        job.status = ProvisionStatus.running
        _stdout, stderr, returncode = await run_guarded_process(
            build_command(job.pack), timeout=PROVISION_TIMEOUT_SECONDS
        )
        if returncode != 0:
            detail = _tail(stderr)
            raise RuntimeError(
                f"El script {path.name} termino con codigo {returncode}."
                + (f" {detail}" if detail else "")
            )
        job.status = ProvisionStatus.done
