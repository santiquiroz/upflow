from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.services.engines.sdcpp_models import CHECKPOINT_SUFFIXES, sdcpp_model_id
from app.services.hf_client import HfClient
from app.services.missing_pack import missing_pack_message

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vía rápida: instalar un checkpoint suelto para el lane Vulkan.
#
# El camino ONNX necesita exportar el checkpoint (~40 min) porque optimum ejecuta
# grafos, no .safetensors. stable-diffusion.cpp los ejecuta TAL CUAL, así que
# para ese lane "instalar" es bajar el archivo: minutos en vez de una hora, y
# ningún paso que pueda fallar a mitad.
#
# No pasa por el registro de modelos: no hay nada que registrar más allá del
# archivo, y list_sdcpp_models() ya lo descubre solo mirando la carpeta.
# ---------------------------------------------------------------------------


class VulkanInstallStatus(str, Enum):
    queued = "queued"
    downloading = "downloading"
    installed = "installed"
    error = "error"


@dataclass(slots=True, kw_only=True)
class VulkanInstallJob:
    id: str
    repo_id: str
    filename: str
    status: VulkanInstallStatus = VulkanInstallStatus.queued
    progress_pct: float | None = None
    model_id: str | None = None
    error: str | None = None


class VulkanModelInstaller:
    def __init__(self, settings: Settings, hf_client: HfClient) -> None:
        self.settings = settings
        self.hf_client = hf_client
        self._jobs: dict[str, VulkanInstallJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

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

    def status(self, job_id: str) -> VulkanInstallJob | None:
        return self._jobs.get(job_id)

    async def install(self, repo_id: str, filename: str) -> str:
        if not self.settings.enable_sdcpp:
            raise ValueError("El lane Vulkan está apagado (ENABLE_SDCPP).")
        if not self.settings.sdcpp_binary_path.exists():
            raise ValueError(missing_pack_message("sdcpp"))
        if Path(filename).suffix.lower() not in CHECKPOINT_SUFFIXES:
            raise ValueError(f"{filename} no es un checkpoint ({', '.join(CHECKPOINT_SUFFIXES)}).")
        # El nombre viene del repo remoto: sin esto, un "../../x" escribiría
        # fuera de la carpeta de modelos.
        if Path(filename).name != filename or not filename.strip():
            raise ValueError("Nombre de archivo inválido.")

        job = VulkanInstallJob(id=uuid4().hex, repo_id=repo_id, filename=filename)
        self._jobs[job.id] = job
        await self._queue.put(job.id)
        return job.id

    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            await self._run(self._jobs[job_id])

    async def _run(self, job: VulkanInstallJob) -> None:
        try:
            await self._download(job)
        except Exception as exc:  # noqa: BLE001 - el job reporta, no propaga
            logger.warning("vulkan install failed", extra={"repo": job.repo_id})
            job.status = VulkanInstallStatus.error
            job.error = str(exc) or type(exc).__name__

    async def _download(self, job: VulkanInstallJob) -> None:
        job.status = VulkanInstallStatus.downloading
        models_dir = self.settings.sdcpp_models_dir_path
        models_dir.mkdir(parents=True, exist_ok=True)
        dest = models_dir / job.filename

        def on_progress(done: int, total: int | None) -> None:
            job.progress_pct = round(done / total * 100, 1) if total else None

        await self.hf_client.download(job.repo_id, job.filename, dest, progress_cb=on_progress)
        if not dest.is_file() or dest.stat().st_size == 0:
            raise RuntimeError("La descarga quedó vacía.")
        job.model_id = sdcpp_model_id(dest)
        job.progress_pct = 100.0
        job.status = VulkanInstallStatus.installed
