from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.services.engines.sdcpp_models import CHECKPOINT_SUFFIXES, sdcpp_model_id
from app.services.hf_client import HfClient
from app.services.install_queue_base import SingleWorkerJobQueue
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


class VulkanModelInstaller(SingleWorkerJobQueue[VulkanInstallJob]):
    _error_status = VulkanInstallStatus.error

    def __init__(self, settings: Settings, hf_client: HfClient) -> None:
        super().__init__()
        self.settings = settings
        self.hf_client = hf_client

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

        return await self._enqueue(
            VulkanInstallJob(id=uuid4().hex, repo_id=repo_id, filename=filename)
        )

    async def _run(self, job: VulkanInstallJob) -> None:
        try:
            await self._download(job)
        except Exception as exc:  # noqa: BLE001 - el job reporta, no propaga
            logger.warning("vulkan install failed", extra={"repo": job.repo_id})
            self._fail_job(job, exc)

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
