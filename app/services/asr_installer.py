from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import Settings
from app.services.asr_files import missing_required_onnx, select_asr_files
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry

logger = logging.getLogger(__name__)

ASR_MODELS_SUBDIR = "asr"


class AsrInstallStatus(str, Enum):
    queued = "queued"
    downloading = "downloading"
    installed = "installed"
    error = "error"


@dataclass(slots=True, kw_only=True)
class AsrInstallJob:
    id: str
    repo_id: str
    status: AsrInstallStatus = AsrInstallStatus.queued
    progress_pct: float | None = None
    model_id: str | None = None
    error: str | None = None


def asr_model_id(repo_id: str) -> str:
    return "asr--" + repo_id.lower().replace("/", "--")


def _safe_staging_dest(staging_root: Path, relative_path: str) -> Path:
    """Resuelve el destino y verifica que no se escape del staging.

    El path viene del listado del repo, o sea de la red: sin este chequeo un
    `../..` escribiria fuera del directorio.
    """
    resolved_root = staging_root.resolve()
    dest = (staging_root / relative_path.replace("\\", "/")).resolve()
    if not dest.is_relative_to(resolved_root):
        raise ValueError(f"La ruta {relative_path!r} se escapa del directorio de staging.")
    return dest


class AsrModelInstaller:
    def __init__(self, settings: Settings, registry: ModelRegistry, hf_client: Any) -> None:
        self.settings = settings
        self.registry = registry
        self.hf_client = hf_client
        self._jobs: dict[str, AsrInstallJob] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
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

    async def install_from_hf(self, repo_id: str) -> str:
        job = AsrInstallJob(id=uuid4().hex, repo_id=repo_id)
        self._jobs[job.id] = job
        await self._queue.put(job.id)
        return job.id

    def status(self, install_id: str) -> AsrInstallJob | None:
        return self._jobs.get(install_id)

    def _lock_for(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()
        return self._locks[model_id]

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

    async def _run(self, job: AsrInstallJob) -> None:
        try:
            await self._download_and_register(job)
        except Exception as exc:  # noqa: BLE001 - el job reporta, no propaga
            logger.warning("asr install failed", extra={"repo_id": job.repo_id})
            job.status = AsrInstallStatus.error
            job.error = str(exc) or type(exc).__name__

    async def _download_and_register(self, job: AsrInstallJob) -> None:
        files = await self.hf_client.repo_files(job.repo_id)
        names = tuple(f.path for f in files)

        # Se valida ANTES de bajar: descubrir que falta el encoder al final dejaria
        # cientos de megas en disco para despues fallar al cargar.
        missing = missing_required_onnx(names)
        if missing:
            raise ValueError(
                f"El repo {job.repo_id!r} no publica {', '.join(missing)}. "
                "Un modelo de reconocimiento de voz para esta app necesita el par "
                "encoder/decoder exportado sin fusionar."
            )

        selected = set(select_asr_files(names))
        wanted = [f for f in files if f.path in selected]

        model_id = asr_model_id(job.repo_id)
        staging_root = self.settings.temp_path / f"asr-staging-{model_id}"
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=True)

        job.status = AsrInstallStatus.downloading
        try:
            total_bytes = sum(f.size for f in wanted) or 1
            downloaded = 0
            for hf_file in wanted:
                dest = _safe_staging_dest(staging_root, hf_file.path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                await self.hf_client.download(
                    job.repo_id, hf_file.path, dest, unlimited=True
                )
                downloaded += hf_file.size
                job.progress_pct = round(downloaded / total_bytes * 100, 1)

            job.model_id = await self._promote_and_register(
                staging_root, job.repo_id, sum(f.size for f in wanted)
            )
            job.status = AsrInstallStatus.installed
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    async def _promote_and_register(
        self, staging_root: Path, repo_id: str, size_bytes: int
    ) -> str:
        model_id = asr_model_id(repo_id)
        final_dir = self.settings.models_path / ASR_MODELS_SUBDIR / model_id
        async with self._lock_for(model_id):
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            shutil.move(str(staging_root), str(final_dir))
            self.registry.register(
                ModelEntry(
                    id=model_id,
                    name=repo_id,
                    kind=ModelKind.asr_onnx,
                    source=f"hf:{repo_id}",
                    size_bytes=size_bytes,
                    file_path=f"{ASR_MODELS_SUBDIR}/{model_id}",
                )
            )
        return model_id
