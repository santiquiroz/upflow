from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import AUDIO_ENHANCE_MODES, AUDIO_OUTPUT_FORMATS, AUDIO_RESTORE_MODES, Settings
from app.models import AudioJob
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
from app.services.audio_pipeline import AudioPipeline
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.job_manager_base import QueuedJobManager
from app.services.missing_pack import missing_pack_message
from app.services.restorer_registry import validate_restore_mode_ready

logger = logging.getLogger(__name__)


class AudioJobManager(QueuedJobManager[AudioJob]):
    """Standalone audio job manager sobre QueuedJobManager (cola acotada, N
    workers, cancel seguro, unlink + task_done en finally — todo en la base).

    Audio jobs do NOT participate in auto-routing: restore is experimental and
    device-pinned, so the `auto` sentinel is rejected here rather than resolved.
    """

    queue_full_message = "Audio job queue is full; try again later"
    worker_name_prefix = "audio-worker"

    def __init__(
        self,
        settings: Settings,
        pipeline: AudioPipeline,
        device_semaphores: DeviceSemaphores,
        *,
        devices: DevicesService | None = None,
        quota_service: QuotaService | None = None,
    ) -> None:
        super().__init__(
            settings,
            quota_service=quota_service,
            worker_count=settings.max_concurrent_jobs,
        )
        self.pipeline = pipeline
        self.devices = devices
        self.device_semaphores = device_semaphores

    async def create_job(
        self,
        *,
        source_path: Path,
        original_filename: str,
        denoise: str | None = None,
        restore: str | None = None,
        device: str | None = None,
        output_format: str = "flac",
        voice_steps: list[str] | None = None,
        voice_delivery: str | None = None,
        voice_presence_db: float | None = None,
        master: str | None = None,
        job_id: str | None = None,
        owner: AuthenticatedUser | None = None,
    ) -> AudioJob:
        self._validate_modes(denoise, restore)
        selected_voice_steps = self._validate_voice_selection(
            voice_steps or [], voice_delivery
        )
        self._validate_output_format(output_format)
        await self._validate_device(device)

        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)

        job = AudioJob(
            source_path=source_path,
            original_filename=original_filename,
            denoise=denoise,
            restore=restore,
            device=device,
            output_format=output_format,
            voice_steps=selected_voice_steps,
            voice_delivery=voice_delivery,
            voice_presence_db=voice_presence_db,
            master=master,
            owner_id=owner.id if owner is not None else None,
        )
        if job_id is not None:
            job.id = job_id
        self._enqueue(job)
        self.jobs[job.id] = job
        return job

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
                missing_pack_message(
                    "deepfilternet", detail=f"Modo de limpieza pedido: {denoise!r}."
                )
            )

    def _validate_restore(self, restore: str) -> None:
        if restore not in AUDIO_RESTORE_MODES:
            raise ValueError(f"restore must be one of {sorted(AUDIO_RESTORE_MODES)}")
        validate_restore_mode_ready(self.settings, restore)

    def _validate_voice_selection(
        self, selected: list[str], delivery: str | None
    ) -> list[str]:
        from app.services.voice_chain import delivery_choices, step_catalog

        known = {info.id for info in step_catalog()}
        unknown = sorted(set(selected) - known)
        if unknown:
            raise ValueError(
                "Pasos de mejora de voz desconocidos: " + ", ".join(unknown)
            )
        if delivery is not None:
            valid = {choice["id"] for choice in delivery_choices()}
            if delivery not in valid:
                raise ValueError(
                    f"Destino de entrega desconocido: {delivery!r}. "
                    f"Validos: {', '.join(sorted(str(v) for v in valid))}."
                )
        # Pedir ajuste de loudness sin destino no tiene numero al que ajustar.
        # Se rechaza en vez de inventar uno o de descartarlo en silencio.
        if "loudness" in selected and delivery is None:
            raise ValueError(
                "El paso de ajuste de loudness necesita un destino de entrega."
            )
        return list(selected)

    def _validate_output_format(self, output_format: str) -> None:
        if output_format not in AUDIO_OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {sorted(AUDIO_OUTPUT_FORMATS)}")

    async def _validate_device(self, device: str | None) -> None:
        if device is None:
            return
        if device == AUTO_DEVICE_ID:
            raise ValueError("device 'auto' is not supported for audio jobs; pin a concrete device (cpu|dml:N)")
        if self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)

    def _admit(self, job: AudioJob):
        return self.device_semaphores.acquire(job.device)

    async def _run_engine(self, job: AudioJob) -> None:
        job.output_path = await self.pipeline.run(job)

    def _cleanup_source(self, job: AudioJob) -> None:
        self._unlink_source_safely(job.source_path)

    @staticmethod
    def _unlink_source_safely(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete source upload %s", source_path)
