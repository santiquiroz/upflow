from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.config import AUDIO_ENHANCE_MODES, AUDIO_OUTPUT_FORMATS, AUDIO_RESTORE_MODES, Settings
from app.models import AudioJob
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
from app.services.audio_conversion import (
    AUDIO_LOSSY_QUALITIES,
    DEFAULT_LOSSY_QUALITY,
    unambiguous_source_format,
)
from app.services.audio_pipeline import AudioPipeline
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.job_manager_base import QueuedJobManager
from app.services.missing_pack import missing_pack_message
from app.services.restorer_registry import validate_restore_mode_ready

logger = logging.getLogger(__name__)

# Tres es el techo por lo mismo que en la comparacion: cada modelo mas es una
# pasada completa sobre el tema, y la mejora del cuarto no se oye.
MAX_ENSEMBLE_MODELS = 3


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
        lossy_quality: str = DEFAULT_LOSSY_QUALITY,
        voice_steps: list[str] | None = None,
        voice_delivery: str | None = None,
        voice_presence_db: float | None = None,
        master: str | None = None,
        cleanup_steps: list[str] | None = None,
        separate: bool = False,
        separation_model: str | None = None,
        ensemble_models: list[str] | None = None,
        job_id: str | None = None,
        owner: AuthenticatedUser | None = None,
    ) -> AudioJob:
        # El formato se valida PRIMERO: sin pasos pedidos, el resto de la
        # validacion razona sobre el (un job de pura conversion), y un formato
        # inexistente ahi daria un mensaje sobre pasos en vez de sobre formatos.
        self._validate_output_format(output_format)
        self._validate_lossy_quality(lossy_quality)
        if separate:
            separation_model = self._validate_separation(
                separation_model,
                denoise=denoise,
                restore=restore,
                voice_steps=voice_steps or [],
                master=master,
                voice_delivery=voice_delivery,
                voice_presence_db=voice_presence_db,
                cleanup_steps=cleanup_steps or [],
            )
            selected_ensemble = self._validate_ensemble(separation_model, ensemble_models or [])
            selected_voice_steps: list[str] = []
            selected_cleanup_steps: list[str] = []
        else:
            if separation_model is not None:
                raise ValueError(
                    "separation_model solo aplica cuando separate=true."
                )
            if ensemble_models:
                raise ValueError("ensemble_models solo aplica cuando separate=true.")
            selected_ensemble = []
            selected_cleanup_steps = self._validate_cleanup_selection(cleanup_steps or [])
            selected_voice_steps = self._validate_voice_selection(
                voice_steps or [], voice_delivery
            )
            self._validate_modes(
                denoise,
                restore,
                has_other_work=bool(selected_cleanup_steps or selected_voice_steps or master),
                original_filename=original_filename,
                output_format=output_format,
            )
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
            lossy_quality=lossy_quality,
            voice_steps=selected_voice_steps,
            voice_delivery=voice_delivery,
            voice_presence_db=voice_presence_db,
            master=master,
            cleanup_steps=selected_cleanup_steps,
            separate=separate,
            separation_model=separation_model,
            ensemble_models=selected_ensemble,
            owner_id=owner.id if owner is not None else None,
        )
        if job_id is not None:
            job.id = job_id
        self._enqueue(job)
        self.jobs[job.id] = job
        return job

    def _validate_separation(
        self,
        separation_model: str | None,
        *,
        denoise: str | None,
        restore: str | None,
        voice_steps: list[str],
        master: str | None,
        voice_delivery: str | None,
        voice_presence_db: float | None,
        cleanup_steps: list[str],
    ) -> str:
        from app.services.engines.separation_models import SEPARATION_MODELS

        # La cadena de limpieza tiene su propio motivo, distinto del de los
        # demas pasos: no es que se aplicaria a un stem ambiguo, es que las dos
        # cosas tienen FORMA de salida distinta — la separacion entrega dos
        # archivos y la cadena uno. Mezclarlas no tiene una respuesta correcta.
        if cleanup_steps:
            raise ValueError(
                "La separacion entrega DOS archivos y la cadena de limpieza UNO; "
                "pedir las dos en el mismo trabajo no define que se entrega. "
                "Corre la separacion y despues la limpieza sobre el stem que "
                "quieras, o pedi la limpieza sola."
            )
        # voice_delivery por truthiness ("" de un form cuenta como ausente);
        # voice_presence_db por is-not-None (un 0.0 explicito tambien se rechaza).
        if (
            denoise
            or restore
            or voice_steps
            or master
            or voice_delivery
            or voice_presence_db is not None
        ):
            raise ValueError(
                "El modo karaoke corre solo; los demas pasos se aplicarian a un "
                "stem ambiguo. Pedilos en un segundo trabajo sobre el stem que "
                "quieras."
            )
        model_id = separation_model or self._default_installed_model()
        if model_id not in SEPARATION_MODELS:
            known = ", ".join(sorted(SEPARATION_MODELS))
            raise ValueError(
                f"Modelo de separacion desconocido: {model_id!r}. Validos: {known}."
            )
        if model_id not in self.settings.karaoke_installed_models():
            raise ValueError(missing_pack_message("karaoke", variant=model_id))
        return model_id

    def _validate_ensemble(self, primary: str, extras: list[str]) -> list[str]:
        """Los modelos extra a combinar, o vacio.

        La regla dura es que TODOS declaren los mismos stems: promediar un
        instrumental con un bajo no da un instrumental mejor, da una suma sin
        sentido que ademas nadie puede etiquetar.
        """
        from app.services.engines.separation_models import SEPARATION_MODELS

        elegidos = [model_id for model_id in dict.fromkeys(extras) if model_id != primary]
        if not elegidos:
            return []
        if len(elegidos) + 1 > MAX_ENSEMBLE_MODELS:
            raise ValueError(
                f"Se pueden combinar hasta {MAX_ENSEMBLE_MODELS} modelos; "
                "mas es esperar el doble por una diferencia que no se oye."
            )
        instalados = self.settings.karaoke_installed_models()
        stems_esperados = SEPARATION_MODELS[primary].stem_ids()
        for model_id in elegidos:
            spec = SEPARATION_MODELS.get(model_id)
            if spec is None:
                known = ", ".join(sorted(SEPARATION_MODELS))
                raise ValueError(
                    f"Modelo de separacion desconocido: {model_id!r}. Validos: {known}."
                )
            if model_id not in instalados:
                raise ValueError(missing_pack_message("karaoke", variant=model_id))
            if spec.stem_ids() != stems_esperados:
                raise ValueError(
                    f"{model_id!r} entrega {spec.stem_ids()} y {primary!r} entrega "
                    f"{stems_esperados}: solo se combinan modelos con las mismas pistas."
                )
        return elegidos

    def _default_installed_model(self) -> str:
        """Vacio = primer modelo instalado: la capability marca disponible con
        cualquier modelo, asi que el default fijo daria un 400 enganoso. Sin
        ninguno instalado cae al default del catalogo (mensaje missing-pack)."""
        from app.services.engines.separation_models import DEFAULT_SEPARATION_MODEL

        installed = self.settings.karaoke_installed_models()
        return installed[0] if installed else DEFAULT_SEPARATION_MODEL

    def _validate_cleanup_selection(self, selected: list[str]) -> list[str]:
        """Los pasos de limpieza YA normalizados: en orden de catalogo y sin
        redundancias. Se guardan asi en el job para que pipeline, mapa de etapas
        y respuesta de la API vean exactamente la misma cadena."""
        from app.services.cleanup_chain import cleanup_steps_from_selection

        # UnknownCleanupStep y RedundantCleanupSelection son ValueError: la ruta
        # ya los convierte en 400 con su mensaje, sin traducirlos aca.
        steps = cleanup_steps_from_selection(selected)
        installed = set(self.settings.karaoke_installed_models())
        missing = [step.model_id for step in steps if step.model_id not in installed]
        if missing:
            raise ValueError(missing_pack_message("karaoke", variant=missing[0]))
        return [step.model_id for step in steps]

    def _validate_modes(
        self,
        denoise: str | None,
        restore: str | None,
        *,
        has_other_work: bool = False,
        original_filename: str,
        output_format: str,
    ) -> None:
        # Un job de limpieza, de voz o de mastering solo es una entrega valida:
        # el pedido es que el archivo pase por ALGO, no que pase por denoise o
        # restore en particular. Y un job SIN ningun paso tambien lo es, si lo
        # que se pide es cambiar de formato.
        if denoise is None and restore is None and not has_other_work:
            self._validate_conversion_only(original_filename, output_format)
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

    def _validate_conversion_only(self, original_filename: str, output_format: str) -> None:
        """Sin pasos, lo unico que queda es convertir — y solo si hay a que.

        Se rechaza en vez de copiar el archivo: copiarlo ocuparia una plaza de
        la cola y una descarga para devolver el mismo archivo, y quien lo pidio
        creeria que algo paso. Un 400 con el motivo se entiende de una.

        La comparacion va contra la EXTENSION y no contra un ffprobe porque solo
        necesita responder "esto ya es lo que pediste", y las extensiones que
        consulta (wav/flac/mp3) determinan el codec sin ambiguedad. `.m4a` queda
        afuera a proposito: puede traer ALAC, y ALAC -> AAC es una conversion de
        verdad (el pipeline la resuelve copiando el stream si resulta que el
        origen ya era AAC).
        """
        if unambiguous_source_format(original_filename) == output_format:
            raise ValueError(
                f"El archivo ya esta en {output_format.upper()} y no se pidio "
                "ningun paso de procesamiento: no hay nada que hacer. Elegi otro "
                "formato de salida, o agrega limpieza, reduccion de ruido, "
                "restauracion, mejora de voz o acabado."
            )

    def _validate_output_format(self, output_format: str) -> None:
        if output_format not in AUDIO_OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {sorted(AUDIO_OUTPUT_FORMATS)}")

    def _validate_lossy_quality(self, lossy_quality: str) -> None:
        if lossy_quality not in AUDIO_LOSSY_QUALITIES:
            raise ValueError(f"lossy_quality must be one of {sorted(AUDIO_LOSSY_QUALITIES)}")

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
