from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.models import UpscaleJob
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
from app.services.device_router import DeviceRouter, has_compatible_device
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.base import UpscaleEngine
from app.services.engines.photo_restore_engine import run_cancellable
from app.services.classic_upscalers import is_classic_upscaler
from app.services.job_manager_base import QueuedJobManager
from app.services.model_registry import ModelKind, ModelRegistry, ModelStatus, is_generative
from app.services.photo_geometry import Geometry
from app.services.photo_restore_job import (
    SR_INPUT_NAME,
    SR_OUTPUT_NAME,
    UPSCALE_AI,
    UPSCALE_CLASSIC,
    UPSCALE_NONE,
    PhotoRestoreJobRunner,
    PreStage,
    RestoreSelection,
    classic_upscale,
    load_sr_output,
    move_into,
    remove_work_dir,
    restore_upscale_mode,
    restore_uses_model,
    restore_work_dir,
    run_shielded,
    validate_restore_selection,
    write_sr_input,
)
from app.services.photo_restore_pipeline import PixelLimits, check_pixel_limits, preview_region
from app.services.progress import (
    UPSCALING_STAGE,
    advance_image_stage,
    complete_image_stages,
    enter_image_stage,
    running_image_stage,
)
from app.services.restore_provenance import UpscaleInfo
from app.services.scale_fit import effective_scale, fit_output_to_scale, native_scale_for_engine_model
from app.services.tile_params import validate_tile_params

ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}
# TIFF entra solo con pasos de restauracion (escaneos); el reescalado puro no cambia.
ALLOWED_RESTORE_FORMATS = ALLOWED_IMAGE_FORMATS | {"TIFF"}
OUTPUT_FORMATS = {"png", "jpg", "jpeg", "webp"}
CPU_DEVICE = "cpu"
RESTORE_ONLY_SCALE = 1
EXIF_ORIENTATION_TAG = 0x0112
TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelResolution:
    model_id: str
    engine_model_name: str
    kind: ModelKind
    scale: int
    native_scale: int


def select_upscale_engine(
    job: UpscaleJob,
    registry: ModelRegistry | None,
    builtin_engine: UpscaleEngine,
    onnx_engine: UpscaleEngine | None,
) -> UpscaleEngine:
    if job.model_id is not None and registry is not None:
        entry = registry.get(job.model_id)
        if entry is not None and entry.kind == ModelKind.onnx:
            if onnx_engine is None:
                raise RuntimeError(
                    f"Model {job.model_id!r} requires the ONNX engine, which is not configured"
                )
            return onnx_engine
    return builtin_engine


class JobManager(QueuedJobManager[UpscaleJob]):
    queue_full_message = "Job queue is full; try again later"
    worker_name_prefix = "upscale-worker"

    def __init__(
        self,
        settings: Settings,
        engine: UpscaleEngine,
        device_semaphores: DeviceSemaphores,
        *,
        onnx_engine: UpscaleEngine | None = None,
        registry: ModelRegistry | None = None,
        devices: DevicesService | None = None,
        device_router: DeviceRouter | None = None,
        quota_service: QuotaService | None = None,
        restore_runner: PhotoRestoreJobRunner | None = None,
    ) -> None:
        super().__init__(
            settings,
            quota_service=quota_service,
            worker_count=settings.max_concurrent_jobs,
        )
        self.engine = engine
        self.onnx_engine = onnx_engine
        self.registry = registry
        self.devices = devices
        self.device_semaphores = device_semaphores
        self.device_router = device_router or DeviceRouter(device_semaphores)
        self.restore_runner = restore_runner

    async def create_job(
        self,
        *,
        source_path: Path,
        original_filename: str,
        model_name: str,
        scale: int,
        output_format: str,
        model_id: str | None = None,
        device: str | None = None,
        job_id: str | None = None,
        owner: AuthenticatedUser | None = None,
        tile_size: int | None = None,
        tile_overlap: int | None = None,
        restore_steps: Sequence[str] | None = None,
        restore_options: Mapping[str, Any] | None = None,
        restore_session: str | None = None,
    ) -> UpscaleJob:
        restore = self._restore_selection(restore_steps, restore_options, restore_session, scale)
        await asyncio.to_thread(self._validate_input_image, source_path, restore)
        validate_tile_params(tile_size, tile_overlap)
        resolved_model_id = model_id if model_id is not None else model_name
        if device is not None and device != AUTO_DEVICE_ID and self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)
        resolution = self._resolution_for(resolved_model_id, scale, output_format, device, restore)
        job = UpscaleJob(
            source_path=source_path,
            original_filename=original_filename,
            model_name=model_name if resolution is None else resolution.engine_model_name,
            scale=scale if resolution is None else resolution.scale,
            output_format=output_format,
            model_id=None if resolution is None else resolution.model_id,
            device=device,
            native_scale=scale if resolution is None else resolution.native_scale,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            owner_id=owner.id if owner is not None else None,
            **restore_job_fields(restore),
        )
        if device == AUTO_DEVICE_ID:
            await self._validate_auto_kind(self._job_kind(job))
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)
        if job_id is not None:
            job.id = job_id
        self._enqueue(job)
        self.jobs[job.id] = job
        return job

    def _restore_selection(
        self,
        steps: Sequence[str] | None,
        options: Mapping[str, Any] | None,
        session: str | None,
        scale: int,
    ) -> RestoreSelection | None:
        if not steps:
            if options or session is not None:
                raise ValueError("Restore options need at least one restore step")
            return None
        if self.restore_runner is None:
            raise ValueError("Photo restoration is not configured on this server")
        return validate_restore_selection(
            self.settings,
            steps,
            options or {},
            scale,
            session=session,
            check_ready=self.restore_runner.check_ready,
            check_session=self.restore_runner.check_session,
        )

    def _resolution_for(
        self,
        model_id: str,
        scale: int,
        output_format: str,
        device: str | None,
        restore: RestoreSelection | None,
    ) -> ModelResolution | None:
        if restore is None:
            return self._resolve_model(model_id=model_id, scale=scale, output_format=output_format, device=device)
        self._validate_scale(scale, has_restore=True)
        validate_output_format(output_format)
        # Sin reescalado IA no hay motor SR que resolver: "none" (escala 1) o Lanczos en CPU.
        if restore.upscale_mode != UPSCALE_AI:
            return None
        return self._resolve_model(model_id=model_id, scale=scale, output_format=output_format, device=device)

    def _validate_scale(self, scale: int, has_restore: bool) -> None:
        allowed = self.settings.allowed_scale_values
        if has_restore:
            allowed = sorted({RESTORE_ONLY_SCALE, *allowed})
        if scale not in allowed:
            raise ValueError(f"Scale must be one of {allowed}")

    def _resolve_model(
        self, *, model_id: str, scale: int, output_format: str, device: str | None
    ) -> ModelResolution:
        self._validate_scale(scale, has_restore=False)
        validate_output_format(output_format)
        if not model_id.strip():
            raise ValueError("Model id is required")
        if is_classic_upscaler(model_id):
            # El reescalado clasico lo hace swscale DENTRO del encode de video, y el
            # pipeline de imagen no tiene ese paso. Sin este rechazo el job se encolaria
            # y fallaria en el motor ncnn buscando un modelo llamado "lanczos".
            raise ValueError(
                f"Classic upscaler {model_id!r} is only available for video jobs "
                "(image upscaling has no resize stage yet)"
            )

        if model_id in self.settings.model_keys:
            return self._resolve_builtin_model(model_id, scale, device)
        return self._resolve_onnx_model(model_id, scale)

    def _resolve_builtin_model(self, model_id: str, scale: int, device: str | None) -> ModelResolution:
        option = self.settings.get_model_option(model_id)
        if option and scale not in option["scales"]:
            raise ValueError(f"Model {model_id} supports only scales {option['scales']}")
        if device == "cpu":
            raise ValueError(
                f"Device 'cpu' is not supported for builtin model {model_id!r} (requires a Vulkan GPU device)"
            )
        engine_model_name = self.settings.resolve_engine_model_name(model_id, scale)
        # El binario ncnn solo sabe la escala del modelo (x4plus = 4): 2x/3x se
        # resuelven reduciendo su salida nativa, nunca con `-s` (ver scale_fit.py).
        native_scale = native_scale_for_engine_model(engine_model_name)
        return ModelResolution(
            model_id=model_id,
            engine_model_name=engine_model_name,
            kind=ModelKind.builtin_ncnn,
            scale=effective_scale(scale, native_scale),
            native_scale=native_scale,
        )

    def _resolve_onnx_model(self, model_id: str, scale: int) -> ModelResolution:
        if self.registry is None:
            raise ValueError(f"Model must be one of {sorted(self.settings.model_keys)}")
        entry = self.registry.get(model_id)
        if entry is None or entry.kind != ModelKind.onnx:
            raise ValueError(f"Unknown model id: {model_id!r}")
        if entry.status != ModelStatus.installed:
            raise ValueError(f"Model {model_id!r} is not ready for inference (status={entry.status.value})")
        # An onnx model's real up-ratio is whatever its weights produce
        # (entry.scale, detected at install time). A SMALLER requested scale is
        # honored by downscaling the native output; a larger one is not
        # invented -- the job runs at the model's scale, as before.
        native_scale = entry.scale or scale
        return ModelResolution(
            model_id=model_id,
            engine_model_name=model_id,
            kind=ModelKind.onnx,
            scale=effective_scale(scale, native_scale),
            native_scale=native_scale,
        )

    def _select_engine(self, job: UpscaleJob) -> UpscaleEngine:
        return select_upscale_engine(job, self.registry, self.engine, self.onnx_engine)

    async def _validate_auto_kind(self, kind: ModelKind | None) -> None:
        # None = restauracion solo DSP: corre en CPU, que siempre existe.
        if kind is not None:
            await self._validate_auto_device(kind)

    async def _validate_auto_device(self, kind: ModelKind) -> None:
        if self.devices is None:
            raise ValueError("Device 'auto' requires a devices service to be configured")
        devices = await asyncio.to_thread(self.devices.list_devices)
        if not has_compatible_device(devices, kind):
            raise ValueError(
                f"No compatible device available for model kind {kind.value!r} (requested device='auto')"
            )

    def _job_kind(self, job: UpscaleJob) -> ModelKind | None:
        return self._restore_kind(job) if job.restore_steps else self._model_kind_for_job(job)

    def _restore_kind(self, job: UpscaleJob) -> ModelKind | None:
        mode = restore_upscale_mode(job.restore_options, job.scale)
        sr_kind = self._model_kind_for_job(job) if mode == UPSCALE_AI else None
        if sr_kind == ModelKind.builtin_ncnn:
            # ncnn exige una GPU Vulkan; los pasos ONNX corren en esa misma GPU.
            return sr_kind
        if sr_kind is not None or restore_uses_model(job.restore_steps, job.restore_options):
            return ModelKind.onnx
        return None

    def _model_kind_for_job(self, job: UpscaleJob) -> ModelKind:
        if job.model_id in self.settings.model_keys:
            return ModelKind.builtin_ncnn
        if self.registry is not None:
            entry = self.registry.get(job.model_id) if job.model_id is not None else None
            if entry is not None:
                return entry.kind
        raise ValueError(f"Cannot resolve model kind for job (model_id={job.model_id!r})")

    def _validate_input_image(self, source_path: Path, restore: RestoreSelection | None = None) -> None:
        try:
            with Image.open(source_path) as img:
                if restore is None:
                    self._validate_upscale_image(img)
                else:
                    self._validate_restore_image(img, restore)
        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file is not a valid image") from exc
        except Image.DecompressionBombError as exc:
            raise ValueError("Uploaded image exceeds the maximum allowed dimensions") from exc

    def _validate_upscale_image(self, img: Image.Image) -> None:
        self._validate_image_format(img)
        width, height = img.size
        if width * height > self.settings.max_image_pixels:
            raise ValueError(f"Image is too large. Maximum pixels allowed: {self.settings.max_image_pixels}")

    def _validate_restore_image(self, img: Image.Image, restore: RestoreSelection) -> None:
        self._validate_image_format(img, ALLOWED_RESTORE_FORMATS)
        height, width = Geometry.from_mapping(restore.options.get("geometry")).output_size(*oriented_size(img))
        check_pixel_limits(height, width, float(restore.scale), PixelLimits.from_settings(self.settings))
        crop = restore.options.get("preview_crop")
        if crop is not None:
            preview_region((height, width), tuple(int(value) for value in crop))

    @staticmethod
    def _validate_image_format(img: Image.Image, allowed: set[str] = ALLOWED_IMAGE_FORMATS) -> None:
        if img.format not in allowed:
            raise ValueError(f"Unsupported image format: {img.format}. Allowed formats: {sorted(allowed)}")

    async def _dispatch(self, job: UpscaleJob) -> None:
        if job.device == AUTO_DEVICE_ID:
            await self._run_auto_job(job)
        else:
            await self._run_pinned_job(job)

    async def _run_pinned_job(self, job: UpscaleJob) -> None:
        async with self.device_semaphores.acquire(job.device):
            await self._execute_job(job)

    async def _run_auto_job(self, job: UpscaleJob) -> None:
        # Device resolution (kind lookup + hardware enumeration) happens
        # BEFORE any semaphore/router acquire and is guarded on its own, so a
        # failure here (e.g. hardware changed since create_job's own
        # compatibility check) fails the job cleanly instead of leaving it
        # stuck at status=queued forever with task_done() never called.
        try:
            kind = self._job_kind(job)
            devices = None if kind is None else await asyncio.to_thread(self.devices.list_devices)
        except Exception as exc:  # noqa: BLE001
            self._fail_dequeued_job(job, str(exc))
            return
        if kind is None:
            # Restauracion solo DSP: CPU sin tomar el permiso de ninguna GPU.
            job.device = CPU_DEVICE
            await self._run_pinned_job(job)
            return
        try:
            async with self.device_router.acquire_auto(devices, kind) as device_id:
                job.device = device_id
                await self._execute_job(job)
        except ValueError as exc:
            self._fail_dequeued_job(job, str(exc))

    def _on_running(self, job: UpscaleJob) -> None:
        advance_image_stage(job, running_image_stage(job))

    def _on_completed(self, job: UpscaleJob) -> None:
        complete_image_stages(job)

    def _cleanup_source(self, job: UpscaleJob) -> None:
        # El original de una sesion de analisis es de la sesion: otro job (la vista previa y
        # despues la foto entera) lo vuelve a usar, y el barrido la borra por edad.
        if job.restore_session is None:
            self._unlink_source_safely(job.source_path)

    async def _run_engine(self, job: UpscaleJob) -> None:
        if job.restore_steps:
            await self._run_restore(job)
            return
        engine = self._select_engine(job)
        native_output = await engine.run(job)
        job.output_path = await asyncio.to_thread(fit_output_to_scale, native_output, job, self.settings)

    async def _run_restore(self, job: UpscaleJob) -> None:
        runner = self._require_restore_runner()
        work_dir = restore_work_dir(self.settings, job.id)
        try:
            stage = await run_cancellable(runner.run_pre, job)
            upscaled = await self._restore_upscale(job, stage, work_dir)
            upscale = self._upscale_info(job)
            job.output_path = await run_cancellable(runner.finish, job, stage, upscaled, upscale)
        finally:
            await asyncio.to_thread(remove_work_dir, work_dir)

    async def _restore_upscale(self, job: UpscaleJob, stage: PreStage, work_dir: Path) -> np.ndarray | None:
        mode = restore_upscale_mode(job.restore_options, job.scale)
        if mode == UPSCALE_NONE:
            return None
        enter_image_stage(job, UPSCALING_STAGE)
        if mode == UPSCALE_CLASSIC:
            return await run_cancellable(classic_upscale, stage.pre.image, float(job.scale))
        return await self._ai_upscale(job, stage.pre.image, work_dir)

    async def _ai_upscale(self, job: UpscaleJob, image: np.ndarray, work_dir: Path) -> np.ndarray:
        source = await asyncio.to_thread(write_sr_input, image, work_dir / SR_INPUT_NAME)
        # replace copia el id y COMPARTE el dict metadata: el progreso del SR cae en el job.
        derived = replace(job, source_path=source, output_format="png")
        engine = self._select_engine(derived)
        if engine is self.engine:
            await asyncio.to_thread(self._require_restore_runner().release_before_ncnn, str(job.device))
        sr_path = await run_shielded(self._upscale_into(engine, derived, work_dir / SR_OUTPUT_NAME))
        return await asyncio.to_thread(load_sr_output, sr_path)

    async def _upscale_into(self, engine: UpscaleEngine, derived: UpscaleJob, target: Path) -> Path:
        # El motor escribe en outputs/{id}.png: se mueve ya, asi un fallo posterior no deja huerfanos.
        native_output = await engine.run(derived)
        fitted = await asyncio.to_thread(fit_output_to_scale, native_output, derived, self.settings)
        return await asyncio.to_thread(move_into, fitted, target)

    def _upscale_info(self, job: UpscaleJob) -> UpscaleInfo:
        mode = restore_upscale_mode(job.restore_options, job.scale)
        if mode != UPSCALE_AI:
            return UpscaleInfo(mode=mode, scale=float(job.scale))
        return UpscaleInfo(
            mode=mode,
            scale=float(job.scale),
            model=job.model_id,
            generative=self._is_generative_model(job.model_id),
            backend=self._model_kind_for_job(job).value,
        )

    def _is_generative_model(self, model_id: str | None) -> bool:
        if self.registry is None or model_id is None:
            return is_generative(None)
        return is_generative(self.registry.get(model_id))

    def _require_restore_runner(self) -> PhotoRestoreJobRunner:
        if self.restore_runner is None:
            raise RuntimeError("Photo restoration is not configured on this server")
        return self.restore_runner

    def resolve_model(
        self, *, model_id: str, scale: int, output_format: str, device: str | None
    ) -> ModelResolution:
        return self._resolve_model(model_id=model_id, scale=scale, output_format=output_format, device=device)

    async def run_inline(self, job: UpscaleJob) -> Path:
        """Corre un job ya resuelto en el hilo que llama, sin cola ni workers.

        Es la puerta del modo headless (CLI / MCP in-process): mismo motor, misma
        reduccion a la escala pedida y misma metadata `effective` que un job encolado.
        """
        await self._run_engine(job)
        assert job.output_path is not None
        return job.output_path

    @staticmethod
    def _unlink_source_safely(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete source upload %s", source_path)


def validate_output_format(output_format: str) -> None:
    if output_format.lower() not in OUTPUT_FORMATS:
        raise ValueError("Output format must be png, jpg, jpeg, or webp")


def restore_job_fields(restore: RestoreSelection | None) -> dict[str, Any]:
    if restore is None:
        return {}
    return {
        "restore_steps": list(restore.steps),
        "restore_options": dict(restore.options),
        "restore_session": restore.session,
    }


def oriented_size(img: Image.Image) -> tuple[int, int]:
    width, height = img.size
    # La restauracion aplica la orientacion EXIF al cargar: los topes y el recorte van en ese marco.
    if img.getexif().get(EXIF_ORIENTATION_TAG) in TRANSPOSED_ORIENTATIONS:
        return width, height
    return height, width
