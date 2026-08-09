from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.models import UpscaleJob
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
from app.services.device_router import DeviceRouter, has_compatible_device
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.base import UpscaleEngine
from app.services.classic_upscalers import is_classic_upscaler
from app.services.job_manager_base import QueuedJobManager
from app.services.model_registry import ModelKind, ModelRegistry, ModelStatus
from app.services.progress import advance_image_stage, complete_image_stages

ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelResolution:
    model_id: str
    engine_model_name: str
    kind: ModelKind
    scale: int


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
    ) -> UpscaleJob:
        await asyncio.to_thread(self._validate_input_image, source_path)
        resolved_model_id = model_id if model_id is not None else model_name
        if device is not None and device != AUTO_DEVICE_ID and self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)
        resolution = self._resolve_model(
            model_id=resolved_model_id,
            scale=scale,
            output_format=output_format,
            device=device,
        )
        if device == AUTO_DEVICE_ID:
            await self._validate_auto_device(resolution.kind)
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)

        job = UpscaleJob(
            source_path=source_path,
            original_filename=original_filename,
            model_name=resolution.engine_model_name,
            scale=resolution.scale,
            output_format=output_format,
            model_id=resolution.model_id,
            device=device,
            owner_id=owner.id if owner is not None else None,
        )
        if job_id is not None:
            job.id = job_id
        self._enqueue(job)
        self.jobs[job.id] = job
        return job

    def _resolve_model(
        self, *, model_id: str, scale: int, output_format: str, device: str | None
    ) -> ModelResolution:
        if scale not in self.settings.allowed_scale_values:
            raise ValueError(f"Scale must be one of {self.settings.allowed_scale_values}")
        if output_format.lower() not in {"png", "jpg", "jpeg", "webp"}:
            raise ValueError("Output format must be png, jpg, jpeg, or webp")
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
        return self._resolve_onnx_model(model_id)

    def _resolve_builtin_model(self, model_id: str, scale: int, device: str | None) -> ModelResolution:
        option = self.settings.get_model_option(model_id)
        if option and scale not in option["scales"]:
            raise ValueError(f"Model {model_id} supports only scales {option['scales']}")
        if device == "cpu":
            raise ValueError(
                f"Device 'cpu' is not supported for builtin model {model_id!r} (requires a Vulkan GPU device)"
            )
        engine_model_name = self.settings.resolve_engine_model_name(model_id, scale)
        return ModelResolution(
            model_id=model_id, engine_model_name=engine_model_name, kind=ModelKind.builtin_ncnn, scale=scale
        )

    def _resolve_onnx_model(self, model_id: str) -> ModelResolution:
        if self.registry is None:
            raise ValueError(f"Model must be one of {sorted(self.settings.model_keys)}")
        entry = self.registry.get(model_id)
        if entry is None or entry.kind != ModelKind.onnx:
            raise ValueError(f"Unknown model id: {model_id!r}")
        if entry.status != ModelStatus.installed:
            raise ValueError(f"Model {model_id!r} is not ready for inference (status={entry.status.value})")
        # The requested scale is only used to pick a builtin engine variant;
        # an onnx model's real up-ratio is whatever its weights produce
        # (entry.scale, detected at install time), so it must win here --
        # otherwise a scale/model mismatch silently corrupts derived metadata
        # like video outputWidth/outputHeight (computed from job.scale).
        return ModelResolution(model_id=model_id, engine_model_name=model_id, kind=ModelKind.onnx, scale=entry.scale)

    def _select_engine(self, job: UpscaleJob) -> UpscaleEngine:
        return select_upscale_engine(job, self.registry, self.engine, self.onnx_engine)

    async def _validate_auto_device(self, kind: ModelKind) -> None:
        if self.devices is None:
            raise ValueError("Device 'auto' requires a devices service to be configured")
        devices = await asyncio.to_thread(self.devices.list_devices)
        if not has_compatible_device(devices, kind):
            raise ValueError(
                f"No compatible device available for model kind {kind.value!r} (requested device='auto')"
            )

    def _model_kind_for_job(self, job: UpscaleJob) -> ModelKind:
        if job.model_id in self.settings.model_keys:
            return ModelKind.builtin_ncnn
        if self.registry is not None:
            entry = self.registry.get(job.model_id) if job.model_id is not None else None
            if entry is not None:
                return entry.kind
        raise ValueError(f"Cannot resolve model kind for job (model_id={job.model_id!r})")

    def _validate_input_image(self, source_path: Path) -> None:
        try:
            with Image.open(source_path) as img:
                self._validate_image_format(img)
                width, height = img.size
                if width * height > self.settings.max_image_pixels:
                    raise ValueError(
                        f"Image is too large. Maximum pixels allowed: {self.settings.max_image_pixels}"
                    )
        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file is not a valid image") from exc
        except Image.DecompressionBombError as exc:
            raise ValueError("Uploaded image exceeds the maximum allowed dimensions") from exc

    @staticmethod
    def _validate_image_format(img: Image.Image) -> None:
        if img.format not in ALLOWED_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format: {img.format}. Allowed formats: {sorted(ALLOWED_IMAGE_FORMATS)}"
            )

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
            kind = self._model_kind_for_job(job)
            devices = await asyncio.to_thread(self.devices.list_devices)
        except Exception as exc:  # noqa: BLE001
            self._fail_dequeued_job(job, str(exc))
            return
        try:
            async with self.device_router.acquire_auto(devices, kind) as device_id:
                job.device = device_id
                await self._execute_job(job)
        except ValueError as exc:
            self._fail_dequeued_job(job, str(exc))

    def _on_running(self, job: UpscaleJob) -> None:
        advance_image_stage(job, "upscaling")

    def _on_completed(self, job: UpscaleJob) -> None:
        complete_image_stages(job)

    def _cleanup_source(self, job: UpscaleJob) -> None:
        self._unlink_source_safely(job.source_path)

    async def _run_engine(self, job: UpscaleJob) -> None:
        engine = self._select_engine(job)
        job.output_path = await engine.run(job)

    @staticmethod
    def _unlink_source_safely(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete source upload %s", source_path)
