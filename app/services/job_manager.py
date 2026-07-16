from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.exceptions import QueueFullError
from app.models import JobStatus, UpscaleJob, utc_now
from app.services.devices_service import DevicesService
from app.services.engines.base import UpscaleEngine
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


class JobManager:
    def __init__(
        self,
        settings: Settings,
        engine: UpscaleEngine,
        gpu_semaphore: asyncio.Semaphore,
        *,
        onnx_engine: UpscaleEngine | None = None,
        registry: ModelRegistry | None = None,
        devices: DevicesService | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.onnx_engine = onnx_engine
        self.registry = registry
        self.devices = devices
        self.jobs: dict[str, UpscaleJob] = {}
        self.queue: asyncio.Queue[UpscaleJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self.gpu_semaphore = gpu_semaphore
        self.worker_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        if self.worker_tasks:
            return
        self.worker_tasks = [
            asyncio.create_task(self._worker(), name=f"upscale-worker-{i}")
            for i in range(self.settings.gpu_concurrency)
        ]

    async def stop(self) -> None:
        for task in self.worker_tasks:
            task.cancel()
        for task in self.worker_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.worker_tasks = []

    def queue_depth(self) -> int:
        return self.queue.qsize()

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
    ) -> UpscaleJob:
        await asyncio.to_thread(self._validate_input_image, source_path)
        resolved_model_id = model_id if model_id is not None else model_name
        if device is not None and self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)
        resolution = self._resolve_model(
            model_id=resolved_model_id,
            scale=scale,
            output_format=output_format,
            device=device,
        )

        job = UpscaleJob(
            source_path=source_path,
            original_filename=original_filename,
            model_name=resolution.engine_model_name,
            scale=resolution.scale,
            output_format=output_format,
            model_id=resolution.model_id,
            device=device,
        )
        if job_id is not None:
            job.id = job_id
        self._enqueue(job)
        self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> UpscaleJob | None:
        return self.jobs.get(job_id)

    def _enqueue(self, job: UpscaleJob) -> None:
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError("Job queue is full; try again later") from exc

    def _resolve_model(
        self, *, model_id: str, scale: int, output_format: str, device: str | None
    ) -> ModelResolution:
        if scale not in self.settings.allowed_scale_values:
            raise ValueError(f"Scale must be one of {self.settings.allowed_scale_values}")
        if output_format.lower() not in {"png", "jpg", "jpeg", "webp"}:
            raise ValueError("Output format must be png, jpg, jpeg, or webp")
        if not model_id.strip():
            raise ValueError("Model id is required")

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
        if job.model_id is not None and self.registry is not None:
            entry = self.registry.get(job.model_id)
            if entry is not None and entry.kind == ModelKind.onnx:
                if self.onnx_engine is None:
                    raise RuntimeError(
                        f"Model {job.model_id!r} requires the ONNX engine, which is not configured"
                    )
                return self.onnx_engine
        return self.engine

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

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            async with self.gpu_semaphore:
                job.status = JobStatus.running
                job.started_at = utc_now()
                advance_image_stage(job, "upscaling")
                try:
                    engine = self._select_engine(job)
                    job.output_path = await engine.run(job)
                    job.status = JobStatus.completed
                    complete_image_stages(job)
                except asyncio.CancelledError:
                    job.status = JobStatus.failed
                    job.error = "Job cancelled"
                    raise
                except Exception as exc:  # noqa: BLE001
                    job.status = JobStatus.failed
                    job.error = str(exc)
                finally:
                    job.finished_at = utc_now()
                    self._unlink_source_safely(job.source_path)
                    self.queue.task_done()

    @staticmethod
    def _unlink_source_safely(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete source upload %s", source_path)
