from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.config import Settings
from app.models import JobStatus, UpscaleJob, utc_now
from app.services.engines.base import UpscaleEngine

ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}


class JobManager:
    def __init__(self, settings: Settings, engine: UpscaleEngine, gpu_semaphore: asyncio.Semaphore) -> None:
        self.settings = settings
        self.engine = engine
        self.jobs: dict[str, UpscaleJob] = {}
        self.queue: asyncio.Queue[UpscaleJob] = asyncio.Queue()
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
        job_id: str | None = None,
    ) -> UpscaleJob:
        await asyncio.to_thread(self._validate_input_image, source_path)
        resolved_model_name = self._validate_and_resolve_model(
            model_name=model_name,
            scale=scale,
            output_format=output_format,
        )

        job = UpscaleJob(
            source_path=source_path,
            original_filename=original_filename,
            model_name=resolved_model_name,
            scale=scale,
            output_format=output_format,
        )
        if job_id is not None:
            job.id = job_id
        self.jobs[job.id] = job
        await self.queue.put(job)
        return job

    def get_job(self, job_id: str) -> UpscaleJob | None:
        return self.jobs.get(job_id)

    def _validate_and_resolve_model(self, *, model_name: str, scale: int, output_format: str) -> str:
        if scale not in self.settings.allowed_scale_values:
            raise ValueError(f"Scale must be one of {self.settings.allowed_scale_values}")
        if output_format.lower() not in {"png", "jpg", "jpeg", "webp"}:
            raise ValueError("Output format must be png, jpg, jpeg, or webp")
        if not model_name.strip():
            raise ValueError("Model name is required")
        if model_name not in self.settings.model_keys:
            raise ValueError(f"Model must be one of {sorted(self.settings.model_keys)}")

        option = self.settings.get_model_option(model_name)
        if option and scale not in option["scales"]:
            raise ValueError(f"Model {model_name} supports only scales {option['scales']}")

        return self.settings.resolve_engine_model_name(model_name, scale)

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
                try:
                    job.output_path = await self.engine.run(job)
                    job.status = JobStatus.completed
                except asyncio.CancelledError:
                    job.status = JobStatus.failed
                    job.error = "Job cancelled"
                    raise
                except Exception as exc:  # noqa: BLE001
                    job.status = JobStatus.failed
                    job.error = str(exc)
                finally:
                    job.finished_at = utc_now()
                    job.source_path.unlink(missing_ok=True)
                    self.queue.task_done()
