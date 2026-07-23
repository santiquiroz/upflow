from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.exceptions import QueueFullError
from app.models import TERMINAL_JOB_STATUSES, GenerationJob, JobStatus, UpscaleJob, utc_now
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.generation_onnx import GenerationEngine, GenerationRequest
from app.services.job_manager import select_upscale_engine
from app.services.model_registry import ModelKind, ModelRegistry
from app.services.progress import (
    advance_generation_stage,
    apply_generation_step_progress,
    complete_generation_stages,
)

logger = logging.getLogger(__name__)

MAX_STEPS = 100
MAX_DIMENSION = 1024
MIN_DIMENSION = 64
DIMENSION_MULTIPLE = 64
UPSCALE_SCALE_RANGE = (2, 4)


class GenerationJobManager:
    """Standalone text-to-image job manager. Mirrors AudioJobManager: bounded
    queue, N workers, shared device_semaphores, CancelledError-safe execution.

    Generation jobs have no source upload (the request is JSON, not a file),
    so there is no source-unlink step in _execute_job's finally. A job may
    optionally auto-upscale its own output in the SAME job (two stages, one
    worker, no re-queue) via _run_engine -> _run_auto_upscale.

    Like audio, generation is device-pinned: the `auto` sentinel is rejected
    at create_job time rather than resolved.
    """

    def __init__(
        self,
        settings: Settings,
        engine: GenerationEngine,
        device_semaphores: DeviceSemaphores,
        *,
        registry: ModelRegistry,
        upscale_engine: Any,
        onnx_upscale_engine: Any | None = None,
        devices: DevicesService | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.device_semaphores = device_semaphores
        self.registry = registry
        self.upscale_engine = upscale_engine
        self.onnx_upscale_engine = onnx_upscale_engine
        self.devices = devices
        self.jobs: dict[str, GenerationJob] = {}
        self.queue: asyncio.Queue[GenerationJob] = asyncio.Queue(maxsize=settings.max_queue_size)
        self.worker_tasks: list[asyncio.Task] = []
        self._active: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        if self.worker_tasks:
            return
        self.worker_tasks = [
            asyncio.create_task(self._worker(), name=f"generation-worker-{i}")
            for i in range(self.settings.max_concurrent_jobs)
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
        prompt: str,
        model_id: str,
        negative_prompt: str | None = None,
        steps: int = 25,
        guidance: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
        device: str | None = None,
        auto_upscale: bool = False,
        upscale_model_name: str | None = None,
        upscale_scale: int | None = None,
        upscale_model_id: str | None = None,
        job_id: str | None = None,
    ) -> GenerationJob:
        self._validate_generation_model(model_id)
        self._validate_params(prompt, steps, width, height)
        await self._validate_device(device)
        if auto_upscale:
            self._validate_upscale_params(upscale_model_name, upscale_scale, upscale_model_id)
        job = GenerationJob(
            prompt=prompt, model_id=model_id, negative_prompt=negative_prompt, steps=steps,
            guidance=guidance, width=width, height=height, seed=seed, device=device,
            auto_upscale=auto_upscale, upscale_model_name=upscale_model_name,
            upscale_scale=upscale_scale, upscale_model_id=upscale_model_id,
        )
        if job_id is not None:
            job.id = job_id
        self.jobs[job.id] = job
        self._enqueue(job)
        return job

    def get_job(self, job_id: str) -> GenerationJob | None:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        if job.status in TERMINAL_JOB_STATUSES:
            return False
        if job.status == JobStatus.queued:
            # Still in the queue: mark it so the worker skips it on dequeue.
            job.status = JobStatus.cancelled
            job.finished_at = utc_now()
            return True
        task = self._active.get(job_id)
        if task is not None:
            task.cancel()
        return True

    def _enqueue(self, job: GenerationJob) -> None:
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError("Generation job queue is full; try again later") from exc

    def _validate_generation_model(self, model_id: str) -> None:
        entry = self.registry.get(model_id)
        if entry is None or entry.kind != ModelKind.diffusion_onnx:
            raise ValueError(f"Unknown generation model: {model_id!r}")

    def _validate_params(self, prompt: str, steps: int, width: int, height: int) -> None:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be between 1 and {MAX_STEPS}")
        for label, value in (("width", width), ("height", height)):
            if not MIN_DIMENSION <= value <= MAX_DIMENSION or value % DIMENSION_MULTIPLE:
                raise ValueError(
                    f"{label} must be a multiple of {DIMENSION_MULTIPLE} between {MIN_DIMENSION} and {MAX_DIMENSION}"
                )

    async def _validate_device(self, device: str | None) -> None:
        if device is None:
            return
        if device == AUTO_DEVICE_ID:
            raise ValueError("device 'auto' is not supported for generation jobs; pin a concrete device (cpu|dml:N)")
        if self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)

    def _validate_upscale_params(self, model_name: str | None, scale: int | None, model_id: str | None) -> None:
        if scale is None or not UPSCALE_SCALE_RANGE[0] <= scale <= UPSCALE_SCALE_RANGE[1]:
            raise ValueError("auto_upscale requires upscale_scale between 2 and 4")
        if not model_name and not model_id:
            raise ValueError("auto_upscale requires an upscale model (name or id)")
        if model_id is not None:
            entry = self.registry.get(model_id)
            if entry is None or entry.kind != ModelKind.onnx:
                raise ValueError(f"Unknown upscale model: {model_id!r}")

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            if job.status == JobStatus.cancelled:
                # Cancelled while waiting in the queue: skip without processing.
                self.queue.task_done()
                continue
            await self._run_job(job)

    async def _run_job(self, job: GenerationJob) -> None:
        async with self.device_semaphores.acquire(job.device):
            await self._execute_job(job)

    async def _execute_job(self, job: GenerationJob) -> None:
        job.status = JobStatus.running
        job.started_at = utc_now()
        run_task = asyncio.ensure_future(self._run_engine(job))
        self._active[job.id] = run_task
        try:
            await run_task
            job.status = JobStatus.completed
        except asyncio.CancelledError:
            run_task.cancel()
            if asyncio.current_task().cancelling() > 0:
                # The WORKER task itself was cancelled (shutdown via stop()):
                # fail the job and re-raise so the worker actually dies.
                job.status = JobStatus.failed
                job.error = "Job cancelled"
                raise
            # Only the child engine task was cancelled (per-job cancel_job):
            # mark cancelled and let the worker live on for other jobs.
            job.status = JobStatus.cancelled
            job.error = None
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            self._active.pop(job.id, None)
            job.finished_at = utc_now()
            self.queue.task_done()

    async def _run_engine(self, job: GenerationJob) -> None:
        entry = self.registry.get(job.model_id)
        if entry is None or entry.kind != ModelKind.diffusion_onnx:
            raise RuntimeError(f"Generation model not found: {job.model_id!r}")
        pipeline_dir = self._resolve_pipeline_dir(entry)
        device = job.device or self.settings.default_device
        include_upscale = job.auto_upscale
        advance_generation_stage(job, "generating", include_upscale)
        request = GenerationRequest(
            prompt=job.prompt, negative_prompt=job.negative_prompt, steps=job.steps,
            guidance=job.guidance, width=job.width, height=job.height, seed=job.seed,
        )

        def on_progress(done: int, total: int) -> None:
            apply_generation_step_progress(job, done, total, include_upscale)

        generated = await self.engine.run(
            model_id=job.model_id, pipeline_dir=pipeline_dir, request=request,
            device=device, output_path=self.settings.outputs_path / f"{job.id}.png",
            progress_cb=on_progress,
        )
        if not job.auto_upscale:
            job.output_path = generated
            complete_generation_stages(job, include_upscale)
            return
        advance_generation_stage(job, "upscaling", include_upscale)
        job.output_path = await self._run_auto_upscale(job, generated, device)
        complete_generation_stages(job, include_upscale)

    def _resolve_pipeline_dir(self, entry: Any) -> Path:
        models_root = self.settings.models_path.resolve()
        target = (self.settings.models_path / (entry.file_path or "")).resolve()
        if not target.is_relative_to(models_root):
            raise RuntimeError(f"Model path escapes models directory: {entry.file_path!r}")
        if not target.is_dir():
            raise RuntimeError(f"Model folder missing on disk: {entry.file_path!r}")
        return target

    async def _run_auto_upscale(self, job: GenerationJob, generated: Path, device: str) -> Path:
        upscale_job = UpscaleJob(
            source_path=generated,
            original_filename=generated.name,
            model_name=job.upscale_model_name or "",
            scale=job.upscale_scale or UPSCALE_SCALE_RANGE[0],
            output_format="png",
            model_id=job.upscale_model_id,
            device=device,
        )
        upscale = select_upscale_engine(
            upscale_job, self.registry, self.upscale_engine, self.onnx_upscale_engine
        )
        output = await upscale.run(upscale_job)
        generated.unlink(missing_ok=True)
        return output
