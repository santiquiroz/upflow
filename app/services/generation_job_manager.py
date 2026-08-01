from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.exceptions import QueueFullError
from app.models import TERMINAL_JOB_STATUSES, GenerationJob, JobStatus, UpscaleJob, utc_now
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.generation_onnx import GenerationEngine, GenerationRequest
from app.services.engines.migan_eraser import ERASER_MODEL_ID
from app.services.engines.sdcpp_engine import SDCPP_MODEL_ID
from app.services.engines.sdcpp_models import SDCPP_MODEL_PREFIX, resolve_sdcpp_model
from app.services.generation_img2img import supports_img2img
from app.services.generation_inpaint import supports_inpaint
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
        quota_service: QuotaService | None = None,
        sdcpp_engine: Any | None = None,
        migan_eraser: Any | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.device_semaphores = device_semaphores
        self.registry = registry
        self.upscale_engine = upscale_engine
        self.onnx_upscale_engine = onnx_upscale_engine
        self.devices = devices
        self.quota_service = quota_service
        self.sdcpp_engine = sdcpp_engine
        self.migan_eraser = migan_eraser
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
        init_image_path: Path | None = None,
        strength: float = 0.6,
        mask_image_path: Path | None = None,
        auto_upscale: bool = False,
        upscale_model_name: str | None = None,
        upscale_scale: int | None = None,
        upscale_model_id: str | None = None,
        job_id: str | None = None,
        owner: AuthenticatedUser | None = None,
    ) -> GenerationJob:
        self._validate_generation_model(model_id)
        self._validate_params(prompt, steps, width, height)
        await self._validate_device(device)
        if mask_image_path is not None:
            self._validate_inpaint(model_id, init_image_path, mask_image_path, strength)
        elif init_image_path is not None:
            self._validate_img2img(model_id, init_image_path, strength)
        if auto_upscale:
            self._validate_upscale_params(upscale_model_name, upscale_scale, upscale_model_id)
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)
        job = GenerationJob(
            prompt=prompt, model_id=model_id, negative_prompt=negative_prompt, steps=steps,
            guidance=guidance, width=width, height=height, seed=seed, device=device,
            init_image_path=init_image_path, strength=strength,
            mask_image_path=mask_image_path,
            auto_upscale=auto_upscale, upscale_model_name=upscale_model_name,
            upscale_scale=upscale_scale, upscale_model_id=upscale_model_id,
            owner_id=owner.id if owner is not None else None,
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
        if model_id == ERASER_MODEL_ID:
            # Motor de borrado: no vive en el registro de modelos, se gatea por
            # su archivo en disco igual que el lane experimental.
            if self.migan_eraser is None or not self.migan_eraser.available():
                raise ValueError(
                    "El borrado rápido no está instalado: instalalo desde Ajustes "
                    "o corré scripts/download-migan.ps1"
                )
            return
        if model_id.startswith(SDCPP_MODEL_PREFIX):
            # Lane Vulkan: el modelo es un archivo en disco, no una entrada del
            # registro. Se valida que exista de verdad y no por el nombre.
            if self.sdcpp_engine is None or resolve_sdcpp_model(model_id, self.settings) is None:
                raise ValueError(
                    "Ese modelo Vulkan ya no está disponible. Instalá uno desde Models."
                )
            return
        if model_id == SDCPP_MODEL_ID:
            # Lane experimental Fase 3: no vive en el registry, se gatea por flag.
            if self.sdcpp_engine is None or not self.settings.sdcpp_available():
                raise ValueError(
                    "El lane sd.cpp es experimental y está apagado: ENABLE_SDCPP=true "
                    "+ scripts/download-sdcpp.ps1 + SDCPP_MODEL con tu checkpoint"
                )
            return
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

    def _validate_img2img(
        self, model_id: str, init_image_path: Path, strength: float
    ) -> None:
        """Rechaza al CREAR el job y no a mitad de la ejecucion.

        Un modelo Flux corre perfecto para texto a imagen y no tiene camino de
        imagen a imagen en optimum. Descubrirlo despues de encolar significaria un
        job que falla varios segundos mas tarde con un error de carga de clase.
        """
        if not 0 < strength <= 1:
            raise ValueError("strength must be greater than 0 and at most 1")
        if not init_image_path.exists():
            raise ValueError(f"init image not found: {init_image_path.name}")

        entry = self.registry.get(model_id)
        declared = self._declared_pipeline_class(entry)
        if declared is None:
            # No se pudo leer el model_index.json: no se bloquea por eso. El motor
            # va a fallar con su propio mensaje si de verdad no sirve, y rechazar
            # aca convertiria una lectura fallida en un veredicto.
            return
        if not supports_img2img(declared):
            raise ValueError(
                f"El modelo {model_id!r} ({declared}) no soporta imagen a imagen. "
                "Sirve para texto a imagen; para imagen a imagen usa un modelo "
                "Stable Diffusion, SDXL, SD3 o Latent Consistency."
            )

    def _validate_inpaint(
        self, model_id: str, init_image_path: Path | None, mask_image_path: Path, strength: float
    ) -> None:
        if model_id.startswith(SDCPP_MODEL_PREFIX):
            raise ValueError("Los modelos Vulkan son solo de texto a imagen por ahora")
        if model_id == ERASER_MODEL_ID:
            if init_image_path is None or not init_image_path.exists():
                raise ValueError("el borrado rápido necesita la imagen base")
            if not mask_image_path.exists():
                raise ValueError(f"mask image not found: {mask_image_path.name}")
            return
        """Mismo principio que _validate_img2img: rechazar al crear, no a mitad
        de la ejecución; una lectura fallida del model_index no es un veredicto."""
        if model_id.startswith(SDCPP_MODEL_PREFIX):
            # Lane Vulkan: el modelo es un archivo en disco, no una entrada del
            # registro. Se valida que exista de verdad y no por el nombre.
            if self.sdcpp_engine is None or resolve_sdcpp_model(model_id, self.settings) is None:
                raise ValueError(
                    "Ese modelo Vulkan ya no está disponible. Instalá uno desde Models."
                )
            return
        if model_id == SDCPP_MODEL_ID:
            raise ValueError("El lane experimental sd.cpp es solo texto a imagen")
        if init_image_path is None:
            raise ValueError("inpainting requiere una imagen base además de la máscara")
        if not 0 < strength <= 1:
            raise ValueError("strength must be greater than 0 and at most 1")
        if not init_image_path.exists():
            raise ValueError(f"init image not found: {init_image_path.name}")
        if not mask_image_path.exists():
            raise ValueError(f"mask image not found: {mask_image_path.name}")
        entry = self.registry.get(model_id)
        declared = self._declared_pipeline_class(entry)
        if declared is None:
            return
        if not supports_inpaint(declared):
            raise ValueError(
                f"El modelo {model_id!r} ({declared}) no soporta inpainting. "
                "Usa un modelo Stable Diffusion, SDXL o SD3."
            )

    def _declared_pipeline_class(self, entry: Any) -> str | None:
        from app.services.engines.generation_onnx import _read_declared_class_name

        try:
            return _read_declared_class_name(self._resolve_pipeline_dir(entry))
        except Exception:  # noqa: BLE001 - una lectura fallida no es un veredicto
            return None

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
            self._record_quota_usage(job)

    def _record_quota_usage(self, job: GenerationJob) -> None:
        if self.quota_service is None or job.started_at is None:
            return
        duration = (job.finished_at - job.started_at).total_seconds()
        self.quota_service.record_usage(job.owner_id, duration)

    async def _run_engine(self, job: GenerationJob) -> None:
        if job.model_id.startswith(SDCPP_MODEL_PREFIX):
            await self._run_sdcpp(job, resolve_sdcpp_model(job.model_id, self.settings))
            return
        if job.model_id == ERASER_MODEL_ID:
            await self._run_eraser(job)
            return
        if job.model_id == SDCPP_MODEL_ID:
            await self._run_sdcpp(job)
            return
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
            init_image_path=job.init_image_path, strength=job.strength,
            mask_image_path=job.mask_image_path,
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
        try:
            job.output_path = await self._run_auto_upscale(job, generated, device)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # Upscale failed after generation already succeeded: fall back to
            # the generated image rather than orphaning it as a failed job --
            # the un-upscaled result is still a valid, usable output.
            job.metadata["upscaleError"] = str(exc)
            job.output_path = generated
        complete_generation_stages(job, include_upscale)

    async def _run_eraser(self, job: GenerationJob) -> None:
        # Borrado sin difusión: un solo paso, así que el progreso va de una.
        from PIL import Image

        advance_generation_stage(job, "generating", False)
        with Image.open(job.init_image_path) as source:
            base_image = source.convert("RGB")
        with Image.open(job.mask_image_path) as source_mask:
            mask_image = source_mask.convert("L")
        device = job.device or self.settings.default_device
        result = await self.migan_eraser.erase(base_image, mask_image, device)
        output_path = self.settings.outputs_path / f"{job.id}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        job.output_path = output_path
        complete_generation_stages(job, False)

    async def _run_sdcpp(self, job: GenerationJob, checkpoint: Path | None = None) -> None:
        # Lane experimental: subprocess sd.cpp Vulkan, sin progreso por paso ni
        # auto-upscale en v1. La cancelación mata el proceso (run_guarded_process).
        advance_generation_stage(job, "generating", False)
        request = GenerationRequest(
            prompt=job.prompt, negative_prompt=job.negative_prompt, steps=job.steps,
            guidance=job.guidance, width=job.width, height=job.height, seed=job.seed,
        )
        job.output_path = await self.sdcpp_engine.run(
            request, self.settings.outputs_path / f"{job.id}.png", checkpoint=checkpoint
        )
        complete_generation_stages(job, False)

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
            owner_id=job.owner_id,
        )
        upscale = select_upscale_engine(
            upscale_job, self.registry, self.upscale_engine, self.onnx_upscale_engine
        )
        output = await upscale.run(upscale_job)
        generated.unlink(missing_ok=True)
        return output
