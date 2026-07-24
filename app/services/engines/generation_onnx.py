from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.services.engines.gmfss_engine import _tune_session_options_for_device
from app.services.engines.onnx_upscaler import _build_providers
from app.services.gpu_session_coordinator import GpuSessionCoordinator

GENERATION_IMPORT_ERROR_HINT = (
    "Las dependencias de generación no están instaladas (paquete optimum). "
    "Reinstalá Upflow o corré la receta de docs/superpowers/specs/2026-07-22-optimum-spike-findings.md."
)

CUDA_ONLY_MESSAGE = (
    "Este modelo requiere GPU NVIDIA (CUDA) y no es compatible con DirectML en tu hardware. "
    "Buscá una versión compatible (ej. colección `amd/` en Hugging Face, formato ONNX+DirectML)."
)

VRAM_MESSAGE = (
    "Sin memoria de GPU para generar con esta configuración. Probá menor resolución, "
    "un modelo más liviano, o esperá a que terminen otros trabajos de GPU."
)

CPU_ONLY_WARNING = (
    "No se detectó GPU compatible (DirectX 12). Generar en CPU tarda varios minutos por imagen. "
    "¿Continuar igual?"
)

_MEMORY_TOKENS = ("memory", "alloc", "oom")
_CUDA_TOKENS = ("cudaexecutionprovider", "cuda", "tensorrt")


class GenerationCancelled(Exception):
    pass


def generation_dependencies_available() -> tuple[bool, str | None]:
    try:
        import optimum.onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de import = no disponible
        return False, f"{GENERATION_IMPORT_ERROR_HINT} ({exc})"
    return True, None


def _load_pipeline_class() -> Any:
    # Nombre confirmado por el spike (Task 1). Si el findings doc dice otro, cambiarlo SOLO acá.
    from optimum.onnxruntime import ORTStableDiffusionPipeline

    return ORTStableDiffusionPipeline


def _wrap_generation_error(exc: Exception) -> RuntimeError:
    message = str(exc)
    lowered = message.lower()
    if any(token in lowered for token in _MEMORY_TOKENS):
        return RuntimeError(f"{VRAM_MESSAGE} ({message})")
    if any(token in lowered for token in _CUDA_TOKENS):
        return RuntimeError(f"{CUDA_ONLY_MESSAGE} ({message})")
    return RuntimeError(f"Image generation failed: {message}")


def _build_providers_for_validation(device: str) -> dict[str, Any]:
    primary = _build_providers(device)[0]
    kwargs: dict[str, Any] = {"use_io_binding": False}
    if isinstance(primary, tuple):
        provider_name, provider_options = primary
        kwargs.update(provider=provider_name, provider_options=provider_options)
    else:
        kwargs["provider"] = primary
    return kwargs


def _build_seed_generator(seed: int) -> Any:
    # torch.Generator, NO np.random.RandomState: __call__ es el de diffusers y
    # randn_tensor accede a generator.device (findings §d/§e, verificado empirico).
    import torch

    return torch.Generator(device="cpu").manual_seed(seed)


@dataclass(slots=True, kw_only=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str | None
    steps: int
    guidance: float
    width: int
    height: int
    seed: int | None


class GenerationEngine:
    def __init__(self, settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._pipeline_cache: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._cache_lock = threading.Lock()

    def release_device(self, device: str) -> None:
        with self._cache_lock:
            stale_keys = [key for key in self._pipeline_cache if key[1] == device]
            for key in stale_keys:
                self._pipeline_cache.pop(key, None)

    async def run(
        self,
        *,
        model_id: str,
        pipeline_dir: Path,
        request: GenerationRequest,
        device: str,
        output_path: Path,
        progress_cb: Callable[[int, int], None],
    ) -> Path:
        cancel_event = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(
                self._run_blocking,
                model_id,
                pipeline_dir,
                request,
                device,
                output_path,
                cancel_event,
                progress_cb,
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancel_event.set()
            with contextlib.suppress(BaseException):
                await worker
            raise

    def _run_blocking(
        self,
        model_id: str,
        pipeline_dir: Path,
        request: GenerationRequest,
        device: str,
        output_path: Path,
        cancel_event: threading.Event,
        progress_cb: Callable[[int, int], None],
    ) -> Path:
        pipeline = self._get_pipeline(model_id, pipeline_dir, device)

        def _on_step(step: int, _timestep: Any, _latents: Any) -> None:
            if cancel_event.is_set():
                raise GenerationCancelled()
            progress_cb(step + 1, request.steps)

        call_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance,
            "width": request.width,
            "height": request.height,
            "callback": _on_step,
            "callback_steps": 1,
        }
        if request.negative_prompt:
            call_kwargs["negative_prompt"] = request.negative_prompt
        if request.seed is not None:
            call_kwargs["generator"] = _build_seed_generator(request.seed)

        try:
            result = pipeline(**call_kwargs)
        except GenerationCancelled:
            raise
        except Exception as exc:
            raise _wrap_generation_error(exc) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output_path)
        return output_path

    def _get_pipeline(self, model_id: str, pipeline_dir: Path, device: str) -> Any:
        self.gpu_coordinator.acquire(device, self)
        key = (model_id, device)
        with self._cache_lock:
            cached = self._pipeline_cache.get(key)
            if cached is not None:
                self._pipeline_cache.move_to_end(key)
                return cached
        try:
            pipeline = self._create_pipeline(pipeline_dir, device)
        except Exception as exc:
            raise _wrap_generation_error(exc) from exc
        with self._cache_lock:
            self._pipeline_cache[key] = pipeline
            self._pipeline_cache.move_to_end(key)
            while len(self._pipeline_cache) > 1:
                self._pipeline_cache.popitem(last=False)
        return pipeline

    def _create_pipeline(self, pipeline_dir: Path, device: str) -> Any:
        import onnxruntime as ort

        pipeline_cls = _load_pipeline_class()
        providers = _build_providers(device)
        sess_options = ort.SessionOptions()
        _tune_session_options_for_device(sess_options, device)
        primary = providers[0]
        # use_io_binding=False explicito: hoy es el default para DML pero se
        # blinda ante cambios de default de optimum (IOBinding+DML vetado en este repo).
        from_pretrained_kwargs: dict[str, Any] = {
            "session_options": sess_options,
            "use_io_binding": False,
        }
        if isinstance(primary, tuple):
            provider_name, provider_options = primary
            from_pretrained_kwargs["provider"] = provider_name
            from_pretrained_kwargs["provider_options"] = provider_options
        else:
            from_pretrained_kwargs["provider"] = primary
        return pipeline_cls.from_pretrained(str(pipeline_dir), **from_pretrained_kwargs)
