from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.services.generation_img2img import load_img2img_class
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

# _class_name declarados verificados contra repos reales en
# docs/superpowers/specs/2026-07-25-third-party-spike-findings.md. El repo
# amd/ de SDXL ya declara ORTStableDiffusionXLPipeline; esos nombres ORT pasan
# directo. SDXL Turbo usa la misma clase SDXL (con menos steps y otro
# scheduler), no una clase aparte.
PIPELINE_CLASS_NAMES: dict[str, str] = {
    "OnnxStableDiffusionPipeline": "ORTStableDiffusionPipeline",
    "StableDiffusionXLPipeline": "ORTStableDiffusionXLPipeline",
    "StableDiffusion3Pipeline": "ORTStableDiffusion3Pipeline",
}
_KNOWN_ORT_CLASS_NAMES = frozenset(PIPELINE_CLASS_NAMES.values())

# Duplicado deliberado de generation_installer.MODEL_INDEX_FILENAME: ese módulo
# ya importa de este; importarlo acá sería un import circular.
_MODEL_INDEX_FILENAME = "model_index.json"


class GenerationCancelled(Exception):
    pass


def generation_dependencies_available() -> tuple[bool, str | None]:
    try:
        import optimum.onnxruntime  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - cualquier fallo de import = no disponible
        return False, f"{GENERATION_IMPORT_ERROR_HINT} ({exc})"
    return True, None


def _read_declared_class_name(pipeline_dir: Path) -> str:
    index_path = pipeline_dir / _MODEL_INDEX_FILENAME
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"No se pudo leer {_MODEL_INDEX_FILENAME} del pipeline en {pipeline_dir.name}: {exc}"
        ) from exc
    declared = index.get("_class_name")
    if not isinstance(declared, str) or not declared:
        raise RuntimeError(
            f"El {_MODEL_INDEX_FILENAME} del pipeline no declara _class_name -- "
            "no se puede elegir la clase de carga."
        )
    return declared


def _pipeline_mode(request: "GenerationRequest") -> str:
    return "img2img" if request.init_image_path is not None else "text2img"


def _load_pipeline_class(declared_class_name: str) -> Any:
    if declared_class_name in _KNOWN_ORT_CLASS_NAMES:
        ort_class_name = declared_class_name
    else:
        ort_class_name = PIPELINE_CLASS_NAMES.get(declared_class_name)
    if ort_class_name is None:
        supported = ", ".join(
            sorted(set(PIPELINE_CLASS_NAMES) | _KNOWN_ORT_CLASS_NAMES)
        )
        raise RuntimeError(
            f"Clase de pipeline no soportada: {declared_class_name!r}. "
            f"Clases soportadas: {supported}."
        )
    import optimum.onnxruntime as ort_module

    return getattr(ort_module, ort_class_name)


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


def _load_init_image(path: Path, width: int, height: int) -> Any:
    """Carga la imagen de partida y la lleva al tamaño pedido.

    Se convierte a RGB porque un PNG con canal alfa rompe el VAE, y se
    redimensiona aca en vez de dejarlo al pipeline para que el tamaño de salida
    sea el que el usuario pidio y no el del archivo que subio.
    """
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").resize((width, height), Image.LANCZOS)


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
    # Imagen de partida. Presente = imagen a imagen; ausente = texto a imagen.
    init_image_path: Path | None = None
    # Cuanto se aparta del original: 0 lo devuelve casi igual, 1 lo ignora casi
    # por completo. Solo se usa con init_image_path.
    strength: float = 0.6


class GenerationEngine:
    def __init__(self, settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._pipeline_cache: OrderedDict[tuple[str, str, str], Any] = OrderedDict()
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
        mode = _pipeline_mode(request)
        pipeline = self._get_pipeline(model_id, pipeline_dir, device, mode)

        def _on_step(step: int, _timestep: Any, _latents: Any) -> None:
            if cancel_event.is_set():
                raise GenerationCancelled()
            progress_cb(step + 1, request.steps)

        call_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance,
            "callback": _on_step,
            "callback_steps": 1,
        }
        if request.init_image_path is not None:
            # width/height NO se pasan: el pipeline de imagen a imagen deriva el
            # tamaño de la imagen de entrada, asi que se la redimensiona antes.
            call_kwargs["image"] = _load_init_image(
                request.init_image_path, request.width, request.height
            )
            call_kwargs["strength"] = request.strength
        else:
            call_kwargs["width"] = request.width
            call_kwargs["height"] = request.height
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

    def _get_pipeline(
        self, model_id: str, pipeline_dir: Path, device: str, mode: str = "text2img"
    ) -> Any:
        self.gpu_coordinator.acquire(device, self)
        # El modo entra en la clave: el mismo modelo en el mismo dispositivo carga
        # una clase distinta para texto a imagen que para imagen a imagen, y sin
        # esto un job de imagen a imagen recibiria el pipeline de texto cacheado y
        # la imagen de entrada se ignoraria en silencio.
        key = (model_id, device, mode)
        with self._cache_lock:
            cached = self._pipeline_cache.get(key)
            if cached is not None:
                self._pipeline_cache.move_to_end(key)
                return cached
        try:
            pipeline = self._create_pipeline(pipeline_dir, device, mode)
        except Exception as exc:
            raise _wrap_generation_error(exc) from exc
        with self._cache_lock:
            self._pipeline_cache[key] = pipeline
            self._pipeline_cache.move_to_end(key)
            while len(self._pipeline_cache) > 1:
                self._pipeline_cache.popitem(last=False)
        return pipeline

    def _create_pipeline(
        self, pipeline_dir: Path, device: str, mode: str = "text2img"
    ) -> Any:
        import onnxruntime as ort

        declared = _read_declared_class_name(pipeline_dir)
        pipeline_cls = (
            load_img2img_class(declared)
            if mode == "img2img"
            else _load_pipeline_class(declared)
        )
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
