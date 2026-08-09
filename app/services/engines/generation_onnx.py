from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import Settings
from app.services.dml_device import try_parse_dml_device_id
from app.services.generation_pipeline_modes import load_img2img_class
from app.services.engines.gmfss_engine import _tune_session_options_for_device
from app.services.engines.onnx_common import build_providers
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services import ep_registry

logger = logging.getLogger(__name__)

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
    "StableDiffusionPipeline": "ORTStableDiffusionPipeline",
    "StableDiffusionXLPipeline": "ORTStableDiffusionXLPipeline",
    "StableDiffusion3Pipeline": "ORTStableDiffusion3Pipeline",
    # Latent Consistency: genera en 4-8 pasos en vez de 25-30. optimum lo
    # ejecuta (esta en su mapping y expone la clase), y sin esta entrada un repo
    # LCM se descargaba entero para morir al cargarlo.
    "LatentConsistencyModelPipeline": "ORTLatentConsistencyModelPipeline",
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
    if request.mask_image_path is not None:
        return "inpaint"
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
    primary = build_providers(device)[0]
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


def _load_mask_image(path: Path, width: int, height: int) -> Any:
    """Máscara en escala de grises (blanco=editar, negro=conservar).

    NEAREST y no LANCZOS: un resampling suave crea grises intermedios en el
    borde que el compositing final interpretaría a medias.
    """
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("L").resize((width, height), Image.NEAREST)


def _composite_outside_mask(generated: Any, original: Any, mask: Any) -> Any:
    """Pega los píxeles originales fuera de la máscara, a nivel de píxel.

    El pipeline de inpainting pasa TODA la imagen por el roundtrip del VAE, así
    que fuera de la máscara los píxeles quedan parecidos pero no idénticos.
    El contrato de la feature es "negro=conservar" literal: se compone acá.
    """
    import numpy as np
    from PIL import Image

    gen = np.asarray(generated.convert("RGB"))
    orig = np.asarray(original.convert("RGB"))
    keep_original = np.asarray(mask)[:, :, None] <= 127
    return Image.fromarray(np.where(keep_original, orig, gen))


def _inpaint_strength(pipeline: Any, requested: float) -> float:
    """Checkpoint normal (unet de 4 canales): strength DEBE ser 1.0.

    Medido 2026-07-31 en SDXL/DirectML/diffusers 0.39: con unet 4ch el camino de
    blending latente del pipeline de inpainting devuelve la imagen casi intacta
    para CUALQUIER strength < 1.0 (0.85/0.90/0.95/0.99 -> diff media ~2/255 = solo
    roundtrip VAE; 1.0 -> regeneracion real). add_noise no es el culpable: img2img
    con 0.6/0.85 funciona en el mismo stack. Un checkpoint de inpainting real
    (unet 9ch) si soporta strength parcial y se respeta lo pedido.
    """
    in_channels = getattr(getattr(getattr(pipeline, "unet", None), "config", None), "in_channels", None)
    if in_channels == 4 and requested < 1.0:
        logger.info("inpaint: unet 4ch, strength %.2f -> 1.0 (strength parcial es no-op medido)", requested)
        return 1.0
    return requested


_PIPELINE_SESSION_ATTRS = ("unet", "vae_decoder", "vae_encoder", "text_encoder", "text_encoder_2")

_ORIGINAL_SCHEDULER_ATTR = "_upflow_original_scheduler"

# Umbrales del cache de pipelines. Regla: retener una entrada extra exige que
# DESPUÉS de retenerla quede lugar para UN load más de peor caso (SDXL fp16
# ~5GB) + margen — si no, el tercer modo del mismo modelo hace OOM contra un
# cache lleno (confirmado en review adversarial: 16GB, text2img+inpaint
# retenidos = 10GB, img2img sin lugar). Modelos chicos solo se benefician.
_CACHE_COMFORT_FREE_MB = 11000
_CACHE_TIGHT_FREE_MB = 6000


def _pipeline_cache_capacity(device: str) -> int:
    """Cuántos pipelines retener para este device según su VRAM libre AHORA.

    1 era el default seguro, pero alternar text2img↔inpaint del mismo modelo
    recargaba GBs en cada cambio. Sin lectura de VRAM (None) se queda en 1.
    """
    index = try_parse_dml_device_id(device)
    if index is None:
        return 1
    from app.services.devices_service import adapter_free_vram_mb

    free_mb = adapter_free_vram_mb(index)
    if free_mb is None:
        return 1
    if free_mb >= _CACHE_COMFORT_FREE_MB:
        return 3
    if free_mb >= _CACHE_TIGHT_FREE_MB:
        return 2
    return 1


def _build_scheduler(name: str, config: Any) -> Any:
    from diffusers.schedulers import (
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        LCMScheduler,
    )

    if name == "lcm":
        return LCMScheduler.from_config(config)
    if name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(config)
    if name == "euler_trailing":
        return EulerDiscreteScheduler.from_config(config, timestep_spacing="trailing")
    raise RuntimeError(f"Unsupported scheduler: {name!r}")


def _apply_scheduler(pipeline: Any, name: str | None) -> None:
    """Aplica el scheduler pedido al pipeline cacheado, SIEMPRE bajo su run_lock.

    El pipeline es compartido: sin restaurar, el scheduler de un job quedaría
    pegado para el siguiente. `None` restaura el original del repo.
    """
    original = getattr(pipeline, _ORIGINAL_SCHEDULER_ATTR, None)
    if original is None:
        original = getattr(pipeline, "scheduler", None)
        if original is None:
            # Dobles de test o pipelines sin scheduler: nada que intercambiar.
            return
        setattr(pipeline, _ORIGINAL_SCHEDULER_ATTR, original)
    if name is None:
        pipeline.scheduler = original
        return
    pipeline.scheduler = _build_scheduler(name, original.config)


def _record_pipeline_providers(pipeline: Any, device: str) -> None:
    # optimum no expone los providers del pipeline: hay que preguntarle a la
    # sesión de cada componente, que es donde ORT pudo caer a CPU en silencio.
    for name in _PIPELINE_SESSION_ATTRS:
        session = getattr(getattr(pipeline, name, None), "session", None)
        if session is not None:
            ep_registry.record_session_providers(device, session, context=f"generación/{name}")


def _looks_like_memory_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return any(token in lowered for token in _MEMORY_TOKENS)


def _native_side_for(pipeline: Any) -> int:
    # SDXL y SD3 entrenan a 1024; el resto de las familias soportadas, a 512.
    name = type(pipeline).__name__
    return 1024 if ("XL" in name or "Diffusion3" in name) else 512


def _pipeline_precision(pipeline: Any) -> str | None:
    """fp16/fp32 REAL del UNet cargado, leído del tipo de salida de su sesión.

    Un modelo que cayó a fp32 corre ~7x más lento (medido en este repo) y sin
    esta lectura la regresión es silenciosa.
    """
    session = getattr(getattr(pipeline, "unet", None), "session", None)
    if session is None:
        return None
    try:
        outputs = session.get_outputs()
    except Exception:  # noqa: BLE001 -- diagnóstico jamás tumba un job
        return None
    if not outputs:
        return None
    output_type = getattr(outputs[0], "type", "")
    if "float16" in output_type:
        return "fp16"
    if output_type == "tensor(float)":
        return "fp32"
    return None


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
    # Máscara de inpainting (blanco=editar, negro=conservar). Requiere
    # init_image_path; presente = modo inpaint.
    mask_image_path: Path | None = None
    # Preparación de la máscara (ver inpaint_mask/inpaint_pipeline): agrandar lo
    # marcado se come el halo del contorno, la transición evita la costura, y el
    # contexto es lo que le permite al modelo continuar el fondo.
    mask_dilate_px: int = 8
    mask_feather_px: int = 8
    # `None` = se calcula a partir del tamaño de lo marcado. Era 48 fijo, y eso
    # dejaba al modelo sin ver la cara cuando la marca era grande.
    mask_padding_px: int | None = None
    # Scheduler alternativo para este job ("lcm" | "euler_a" | "euler_trailing").
    # None = el que declara el repo del modelo. El swap es CPU-side (componente
    # diffusers): no toca el grafo ONNX y vale para cualquier EP.
    scheduler: str | None = None


class GenerationEngine:
    def __init__(self, settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._pipeline_cache: OrderedDict[tuple[str, str, str], Any] = OrderedDict()
        self._cache_lock = threading.Lock()
        # Un lock por clave de pipeline: el scheduler del pipeline es mutable y
        # una segunda __call__ solapada lo pisa a mitad de loop (set_timesteps
        # ajeno -> "index N is out of bounds..."). Nunca se evictan: quedan
        # acotados por los modelos usados en la sesión y así una entrada
        # re-creada con la misma clave sigue serializada contra un job en vuelo.
        self._pipeline_run_locks: dict[tuple[str, str, str], threading.Lock] = {}

    def release_device(self, device: str) -> None:
        with self._cache_lock:
            stale_keys = [key for key in self._pipeline_cache if key[1] == device]
            for key in stale_keys:
                self._pipeline_cache.pop(key, None)

    def _pipeline_run_lock(self, key: tuple[str, str, str]) -> threading.Lock:
        with self._cache_lock:
            lock = self._pipeline_run_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._pipeline_run_locks[key] = lock
            return lock

    def precision_for(self, model_id: str, device: str, mode: str) -> str | None:
        with self._cache_lock:
            pipeline = self._pipeline_cache.get((model_id, device, mode))
        return _pipeline_precision(pipeline) if pipeline is not None else None

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
        run_lock = self._pipeline_run_lock((model_id, device, mode))

        def _on_step(step: int, _timestep: Any, _latents: Any) -> None:
            if cancel_event.is_set():
                raise GenerationCancelled()
            progress_cb(step + 1, request.steps)

        if mode == "inpaint":
            return self._run_masked_edit_blocking(
                pipeline, request, output_path, _on_step, run_lock
            )

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
            with run_lock:
                _apply_scheduler(pipeline, request.scheduler)
                result = pipeline(**call_kwargs)
        except GenerationCancelled:
            raise
        except Exception as exc:
            raise _wrap_generation_error(exc) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output_path)
        return output_path

    def _run_masked_edit_blocking(
        self,
        pipeline: Any,
        request: GenerationRequest,
        output_path: Path,
        on_step: Callable[[int, Any, Any], None],
        run_lock: threading.Lock,
    ) -> Path:
        """Edición con máscara por el camino 'solo el área marcada'.

        La foto sale con su resolución original: el modelo solo ve el recorte
        de la zona marcada, así que ni la reduce ni la vuelve a comprimir.
        """
        from PIL import Image

        from app.services.inpaint_pipeline import MaskedEditSettings, run_masked_edit

        with Image.open(request.init_image_path) as source:
            base_image = source.convert("RGB")
        with Image.open(request.mask_image_path) as source_mask:
            mask_image = source_mask.convert("L")

        strength = _inpaint_strength(pipeline, request.strength)

        def run_model(image: Any, mask: Any, width: int, height: int) -> Any:
            from app.services.generation_soft_inpaint import (
                build_soft_inpaint_callback,
                chain_step_callbacks,
            )

            soft_callback = None
            try:
                soft_callback = build_soft_inpaint_callback(pipeline, image, mask, request.seed)
            except Exception:  # noqa: BLE001 -- calidad extra jamás tumba el job
                logger.warning(
                    "soft inpainting no disponible para este pipeline, sigo sin él",
                    exc_info=True,
                )
            call_kwargs: dict[str, Any] = {
                "prompt": request.prompt,
                "image": image,
                "mask_image": mask,
                "num_inference_steps": request.steps,
                "guidance_scale": request.guidance,
                "strength": strength,
                "width": width,
                "height": height,
                "callback": chain_step_callbacks(soft_callback, on_step),
                "callback_steps": 1,
            }
            if request.negative_prompt:
                call_kwargs["negative_prompt"] = request.negative_prompt
            if request.seed is not None:
                call_kwargs["generator"] = _build_seed_generator(request.seed)
            try:
                with run_lock:
                    _apply_scheduler(pipeline, request.scheduler)
                    return pipeline(**call_kwargs).images[0]
            except GenerationCancelled:
                raise
            except Exception as exc:
                raise _wrap_generation_error(exc) from exc

        settings = MaskedEditSettings(
            dilate_px=request.mask_dilate_px,
            feather_px=request.mask_feather_px,
            padding_px=request.mask_padding_px,
            target_side=max(request.width, request.height),
            # Piso de resolución: un parche chico difundido a su propio tamaño
            # produce anatomía blanda — el modelo rinde a su resolución nativa.
            native_side=_native_side_for(pipeline),
        )
        edited = run_masked_edit(base_image, mask_image, run_model, settings)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        edited.save(output_path)
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
            if not _looks_like_memory_error(exc):
                raise _wrap_generation_error(exc) from exc
            # OOM con cache retenido: evictar lo de este device y reintentar UNA
            # vez. Sin esto, el retry del usuario pega contra el mismo muro
            # indefinidamente (el cache lleno nunca se vacía solo).
            self.release_device(device)
            try:
                pipeline = self._create_pipeline(pipeline_dir, device, mode)
            except Exception as retry_exc:
                raise _wrap_generation_error(retry_exc) from retry_exc
        # La capacidad se lee DESPUÉS del load: la VRAM libre ya descuenta el
        # pipeline recién cargado, así que el margen medido es real.
        capacity = _pipeline_cache_capacity(device)
        with self._cache_lock:
            self._pipeline_cache[key] = pipeline
            self._pipeline_cache.move_to_end(key)
            while len(self._pipeline_cache) > capacity:
                self._pipeline_cache.popitem(last=False)
        return pipeline

    def _create_pipeline(
        self, pipeline_dir: Path, device: str, mode: str = "text2img"
    ) -> Any:
        import onnxruntime as ort

        declared = _read_declared_class_name(pipeline_dir)
        from app.services.generation_pipeline_modes import (
            is_dedicated_inpaint_class,
            load_inpaint_class,
        )

        if mode != "inpaint" and is_dedicated_inpaint_class(declared):
            # Un unet de 9 canales EXIGE imagen+máscara como entrada: pedirle
            # txt2img/img2img fallaría en el grafo con un error críptico.
            raise RuntimeError(
                "Este modelo es de inpainting dedicado: usalo desde el Editor "
                "con una máscara. No puede generar desde texto ni desde imagen sola."
            )
        if mode == "inpaint":
            pipeline_cls = load_inpaint_class(declared)
        elif mode == "img2img":
            pipeline_cls = load_img2img_class(declared)
        else:
            pipeline_cls = _load_pipeline_class(declared)
        def build_sess_options() -> Any:
            sess_options = ort.SessionOptions()
            _tune_session_options_for_device(sess_options, device)
            return sess_options

        # Los providers salen del ep_registry y no armados a mano: si no, con un
        # plugin EP instalado aceleraba todo menos generar, sin ninguna señal.
        # optimum reusa este SessionOptions para las varias sesiones del pipeline
        # (unet, vae, encoders), lo cual está verificado contra onnxruntime real.
        # use_io_binding=False explicito: hoy es el default para DML pero se
        # blinda ante cambios de default de optimum (IOBinding+DML vetado en este repo).
        from_pretrained_kwargs: dict[str, Any] = {
            "use_io_binding": False,
            **ep_registry.loader_kwargs(
                device, self.settings, sess_options_factory=build_sess_options
            ),
        }
        pipeline = pipeline_cls.from_pretrained(str(pipeline_dir), **from_pretrained_kwargs)
        _record_pipeline_providers(pipeline, device)
        return pipeline
