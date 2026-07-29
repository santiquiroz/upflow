from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Nombre del vocabulario single-file de diffusers (lo que devuelve
# infer_diffusers_model_type): "xl_base", "v1", "flux-dev", "sd35_medium"...
Architecture = str

# Instalar un checkpoint suelto necesita DOS capacidades, y una arquitectura
# entra solo si tiene las dos:
#
#   1. Que diffusers pueda CARGARLO desde un archivo suelto. Ninguna auto-clase
#      expone from_single_file (ni DiffusionPipeline ni AutoPipelineForText2Image,
#      medido 2026-07-29), asi que hay que nombrar la clase concreta.
#   2. Que optimum-onnx pueda EJECUTAR el ONNX resultante, o sea tener entrada en
#      ORTPipelineForText2Image.ort_pipelines_mapping.
#
# El mapeo no es derivable en runtime: resolverlo por red via el repo base falla
# porque los repos base de SD2.1, SD3.5, FLUX.1 y FLUX.2 estan gated.
#
# Fuera a proposito: xl_refiner e inpainting son img2img/inpaint, no
# text-to-image. flux-2-dev, z-image-turbo y los Qwen cargan en diffusers pero no
# tienen entrada en el mapping de optimum. Y `sana` es el caso inverso: optimum
# SI puede ejecutarlo, pero SanaPipeline no expone from_single_file, asi que no
# se puede instalar desde un checkpoint suelto (si desde un repo diffusers).
_ARCHITECTURE_SUPPORT: dict[Architecture, tuple[str, str]] = {
    #                        (clase diffusers,              model type de optimum)
    "v1": ("StableDiffusionPipeline", "stable-diffusion"),
    "v2": ("StableDiffusionPipeline", "stable-diffusion"),
    "xl_base": ("StableDiffusionXLPipeline", "stable-diffusion-xl"),
    "playground-v2-5": ("StableDiffusionXLPipeline", "stable-diffusion-xl"),
    "sd35_large": ("StableDiffusion3Pipeline", "stable-diffusion-3"),
    "sd35_medium": ("StableDiffusion3Pipeline", "stable-diffusion-3"),
    "flux-dev": ("FluxPipeline", "flux"),
    "flux-schnell": ("FluxPipeline", "flux"),
}

# Un LoRA es composicion sobre un modelo base, no un modelo. Se chequea PRIMERO
# porque algunos usan claves con prefijo `diffusion_model.`, que de otro modo
# satisfaria el rol de backbone (medido en nphSi/Z-Image-Lora).
_LORA_MARKERS = (
    "lora_up",
    "lora_down",
    "lora_A",
    "lora_B",
    ".alpha",
    "lora_unet_",
    "lora_te_",
)

_BACKBONE_PREFIXES = (
    "model.diffusion_model.",
    "double_blocks.",
    "single_blocks.",
    "joint_blocks.",
    "diffusion_model.",
)
_TEXT_ENCODER_PREFIXES = (
    "conditioner.embedders.",
    "cond_stage_model.",
    "text_encoders.",
)
# Solo first_stage_model.: un `decoder.`/`encoder.` en la raiz significa que el
# archivo ES un VAE, no que contenga uno (medido en sdxl_vae.safetensors).
_VAE_PREFIXES = ("first_stage_model.",)


# Copia de DIAGNOSTICO, no de UI. La UI la arma el frontend desde reason_key en
# el idioma activo; esto es para mensajes de error de job y logs, que son texto
# libre (llevan stderr, rutas, nombres de excepcion) y nunca van a estar
# traducidos. Son dos superficies distintas a proposito.
_REASON_DIAGNOSTICS: dict[str, str] = {
    "checkpoint.noTensors": "El archivo no declara tensores.",
    "checkpoint.isLora": "Es un LoRA, no un pipeline completo: necesita un modelo base.",
    "checkpoint.incomplete": "No es un pipeline completo: le faltan componentes ({missing}).",
    "checkpoint.unknownArchitecture": "No se pudo identificar la arquitectura.",
    "checkpoint.unsupportedArchitecture": "Arquitectura {architecture} no soportada.",
    "checkpoint.ready": "Checkpoint {architecture} instalable.",
}


@dataclass(slots=True, frozen=True)
class CheckpointVerdict:
    installable: bool
    architecture: Architecture | None
    ort_model_type: str | None
    # Clave de traduccion + sus datos, no copia: el frontend arma la oracion en
    # el idioma activo (ver el spec de i18n de 2026-07-29).
    reason_key: str
    reason_params: dict[str, str] = field(default_factory=dict)

    def diagnostic(self) -> str:
        """Oracion para logs y errores de job. Ver _REASON_DIAGNOSTICS."""
        template = _REASON_DIAGNOSTICS.get(self.reason_key, self.reason_key)
        # Los params llevan claves de traduccion (`component.vae`): en un log se
        # muestra solo el nombre del rol.
        plain = {
            key: value.replace("component.", "").replace(",", ", ")
            for key, value in self.reason_params.items()
        }
        try:
            return template.format(**plain)
        except KeyError:
            return template


class _ShapeOnly:
    """Sustituto de un tensor para el detector de diffusers, que solo mira
    presencia de clave y .shape -- nunca valores. Permite clasificar leyendo el
    header del safetensors (~360 KB) en vez del archivo entero (GBs)."""

    __slots__ = ("shape",)

    def __init__(self, shape: Any) -> None:
        self.shape = tuple(shape)


def supported_architecture(detected: Architecture | None) -> str | None:
    """Model type de optimum para esa arquitectura, o None si no se puede
    instalar desde un checkpoint suelto."""
    if detected is None:
        return None
    support = _ARCHITECTURE_SUPPORT.get(detected)
    return support[1] if support else None


def pipeline_class_for(detected: Architecture) -> str | None:
    support = _ARCHITECTURE_SUPPORT.get(detected)
    return support[0] if support else None


def validate_architecture_table() -> None:
    """Falla ruidosamente si un upgrade de una dependencia rompe el mapeo. Sin
    esto degradaria en silencio a "arquitectura no soportada"."""
    import diffusers
    from optimum.onnxruntime import ORTPipelineForText2Image

    known_ort = set(ORTPipelineForText2Image.ort_pipelines_mapping)
    unknown_ort = sorted({t for _, t in _ARCHITECTURE_SUPPORT.values()} - known_ort)
    if unknown_ort:
        raise RuntimeError(
            "optimum-onnx ya no soporta estos model types: "
            + ", ".join(unknown_ort)
            + f". Soportados hoy: {sorted(known_ort)}. Revisar _ARCHITECTURE_SUPPORT."
        )

    broken = sorted(
        name
        for name, _ in _ARCHITECTURE_SUPPORT.values()
        if not hasattr(getattr(diffusers, name, None), "from_single_file")
    )
    if broken:
        raise RuntimeError(
            "Estas clases de diffusers ya no exponen from_single_file: "
            + ", ".join(broken)
            + ". Revisar _ARCHITECTURE_SUPPORT."
        )


def _tensor_keys(header: dict[str, Any]) -> list[str]:
    return [
        key
        for key, meta in header.items()
        if key != "__metadata__" and isinstance(meta, dict) and "shape" in meta
    ]


def _has_role(keys: list[str], prefixes: tuple[str, ...]) -> bool:
    return any(key.startswith(prefix) for key in keys for prefix in prefixes)


def _looks_like_lora(keys: list[str]) -> bool:
    return any(marker in key for key in keys for marker in _LORA_MARKERS)


def _missing_roles(keys: list[str]) -> list[str]:
    roles = (
        ("component.backbone", _BACKBONE_PREFIXES),
        ("component.textEncoder", _TEXT_ENCODER_PREFIXES),
        ("component.vae", _VAE_PREFIXES),
    )
    return [name for name, prefixes in roles if not _has_role(keys, prefixes)]


def _detect(keys_with_shapes: dict[str, Any]) -> Architecture | None:
    from diffusers.loaders.single_file_utils import infer_diffusers_model_type

    try:
        return infer_diffusers_model_type(keys_with_shapes)
    except Exception:  # noqa: BLE001 - clasificar es best-effort, nunca propaga
        return None


def classify_checkpoint(header: dict[str, Any]) -> CheckpointVerdict:
    """Decide si un .safetensors suelto es un pipeline instalable.

    El orden NO es cosmetico. infer_diffusers_model_type cae a "v1" cuando no
    reconoce nada -- medido: un VAE suelto, un LoRA de Z-Image y un IP-Adapter
    devuelven todos "v1" con CERO claves consultadas. Es un clasificador que
    asume que ya sabes que el archivo es un checkpoint, no un validador. Por eso
    los gates propios corren antes, y su default nunca decide nada.
    """
    keys = _tensor_keys(header)
    if not keys:
        return CheckpointVerdict(False, None, None, "checkpoint.noTensors")

    if _looks_like_lora(keys):
        return CheckpointVerdict(False, None, None, "checkpoint.isLora")

    missing = _missing_roles(keys)
    if missing:
        return CheckpointVerdict(
            False, None, None, "checkpoint.incomplete",
            {"missing": ",".join(missing)},
        )

    detected = _detect({key: _ShapeOnly(header[key]["shape"]) for key in keys})
    if detected is None:
        return CheckpointVerdict(False, None, None, "checkpoint.unknownArchitecture")

    ort_model_type = supported_architecture(detected)
    if ort_model_type is None:
        return CheckpointVerdict(
            False, detected, None, "checkpoint.unsupportedArchitecture",
            {"architecture": detected},
        )

    return CheckpointVerdict(
        True, detected, ort_model_type, "checkpoint.ready",
        {"architecture": detected},
    )


def materialize(checkpoint_path: Path, out_dir: Path, architecture: Architecture) -> None:
    """Convierte un checkpoint suelto en el arbol diffusers que main_export espera.

    Se resuelve la clase CONCRETA de diffusers: ninguna auto-clase expone
    from_single_file (medido: ni DiffusionPipeline ni AutoPipelineForText2Image).

    Torch pesado y bloqueante: el caller lo corre en asyncio.to_thread, igual
    que el export. Carga el checkpoint entero en RAM, de ahi el aviso de RAM del
    pre-flight.
    """
    import diffusers

    class_name = pipeline_class_for(architecture)
    if class_name is None:
        raise ValueError(
            f"Arquitectura {architecture!r} no se puede cargar desde un checkpoint suelto."
        )
    pipeline_cls = getattr(diffusers, class_name)

    pipeline = pipeline_cls.from_single_file(str(checkpoint_path))
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline.save_pretrained(str(out_dir))
    finally:
        del pipeline
