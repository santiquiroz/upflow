from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Nombre del vocabulario single-file de diffusers (lo que devuelve
# infer_diffusers_model_type): "xl_base", "v1", "flux-dev", "sd35_medium"...
Architecture = str

# Mapeo entre los DOS vocabularios: el de diffusers (claves) y el de optimum
# (valores, los de ORTPipelineForText2Image.ort_pipelines_mapping). No es
# derivable en runtime: resolverlo por red via el repo base falla porque los
# repos base de SD2.1, SD3.5, FLUX.1 y FLUX.2 estan gated (medido 2026-07-29).
#
# Fuera a proposito: xl_refiner y las variantes inpainting son img2img/inpaint,
# no text-to-image; flux-2-dev, z-image-turbo y los Qwen no tienen entrada en el
# mapping de optimum aunque diffusers si los cargue.
_ARCHITECTURE_TO_ORT: dict[Architecture, str] = {
    "v1": "stable-diffusion",
    "v2": "stable-diffusion",
    "xl_base": "stable-diffusion-xl",
    "playground-v2-5": "stable-diffusion-xl",
    "sd35_large": "stable-diffusion-3",
    "sd35_medium": "stable-diffusion-3",
    "flux-dev": "flux",
    "flux-schnell": "flux",
    "sana": "sana",
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


@dataclass(slots=True, frozen=True)
class CheckpointVerdict:
    installable: bool
    architecture: Architecture | None
    ort_model_type: str | None
    reason: str


class _ShapeOnly:
    """Sustituto de un tensor para el detector de diffusers, que solo mira
    presencia de clave y .shape -- nunca valores. Permite clasificar leyendo el
    header del safetensors (~360 KB) en vez del archivo entero (GBs)."""

    __slots__ = ("shape",)

    def __init__(self, shape: Any) -> None:
        self.shape = tuple(shape)


def supported_architecture(detected: Architecture | None) -> str | None:
    if detected is None:
        return None
    return _ARCHITECTURE_TO_ORT.get(detected)


def validate_architecture_table() -> None:
    """Falla ruidosamente si un upgrade de optimum renombra una clave del
    mapping. Sin esto, el mapeo degradaria en silencio a "no soportado"."""
    from optimum.onnxruntime import ORTPipelineForText2Image

    known = set(ORTPipelineForText2Image.ort_pipelines_mapping)
    unknown = sorted(set(_ARCHITECTURE_TO_ORT.values()) - known)
    if unknown:
        raise RuntimeError(
            "optimum-onnx ya no soporta estos model types: "
            + ", ".join(unknown)
            + f". Soportados hoy: {sorted(known)}. Revisar _ARCHITECTURE_TO_ORT."
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
        ("backbone (unet/transformer)", _BACKBONE_PREFIXES),
        ("text encoder", _TEXT_ENCODER_PREFIXES),
        ("VAE", _VAE_PREFIXES),
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
        return CheckpointVerdict(
            False, None, None, "El header del archivo no declara ningun tensor."
        )

    if _looks_like_lora(keys):
        return CheckpointVerdict(
            False,
            None,
            None,
            "Es un LoRA o un adapter, no un modelo completo: se aplica sobre un "
            "modelo base en vez de instalarse solo.",
        )

    missing = _missing_roles(keys)
    if missing:
        return CheckpointVerdict(
            False,
            None,
            None,
            "No es un pipeline completo: le falta " + ", ".join(missing) + ".",
        )

    detected = _detect({key: _ShapeOnly(header[key]["shape"]) for key in keys})
    if detected is None:
        return CheckpointVerdict(
            False, None, None, "No se pudo determinar la arquitectura del checkpoint."
        )

    ort_model_type = supported_architecture(detected)
    if ort_model_type is None:
        return CheckpointVerdict(
            False,
            detected,
            None,
            f"Arquitectura {detected!r}: optimum-onnx no puede ejecutarla, "
            "asi que convertirla no serviria de nada.",
        )

    return CheckpointVerdict(
        True, detected, ort_model_type, f"Checkpoint {detected} listo para convertir."
    )


def materialize(checkpoint_path: Path, out_dir: Path, architecture: Architecture) -> None:
    """Convierte un checkpoint suelto en el arbol diffusers que main_export espera.

    Torch pesado y bloqueante: el caller lo corre en asyncio.to_thread, igual
    que el export. Carga el checkpoint entero en RAM, de ahi el aviso de RAM del
    pre-flight.
    """
    from diffusers import DiffusionPipeline

    pipeline = DiffusionPipeline.from_single_file(
        str(checkpoint_path), local_files_only=False
    )
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        pipeline.save_pretrained(str(out_dir))
    finally:
        del pipeline
