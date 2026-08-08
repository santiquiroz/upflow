from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Convierte cualquier checkpoint de generación del usuario en su versión de
# INPAINTING de 9 canales con la receta clásica add-difference:
#
#     merged = inpaint_oficial + (checkpoint_usuario - base_oficial)
#
# aplicada SOLO al UNet. Los text encoders y el VAE se copian del checkpoint
# del usuario tal cual: el fine-tune de inpainting oficial solo tocó el UNet,
# así que el delta TE/VAE del inpaint contra el base es ~0 y aplicarle
# add-difference equivale numéricamente a copiar — copiar es más barato.
# ---------------------------------------------------------------------------

INPAINT_BASE_REPOS: dict[str, dict[str, str]] = {
    "sdxl": {
        "base": "stabilityai/stable-diffusion-xl-base-1.0",
        "inpaint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    },
    "sd15": {
        "base": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "inpaint": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    },
}

# Mapea tanto el nombre diffusers (repo HF) como el ORT (repo ya convertido por
# la app), misma convención que INPAINT_CLASS_NAMES en generation_inpaint.py.
_MERGE_FAMILIES: dict[str, str] = {
    "StableDiffusionPipeline": "sd15",
    "ORTStableDiffusionPipeline": "sd15",
    "StableDiffusionImg2ImgPipeline": "sd15",
    "StableDiffusionXLPipeline": "sdxl",
    "ORTStableDiffusionXLPipeline": "sdxl",
    "StableDiffusionXLImg2ImgPipeline": "sdxl",
}

# Sin par base+inpaint oficial del MISMO linaje no hay resta posible: SD3 y
# Flux no publican inpaint oficial de 9 canales, y un LCM destilado sumado
# sobre el UNet de inpainting rompería su contrato de sampling.
_NO_MERGE_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "StableDiffusion3Pipeline",
        "ORTStableDiffusion3Pipeline",
        "FluxPipeline",
        "ORTFluxPipeline",
        "SanaPipeline",
        "ORTSanaPipeline",
        "LatentConsistencyModelPipeline",
        "ORTLatentConsistencyModelPipeline",
    }
)

_CONV_IN_KEY = "conv_in.weight"
_MODEL_INDEX_FILENAME = "model_index.json"

# Componentes que salen del checkpoint del USUARIO (ver nota de cabecera); los
# que falten simplemente no se copian y el model_index final se ajusta.
_COMPONENTS_FROM_USER: tuple[str, ...] = (
    "vae",
    "text_encoder",
    "text_encoder_2",
    "tokenizer",
    "tokenizer_2",
    "scheduler",
    "feature_extractor",
)


class InpaintMergeUnsupportedError(RuntimeError):
    """La arquitectura no tiene par base/inpaint oficial para la receta."""


class InpaintMergeIncompatibleError(RuntimeError):
    """El checkpoint del usuario no comparte arquitectura con el base oficial."""


def merge_family_for(declared_class_name: str) -> str:
    family = _MERGE_FAMILIES.get(declared_class_name)
    if family is not None:
        return family
    if declared_class_name in _NO_MERGE_CLASS_NAMES:
        raise InpaintMergeUnsupportedError(
            f"El modelo ({declared_class_name}) no tiene checkpoint de inpainting "
            "oficial de 9 canales para aplicar add-difference. Usa un modelo "
            "Stable Diffusion 1.5 o SDXL."
        )
    raise InpaintMergeUnsupportedError(
        f"Clase de pipeline desconocida para el merge de inpainting: {declared_class_name!r}"
    )


def _add_difference(
    user_t: torch.Tensor,
    base_t: torch.Tensor,
    inpaint_t: torch.Tensor,
) -> torch.Tensor:
    # La aritmética va en float32 y recién al final se castea al dtype del
    # inpaint: restar directamente en fp16 acumula error de redondeo, y con
    # float32 el caso user == base devuelve el inpaint bit a bit.
    merged = inpaint_t.float() + (user_t.float() - base_t.float())
    return merged.to(inpaint_t.dtype)


def _merge_conv_in(
    user_t: torch.Tensor,
    base_t: torch.Tensor,
    inpaint_t: torch.Tensor,
) -> torch.Tensor:
    # El inpaint suma 5 canales de entrada (máscara + imagen enmascarada); el
    # delta del usuario solo existe para los canales latentes originales y los
    # nuevos quedan exactamente como los entrenó el inpaint oficial.
    merged = inpaint_t.float().clone()
    channels = user_t.shape[1]
    merged[:, :channels] += user_t.float() - base_t.float()
    return merged.to(inpaint_t.dtype)


def _is_widened_conv_in(
    key: str,
    user_t: torch.Tensor,
    inpaint_t: torch.Tensor,
) -> bool:
    return (
        key == _CONV_IN_KEY
        and user_t.dim() == 4
        and inpaint_t.dim() == 4
        and inpaint_t.shape[0] == user_t.shape[0]
        and inpaint_t.shape[1] > user_t.shape[1]
        and inpaint_t.shape[2:] == user_t.shape[2:]
    )


def _ensure_no_foreign_keys(
    user: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
) -> None:
    foreign = sorted(set(user) - set(base))
    if foreign:
        ejemplos = ", ".join(foreign[:5])
        raise InpaintMergeIncompatibleError(
            "El checkpoint trae claves de UNet que el base oficial no tiene "
            f"({ejemplos}): no es la misma arquitectura y add-difference no aplica."
        )


def merge_unet_state_dicts(
    user: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    inpaint: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Add-difference clave a clave sobre el UNet.

    El resultado tiene EXACTAMENTE las claves del inpaint: la arquitectura
    final es la del inpaint (quien declara in_channels=9), así que una clave de
    user∩base ausente en el inpaint se descarta y una clave solo-del-inpaint se
    copia tal cual.
    """
    _ensure_no_foreign_keys(user, base)
    merged: dict[str, torch.Tensor] = {}
    for key, inpaint_t in inpaint.items():
        user_t = user.get(key)
        base_t = base.get(key)
        if user_t is None or base_t is None:
            merged[key] = inpaint_t
            continue
        if user_t.shape != base_t.shape:
            raise InpaintMergeIncompatibleError(
                f"El checkpoint y el base oficial difieren en la forma de {key!r} "
                f"({tuple(user_t.shape)} contra {tuple(base_t.shape)}): "
                "no es la misma arquitectura."
            )
        if user_t.shape == inpaint_t.shape:
            merged[key] = _add_difference(user_t, base_t, inpaint_t)
        elif _is_widened_conv_in(key, user_t, inpaint_t):
            merged[key] = _merge_conv_in(user_t, base_t, inpaint_t)
        else:
            raise InpaintMergeIncompatibleError(
                f"La clave {key!r} tiene formas incompatibles entre el checkpoint "
                f"({tuple(user_t.shape)}) y el inpaint oficial "
                f"({tuple(inpaint_t.shape)})."
            )
    return merged


def _load_unet(pipeline_dir: Path) -> Any:
    # Imports perezosos: torch/diffusers nunca se cargan al importar
    # app.services (misma regla que generation_converter.py).
    import torch
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel.from_pretrained(
        str(pipeline_dir),
        subfolder="unet",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )


def _reset_out_dir(out_dir: Path) -> None:
    # Un merge anterior cortado a medias no debe contaminar el resultado nuevo.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _copy_user_components(user_dir: Path, out_dir: Path) -> None:
    for name in _COMPONENTS_FROM_USER:
        src = user_dir / name
        if src.is_dir():
            shutil.copytree(src, out_dir / name)


def _is_component_ref(value: Any) -> bool:
    # En model_index.json un componente es ["libreria", "Clase"]; el resto de
    # claves (flags como requires_safety_checker) se preservan siempre.
    return isinstance(value, list) and len(value) == 2


def _write_intersected_model_index(inpaint_dir: Path, out_dir: Path) -> None:
    # El model_index del INPAINT es quien declara la clase *InpaintPipeline,
    # pero solo puede listar componentes que de verdad quedaron en out_dir: si
    # el usuario no trae feature_extractor, declararlo rompería la carga.
    index = json.loads(
        (inpaint_dir / _MODEL_INDEX_FILENAME).read_text(encoding="utf-8")
    )
    kept = {
        key: value
        for key, value in index.items()
        if key.startswith("_")
        or not _is_component_ref(value)
        or (out_dir / key).is_dir()
    }
    (out_dir / _MODEL_INDEX_FILENAME).write_text(
        json.dumps(kept, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _notify(progress_cb: Callable[[str], None] | None, stage: str) -> None:
    if progress_cb is not None:
        progress_cb(stage)


def merge_to_inpaint_dir(
    user_dir: Path,
    base_dir: Path,
    inpaint_dir: Path,
    out_dir: Path,
    progress_cb: Callable[[str], None] | None = None,
) -> Path:
    """Arma en out_dir un pipeline diffusers de inpainting completo.

    Los tres dirs de entrada ya están descargados en formato diffusers; acá no
    hay red. Pico de RAM estimado: ~3x el UNet en fp16 (los tres cargados a la
    vez; SDXL ≈ 3 × 5 GiB) más temporarios float32 de a un tensor. Cargar de a
    un UNet bajaría el pico pero no hace falta en v1.
    """
    _notify(progress_cb, "loading")
    user_unet = _load_unet(user_dir)
    base_unet = _load_unet(base_dir)
    inpaint_unet = _load_unet(inpaint_dir)

    _notify(progress_cb, "merging")
    merged = merge_unet_state_dicts(
        user_unet.state_dict(),
        base_unet.state_dict(),
        inpaint_unet.state_dict(),
    )
    # El state dict mergeado se vuelca en el modelo del INPAINT: así el config
    # guardado es el del inpaint, que es quien declara in_channels=9.
    inpaint_unet.load_state_dict(merged)

    _notify(progress_cb, "saving")
    _reset_out_dir(out_dir)
    inpaint_unet.save_pretrained(str(out_dir / "unet"), safe_serialization=True)
    _copy_user_components(user_dir, out_dir)
    _write_intersected_model_index(inpaint_dir, out_dir)
    return out_dir
