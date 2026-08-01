from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings

# ---------------------------------------------------------------------------
# Modelos del lane Vulkan (stable-diffusion.cpp).
#
# Este lane corre los checkpoints .safetensors/.gguf TAL CUAL los publica la
# comunidad: sin exportar a ONNX y sin los ~40 minutos de conversión. Por eso
# soporta varios y no uno solo -- cada archivo que aparece en la carpeta es un
# modelo elegible, y bajar uno nuevo es copiar un archivo.
#
# El id lleva prefijo para que el job manager sepa a qué motor mandarlo sin
# consultar el registro (los modelos de este lane no viven ahí: no hay nada que
# instalar ni convertir, solo un archivo en disco).
# ---------------------------------------------------------------------------

SDCPP_MODEL_PREFIX = "sdcpp:"
CHECKPOINT_SUFFIXES = (".safetensors", ".gguf", ".ckpt")


@dataclass(frozen=True, slots=True)
class SdcppModel:
    id: str
    name: str
    path: Path


def sdcpp_model_id(path: Path) -> str:
    return f"{SDCPP_MODEL_PREFIX}{path.stem}"


def list_sdcpp_models(settings: Settings) -> list[SdcppModel]:
    models: list[SdcppModel] = []
    seen: set[Path] = set()

    models_dir = settings.sdcpp_models_dir_path
    if models_dir.is_dir():
        for path in sorted(models_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES:
                models.append(SdcppModel(id=sdcpp_model_id(path), name=path.stem, path=path))
                seen.add(path.resolve())

    # Instalaciones anteriores a la carpeta apuntaban a un archivo suelto.
    legacy = settings.sdcpp_model_path
    if legacy is not None and legacy.is_file() and legacy.resolve() not in seen:
        models.append(SdcppModel(id=sdcpp_model_id(legacy), name=legacy.stem, path=legacy))
    return models


def resolve_sdcpp_model(model_id: str, settings: Settings) -> Path | None:
    """Ruta del checkpoint, o None si el id no es de este lane o no existe.

    Se resuelve contra la lista real y no armando una ruta con el id: el id
    llega por la API y concatenarlo dejaría leer cualquier archivo del disco.
    """
    if not model_id.startswith(SDCPP_MODEL_PREFIX):
        return None
    for model in list_sdcpp_models(settings):
        if model.id == model_id:
            return model.path
    return None
