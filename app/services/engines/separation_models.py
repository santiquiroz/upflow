"""Catalogo UNICO de modelos de separacion de stems, de las dos arquitecturas.

UNA sola lista a proposito: el usuario elige un modelo por lo que HACE
(instrumental, quitar reverb, quitar eco), no por como esta construido. El id
del modelo es lo unico que viaja en el job (`separation_model`), en el pack de
descarga y en la API; la arquitectura viaja solo como dato informativo y la usa
el pipeline para elegir motor.

El orden del dict ES el orden del picker: primero karaoke, despues limpieza.
"""

from __future__ import annotations

from pathlib import Path

from app.services.engines.mdx_models import MDX_MODELS
from app.services.engines.separation_spec import SeparationModelSpec
from app.services.engines.vr_models import VR_MODELS

DEFAULT_SEPARATION_MODEL = "inst_hq_3"

SEPARATION_MODELS: dict[str, SeparationModelSpec] = {**MDX_MODELS, **VR_MODELS}


def model_file(model_dir: Path, model_id: str) -> Path:
    return model_dir / SEPARATION_MODELS[model_id].filename


def installed_model_ids(model_dir: Path) -> list[str]:
    return [
        model_id
        for model_id in SEPARATION_MODELS
        if model_file(model_dir, model_id).exists()
    ]


def first_installed_model_path(model_dir: Path) -> Path | None:
    for model_id in installed_model_ids(model_dir):
        return model_file(model_dir, model_id)
    return None
