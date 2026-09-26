"""Reescaladores IA del carril visual de CCTV y su etiqueta Generative (spec §4.7 puntos 5 y 9).

Solo entran los modelos builtin con export ONNX para la escala pedida: son los
que corre la etapa compuesta dentro del stream. ncnn y los modelos de Hugging
Face quedan fuera del carril en v1. Un reescalador general puede inventar rasgos
y caracteres de placas, por eso cada modelo lleva su etiqueta y el default es "None".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.backend_registry import get_builtin_onnx_model

GENERATIVE_LABEL = "Generative (invents texture)"
NON_GENERATIVE_LABEL = "Non-generative"
AI_UPSCALE_SCALES = (2, 3, 4)

ExportInstalled = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class StreamUpscaleModel:
    id: str
    label: str
    scales: tuple[int, ...]
    generative: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "scales": list(self.scales),
            "generative": self.generative,
            "generativeLabel": generative_label(self.generative),
        }


def generative_label(generative: bool) -> str:
    return GENERATIVE_LABEL if generative else NON_GENERATIVE_LABEL


def runs_in_stream(model_id: str, scale: int) -> bool:
    return scale in AI_UPSCALE_SCALES and get_builtin_onnx_model(model_id, scale) is not None


def export_filename(model_id: str, scale: int) -> str | None:
    model = get_builtin_onnx_model(model_id, scale)
    return None if model is None else model.filename


def export_on_disk(directory: Path, filename: str) -> bool:
    return (directory / filename).is_file()


def installed_stream_scales(model_id: str, scales: Iterable[int], installed: ExportInstalled) -> tuple[int, ...]:
    return tuple(
        scale for scale in scales if runs_in_stream(model_id, scale) and installed(export_filename(model_id, scale))
    )


def catalog_generative(catalog: Sequence[Mapping[str, Any]], model_id: str) -> bool:
    # Un modelo sin declarar se trata como generativo: de el no se sabe si inventa textura.
    entry = next((option for option in catalog if option["key"] == model_id), None)
    return True if entry is None else bool(entry.get("generative", True))


def stream_model(option: Mapping[str, Any], installed: ExportInstalled) -> StreamUpscaleModel:
    return StreamUpscaleModel(
        id=option["key"],
        label=option["label"],
        scales=installed_stream_scales(option["key"], option["scales"], installed),
        generative=bool(option.get("generative", True)),
    )


def stream_upscale_models(
    catalog: Sequence[Mapping[str, Any]], installed: ExportInstalled
) -> tuple[StreamUpscaleModel, ...]:
    models = (stream_model(option, installed) for option in catalog)
    return tuple(model for model in models if model.scales)
