from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CompatVerdict = Literal[
    "ready_onnx",
    "needs_conversion",
    "single_file",
    "gated",
    "incompatible",
]

@dataclass(frozen=True, slots=True)
class CompatReason:
    """Clave de traduccion mas sus datos, no copia.

    Misma decision que el resto del backend desde el spec de i18n de 2026-07-29:
    con dos idiomas la unica fuente de verdad de la copia es el catalogo de
    traducciones, y tenerla aca significaria mantener dos catalogos en dos
    lenguajes distintos. Los params existen para no perder el DATO -- que
    componentes les falta el ONNX -- al pasar de oracion a clave.
    """

    key: str
    params: dict[str, str] = field(default_factory=dict)


MODEL_INDEX_FILENAME = "model_index.json"
_TORCH_SUFFIXES = (".safetensors", ".bin")
_ONNX_SUFFIX = ".onnx"


def _top_level_dirs_with(filenames: tuple[str, ...], suffixes: tuple[str, ...]) -> set[str]:
    return {
        name.split("/", 1)[0]
        for name in filenames
        if "/" in name and name.lower().endswith(suffixes)
    }


def has_component_weights(filenames: tuple[str, ...]) -> bool:
    """Hay al menos un peso DENTRO de una carpeta de componente.

    Un pipeline diffusers guarda unet/vae/text_encoder en carpetas propias por
    definicion, asi que sin esto no es ese layout -- es un repo de checkpoints
    sueltos (formato single-file, estilo Civitai), que este instalador no sabe
    usar aunque traiga un model_index.json.

    Existe porque comparar conjuntos de carpetas no distingue "todos los
    componentes torch tienen su ONNX" de "no hay ningun componente": la resta
    de dos conjuntos vacios tambien da vacio.
    """
    return bool(
        _top_level_dirs_with(filenames, _TORCH_SUFFIXES)
        | _top_level_dirs_with(filenames, (_ONNX_SUFFIX,))
    )


def _has_root_safetensors(filenames: tuple[str, ...]) -> bool:
    return any(
        "/" not in name and name.lower().endswith(".safetensors")
        for name in filenames
    )


def classify(
    filenames: tuple[str, ...], gated: bool | str | None
) -> tuple[CompatVerdict, CompatReason]:
    # gated primero y sin excepcion: sin token no se puede leer nada mas del
    # repo, asi que cualquier otro veredicto seria una conjetura.
    if gated:
        return "gated", CompatReason("compat.gated")

    if MODEL_INDEX_FILENAME not in filenames:
        if _has_root_safetensors(filenames):
            return "single_file", CompatReason("compat.singleFile")
        return (
            "incompatible",
            CompatReason("compat.noModelIndex", {"filename": MODEL_INDEX_FILENAME}),
        )

    if not has_component_weights(filenames):
        return "incompatible", CompatReason("compat.weightsAtRoot")

    torch_dirs = _top_level_dirs_with(filenames, _TORCH_SUFFIXES)
    onnx_dirs = _top_level_dirs_with(filenames, (_ONNX_SUFFIX,))
    missing_onnx = sorted(torch_dirs - onnx_dirs)
    if missing_onnx:
        return (
            "needs_conversion",
            CompatReason("compat.needsConversion", {"components": ", ".join(missing_onnx)}),
        )
    return "ready_onnx", CompatReason("compat.readyOnnx")
