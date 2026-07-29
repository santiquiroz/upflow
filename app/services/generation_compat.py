from __future__ import annotations

from typing import Literal

CompatVerdict = Literal[
    "ready_onnx",
    "needs_conversion",
    "single_file",
    "gated",
    "incompatible",
]

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
) -> tuple[CompatVerdict, str]:
    # gated primero y sin excepcion: sin token no se puede leer nada mas del
    # repo, asi que cualquier otro veredicto seria una conjetura.
    if gated:
        return (
            "gated",
            "Repo con acceso restringido: necesita un token de Hugging Face y aceptar la licencia.",
        )

    if MODEL_INDEX_FILENAME not in filenames:
        if _has_root_safetensors(filenames):
            return (
                "single_file",
                "Tiene checkpoints .safetensors en la raiz: hay que evaluar sus "
                "headers antes de saber cuales se pueden instalar.",
            )
        return (
            "incompatible",
            f"No es un pipeline diffusers: falta {MODEL_INDEX_FILENAME}.",
        )

    if not has_component_weights(filenames):
        return (
            "incompatible",
            "Los pesos estan sueltos en la raiz del repo, sin carpetas por componente "
            "(unet, vae, text_encoder...). Es un checkpoint single-file: este instalador "
            "necesita el layout de carpetas de diffusers.",
        )

    torch_dirs = _top_level_dirs_with(filenames, _TORCH_SUFFIXES)
    onnx_dirs = _top_level_dirs_with(filenames, (_ONNX_SUFFIX,))
    missing_onnx = sorted(torch_dirs - onnx_dirs)
    if missing_onnx:
        return (
            "needs_conversion",
            "Sin ONNX propio para " + ", ".join(missing_onnx) + ": requiere conversion local.",
        )
    return "ready_onnx", "Trae ONNX para todos los componentes: se instala directo."
