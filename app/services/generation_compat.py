from __future__ import annotations

from typing import Literal

CompatVerdict = Literal["ready_onnx", "needs_conversion", "gated", "incompatible"]

MODEL_INDEX_FILENAME = "model_index.json"
_TORCH_SUFFIXES = (".safetensors", ".bin")
_ONNX_SUFFIX = ".onnx"


def _top_level_dirs_with(filenames: tuple[str, ...], suffixes: tuple[str, ...]) -> set[str]:
    return {
        name.split("/", 1)[0]
        for name in filenames
        if "/" in name and name.lower().endswith(suffixes)
    }


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
        return (
            "incompatible",
            f"No es un pipeline diffusers: falta {MODEL_INDEX_FILENAME}.",
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
