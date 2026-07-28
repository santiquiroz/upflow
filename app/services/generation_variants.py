from __future__ import annotations

from typing import Literal

from app.services.hf_client import HfFile

Precision = Literal["fp16", "fp32"]

MODEL_INDEX_FILENAME = "model_index.json"

# La conversion descarga los pesos PyTorch que el installer normal EXCLUYE
# (.safetensors/.bin son la fuente del export). .ckpt/.msgpack/.h5 fuera: son
# duplicados legacy de los mismos pesos. .onnx fuera: si el repo ya trae ONNX
# no deberia estar en este camino.
CONVERSION_SKIP_SUFFIXES = (".ckpt", ".msgpack", ".h5", ".onnx", ".onnx_data", ".pb")

WEIGHT_SUFFIXES = (".safetensors", ".bin")
_FP16_MARKER = ".fp16."
# Checkpoint de entrenamiento: la inferencia no lo lee nunca. En SD1.5 son
# 3278 MB que se bajaban para nada.
_NON_EMA_MARKER = ".non_ema."


def _is_weight(path: str) -> bool:
    return path.lower().endswith(WEIGHT_SUFFIXES)


def _is_usable_weight(path: str) -> bool:
    return _is_weight(path) and _NON_EMA_MARKER not in path.lower()


def _precision_of(path: str) -> Precision:
    return "fp16" if _FP16_MARKER in path.lower() else "fp32"


def _component_of(path: str) -> str:
    return path.rsplit("/", 1)[0]


def available_precisions_from_names(filenames: tuple[str, ...]) -> list[Precision]:
    found = {
        _precision_of(name)
        for name in filenames
        if "/" in name and _is_usable_weight(name)
    }
    return [p for p in ("fp16", "fp32") if p in found]


def available_precisions(files: list[HfFile]) -> tuple[Precision, ...]:
    return tuple(available_precisions_from_names(tuple(f.path for f in files)))


def canonical_weight_name(path: str) -> str:
    """Quita el marcador de variante del nombre. main_export no expone un
    selector de variante, asi que la eleccion se materializa escribiendo el
    archivo elegido bajo el nombre que diffusers busca. safetensors lleva el
    dtype en su propio header, asi que el rename no pierde informacion."""
    if not _is_weight(path):
        return path
    directory, _, name = path.rpartition("/")
    replaced = name.replace(_FP16_MARKER, ".", 1) if _FP16_MARKER in name.lower() else name
    return f"{directory}/{replaced}" if directory else replaced


def _pick_weight_for_component(weights: list[HfFile], precision: Precision) -> HfFile:
    """Un solo archivo de pesos por componente: la precision pedida si existe,
    si no la otra. Entre empatados gana .safetensors sobre .bin."""
    preferred = [f for f in weights if _precision_of(f.path) == precision]
    candidates = preferred or weights
    return min(
        candidates,
        key=lambda f: (0 if f.path.lower().endswith(".safetensors") else 1, f.path),
    )


def select_for_precision(
    files: list[HfFile], declared: list[str], precision: Precision
) -> list[HfFile]:
    declared_set = set(declared)
    kept: list[HfFile] = []
    weights_by_component: dict[str, list[HfFile]] = {}

    for hf_file in files:
        path = hf_file.path
        lowered = path.lower()
        if path == MODEL_INDEX_FILENAME or lowered.endswith(CONVERSION_SKIP_SUFFIXES):
            continue
        if "/" not in path:
            if lowered.endswith((".json", ".txt")):
                kept.append(hf_file)
            continue
        if path.split("/", 1)[0] not in declared_set:
            continue
        if _is_weight(path):
            if _is_usable_weight(path):
                weights_by_component.setdefault(_component_of(path), []).append(hf_file)
            continue
        kept.append(hf_file)

    for weights in weights_by_component.values():
        kept.append(_pick_weight_for_component(weights, precision))
    return kept
