from __future__ import annotations

from pathlib import PurePosixPath

# Que archivos hace falta bajar de un repo de reconocimiento de voz.
#
# Medido el 2026-07-29 contra onnx-community/whisper-tiny.en: con exactamente estos
# archivos, ORTModelForSpeechSeq2Seq.from_pretrained(dir, use_merged=False) y
# AutoProcessor.from_pretrained(dir) cargan y transcriben. 14 archivos, 257 MB.
#
# No es una lista adivinada: bajar de menos da una instalacion que falla al cargar,
# y bajar de mas trae las seis variantes cuantizadas del repo (fp16, int8, bnb4, q4,
# uint8, quantized) que multiplican el peso sin que se usen.

# El par NO merged. `use_merged=False` no es opcional: con el decoder merged
# DirectML carga, corre sin excepcion y devuelve basura (ver el spike de whisper),
# y ese par necesita decoder_model MAS decoder_with_past_model.
_REQUIRED_ONNX = (
    "onnx/encoder_model.onnx",
    "onnx/decoder_model.onnx",
    "onnx/decoder_with_past_model.onnx",
)

# Config del modelo, del generador y del extractor de features, mas el tokenizer.
# Son KBs y sin ellos no carga.
_METADATA_SUFFIXES = (".json", ".txt")


def is_required_asr_file(path: str) -> bool:
    """Un archivo del repo entra en la instalacion.

    Solo la raiz para la metadata: un .json dentro de onnx/ es config de una
    variante cuantizada que no se usa.
    """
    normalized = path.replace("\\", "/")
    if normalized in _REQUIRED_ONNX:
        return True
    if "/" in normalized:
        return False
    return PurePosixPath(normalized).suffix.lower() in _METADATA_SUFFIXES


def select_asr_files(filenames: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in filenames if is_required_asr_file(name))


def missing_required_onnx(filenames: tuple[str, ...]) -> tuple[str, ...]:
    """Los ONNX obligatorios que el repo NO publica.

    Se chequea ANTES de bajar: descubrirlo al final significaria dejar cientos de
    megas en disco para despues fallar al cargar.
    """
    available = {name.replace("\\", "/") for name in filenames}
    return tuple(name for name in _REQUIRED_ONNX if name not in available)
