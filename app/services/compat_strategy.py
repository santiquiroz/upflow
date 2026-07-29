from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.services.generation_compat import (
    CompatReason,
    CompatVerdict,
    classify as classify_generation,
)
from app.services.generation_variants import available_precisions_from_names

# Lo unico especifico de un dominio en todo el descubrimiento de modelos es la
# CLASIFICACION: mirar los archivos de un repo y decidir si sirve y como. El
# pre-flight (disco, VRAM por dispositivo, RAM), los avisos y la tarjeta ya son
# genericos. Esta es la costura por donde entra cada dominio.

_ONNX_SUFFIX = ".onnx"
# Mismo orden de preferencia que pick_weight_file: un .onnx se instala directo y
# cualquier otro formato de pesos tiene que pasar por Spandrel.
_CONVERTIBLE_SUFFIXES = (".safetensors", ".pth")


@dataclass(frozen=True, slots=True)
class InstallOptions:
    """Que le podemos ofrecer a elegir al usuario antes de instalar.

    Vacio no es un caso degradado: los upscalers no ofrecen nada porque el
    instalador elige el archivo de pesos por su cuenta.
    """

    precisions: tuple[str, ...] = ()
    checkpoints: tuple[str, ...] = field(default=())


class CompatStrategy(Protocol):
    def classify(
        self, filenames: tuple[str, ...], gated: bool | str | None
    ) -> tuple[CompatVerdict, CompatReason]: ...

    def install_options(self, filenames: tuple[str, ...]) -> InstallOptions: ...


class GenerationCompatStrategy:
    """Envuelve lo que ya existia, sin cambiarle el comportamiento.

    Los tests de generation_compat son el contrato: si alguno cambiara, el
    refactor estaria mal.
    """

    def classify(
        self, filenames: tuple[str, ...], gated: bool | str | None
    ) -> tuple[CompatVerdict, CompatReason]:
        return classify_generation(filenames, gated)

    def install_options(self, filenames: tuple[str, ...]) -> InstallOptions:
        return InstallOptions(
            precisions=tuple(available_precisions_from_names(list(filenames))),
            checkpoints=tuple(
                name
                for name in filenames
                if "/" not in name and name.lower().endswith(".safetensors")
            ),
        )


def _has_suffix(filenames: tuple[str, ...], suffixes: tuple[str, ...]) -> bool:
    return any(Path(name).suffix.lower() in suffixes for name in filenames)


class UpscalerCompatStrategy:
    """La clasificacion que hasta ahora vivia implicita en el instalador.

    `pick_weight_file` prefiere .onnx sobre .safetensors sobre .pth, y el
    instalador convierte con Spandrel todo lo que no sea .onnx. Eso es
    exactamente un veredicto de compatibilidad, solo que se descubria recien al
    apretar instalar.
    """

    def classify(
        self, filenames: tuple[str, ...], gated: bool | str | None
    ) -> tuple[CompatVerdict, CompatReason]:
        # gated primero y sin excepcion: sin token no se puede leer nada mas del
        # repo, asi que cualquier otro veredicto seria una conjetura.
        if gated:
            return "gated", CompatReason("compat.gated")

        if _has_suffix(filenames, (_ONNX_SUFFIX,)):
            return "ready_onnx", CompatReason("compat.upscaler.readyOnnx")

        if _has_suffix(filenames, _CONVERTIBLE_SUFFIXES):
            return "needs_conversion", CompatReason("compat.upscaler.needsConversion")

        return "incompatible", CompatReason("compat.upscaler.noWeights")

    def install_options(self, filenames: tuple[str, ...]) -> InstallOptions:
        # El instalador de upscalers elige el archivo de pesos solo, con
        # pick_weight_file: no hay nada que preguntarle al usuario todavia.
        return InstallOptions()


def strategy_for(domain: str) -> CompatStrategy:
    if domain == "generate":
        return GenerationCompatStrategy()
    if domain in ("video", "image"):
        return UpscalerCompatStrategy()
    raise ValueError(f"No hay estrategia de compatibilidad para el dominio {domain!r}.")
