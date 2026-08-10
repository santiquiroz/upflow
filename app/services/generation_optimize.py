"""Reglas de elegibilidad de "optimizar modelo" (fusión de grafo del UNet).

Lógica pura: qué modelos pueden optimizarse, cuánta RAM pide cada arquitectura y
cómo se llama la variante resultante. La fusión en sí vive en
`generation_graph_fusion.py`; el carril de trabajo, en `generation_converter.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024 ** 3

OPTIMIZED_SUFFIX = "--optimized"
INPAINT_SUFFIX = "--inpainting"


class OptimizeUnsupportedError(ValueError):
    """El modelo existe pero no se puede optimizar. El mensaje dice por qué."""


@dataclass(frozen=True, slots=True)
class OptimizeArchitecture:
    key: str
    label: str
    # Pico de RAM del proceso durante export fp32 + fusión. La conversión corre
    # en CPU y pide varios GB de una sola vez; sin este gate el job muere por OOM
    # a los diez minutos en vez de rechazarse en el acto.
    peak_ram_bytes: int
    # Sólo para la copia de la UI: la ganancia end-to-end medida.
    expected_speedup: str


# Medido en el spike del 2026-08-10 (RX 7800 XT, DirectML):
#   SDXL  UNet 1439 -> 508 ms (2.83x), generación completa 20.07 -> 6.61 s (3.03x)
#   SD15  UNet  132 ->  91 ms (1.44x), generación completa  2.79 -> 2.14 s (1.31x)
# Confirmado end-to-end por este carril con dreamshaper-8 (SD15): 2.84 -> 2.20 s
# (1.29x), export 74 s + fusión 42 s + fp16 44 s.
SDXL = OptimizeArchitecture(
    key="sdxl",
    label="SDXL",
    # Pico REAL medido durante el export fp32 + fusión de epicrealism SDXL.
    peak_ram_bytes=50 * GIB,
    expected_speedup="3x",
)
SD15 = OptimizeArchitecture(
    key="sd15",
    label="SD 1.5",
    # Pico REAL medido en el smoke de dreamshaper-8 (25.7 GiB de working set).
    # La estimación previa por relación de parámetros del UNet daba 16 GiB y se
    # quedaba corta por 10: el costo no escala con el tamaño del modelo, hay un
    # piso grande de export torch->ONNX que SD15 paga igual.
    peak_ram_bytes=26 * GIB,
    expected_speedup="1.3x",
)

# Clase declarada en el model_index.json del pipeline instalado -> arquitectura.
# Se listan las tres familias de prefijo porque el `_class_name` cambia según
# quién escribió el pipeline (diffusers, el export de optimum viejo `Onnx*`, o el
# nuevo `ORT*`).
_ARCHITECTURE_BY_CLASS: dict[str, OptimizeArchitecture] = {
    name: architecture
    for architecture, bases in (
        (
            SD15,
            (
                "StableDiffusionPipeline",
                "StableDiffusionImg2ImgPipeline",
                "StableDiffusionInpaintPipeline",
            ),
        ),
        (
            SDXL,
            (
                "StableDiffusionXLPipeline",
                "StableDiffusionXLImg2ImgPipeline",
                "StableDiffusionXLInpaintPipeline",
            ),
        ),
    )
    for base in bases
    for name in (base, f"ORT{base}", f"Onnx{base}")
}


def architecture_for(class_name: str) -> OptimizeArchitecture:
    """Arquitectura de fusión de una clase de pipeline declarada.

    Fuera quedan a propósito SD3, Flux, Sana, PixArt, Kandinsky y los pipelines
    de consistencia latente: la fusión de `onnxruntime.transformers` está escrita
    contra el UNet de Stable Diffusion, y en un backbone distinto (o en un LCM,
    donde la clase no dice si el UNet es de SD15 o de SDXL) no se puede ni
    garantizar la fusión ni estimar la RAM que va a pedir.
    """
    architecture = _ARCHITECTURE_BY_CLASS.get(class_name)
    if architecture is None:
        raise OptimizeUnsupportedError(
            f"La optimización sólo está probada sobre Stable Diffusion 1.5 y SDXL; "
            f"este modelo declara {class_name}."
        )
    return architecture


def optimized_model_id(model_id: str) -> str:
    return f"{model_id}{OPTIMIZED_SUFFIX}"


def optimized_display_name(name: str) -> str:
    return f"{name} (optimized)"


def is_optimized(model_id: str) -> bool:
    return model_id.endswith(OPTIMIZED_SUFFIX)


def is_inpaint_merge(model_id: str) -> bool:
    return model_id.endswith(INPAINT_SUFFIX)


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def ensure_enough_ram(
    architecture: OptimizeArchitecture, free_ram_bytes: int | None
) -> None:
    """Rechaza antes de encolar cuando la RAM libre no alcanza.

    `None` = no se pudo medir: se deja pasar (mismo fail-open que el resto de las
    sondas de capacidad del repo). Medir mal no puede ser motivo de bloqueo.
    """
    if free_ram_bytes is None or free_ram_bytes >= architecture.peak_ram_bytes:
        return
    raise OptimizeUnsupportedError(
        f"Optimizar un {architecture.label} necesita unos "
        f"{_gib(architecture.peak_ram_bytes)} de RAM libre y ahora hay "
        f"{_gib(free_ram_bytes)}. La conversión corre en CPU y pide todo de una "
        "vez: cerrá otras aplicaciones y volvé a intentar."
    )
