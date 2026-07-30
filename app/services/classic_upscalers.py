from __future__ import annotations

from dataclasses import dataclass

# Reescalado CLASICO: sin IA, sin modelo, sin descarga. Lo hace swscale, que ya viene
# en el ffmpeg vendorizado.
#
# No es un plan B. Hay tres casos donde es la herramienta CORRECTA y no un compromiso:
#
#   1. Reducir (4K a 1080p). Un modelo de super-resolucion no tiene nada que aportar
#      cuando sobra informacion: lanczos es lo que hay que usar. Y es el caso que
#      motivo todo esto -- alguien pidio x4 sobre un 4K y esperó 2,8 horas.
#   2. Trabajos rapidos. La inferencia esta dominada por la resolucion de ENTRADA (ver
#      la nota en target_resolution.py), asi que en material grande el modelo cuesta
#      caro incluso para subir poco. Clasico corre en tiempo de encode.
#   3. Material que la IA arruina. Un upscaler entrenado en fotos mete artefactos en
#      capturas de pantalla, texto o line art plano.
#
# Se exponen como entradas del catalogo con categoria propia, para que la eleccion diga
# CLARAMENTE que no usan IA en vez de esconderlo en una opcion avanzada.

CLASSIC_CATEGORY = "classic"

# Escala 1 incluida a proposito: es lo que habilita reducir o cambiar de resolucion sin
# ampliar. Los modelos no la ofrecen porque no tendria sentido para ellos.
CLASSIC_SCALES = (1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class ClassicUpscaler:
    key: str
    # Flag de libswscale. Verificado contra el ffmpeg vendorizado el 2026-07-29.
    swscale_flag: str
    label: str
    description: str


# Textos en ingles como el resto de MODEL_CATALOG: es el locale base de la app (medido:
# 1550 marcadores en ingles contra 53 en español), y estas entradas viajan por la misma
# API que las demas.
CLASSIC_UPSCALERS: tuple[ClassicUpscaler, ...] = (
    ClassicUpscaler(
        key="classic-lanczos",
        swscale_flag="lanczos",
        label="Lanczos (no AI)",
        description=(
            "Best for DOWNSCALING and the sharpest choice for small upscales. "
            "No model, no GPU: runs in seconds."
        ),
    ),
    ClassicUpscaler(
        key="classic-spline",
        swscale_flag="spline",
        label="Spline (no AI)",
        description=(
            "Softer than Lanczos, without the halo it can leave on hard edges. "
            "Good for high-contrast footage."
        ),
    ),
    ClassicUpscaler(
        key="classic-bicubic",
        swscale_flag="bicubic",
        label="Bicubic (no AI)",
        description=(
            "The old reliable: predictable result and the fastest of the three. "
            "Use it when speed matters more than detail."
        ),
    ),
    ClassicUpscaler(
        key="classic-neighbor",
        swscale_flag="neighbor",
        label="Nearest neighbor (no AI)",
        description=(
            "Does not interpolate, it duplicates pixels. The only correct option for "
            "pixel art, where any smoothing ruins the style."
        ),
    ),
)

_BY_KEY = {upscaler.key: upscaler for upscaler in CLASSIC_UPSCALERS}


def is_classic_upscaler(model_key: str | None) -> bool:
    return model_key in _BY_KEY


def get_classic_upscaler(model_key: str) -> ClassicUpscaler | None:
    return _BY_KEY.get(model_key)


def swscale_flag_for(model_key: str | None, default: str = "lanczos") -> str:
    upscaler = _BY_KEY.get(model_key or "")
    return upscaler.swscale_flag if upscaler else default


def catalog_entries() -> list[dict]:
    """Las entradas para el catalogo de modelos.

    Misma forma que un ModelOption para que el selector de la UI y la validacion no
    necesiten un camino aparte. `engine_name` lleva el flag de swscale porque es el
    equivalente al nombre de modelo del motor: lo que se le pasa al ejecutor.
    """
    return [
        {
            "key": upscaler.key,
            "engine_name": upscaler.swscale_flag,
            "label": upscaler.label,
            "category": CLASSIC_CATEGORY,
            "description": upscaler.description,
            "scales": list(CLASSIC_SCALES),
        }
        for upscaler in CLASSIC_UPSCALERS
    ]
