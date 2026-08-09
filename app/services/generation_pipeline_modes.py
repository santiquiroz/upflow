from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Fusion de los ex-modulos gemelos generation_img2img y generation_inpaint:
# un registro por modo (tabla de clases, exclusiones con motivo, error propio,
# clase base ORT para validar) + funciones parametrizadas por modo. Los
# wrappers del final conservan los nombres publicos historicos.

# ---------------------------------------------------------------------------
# Imagen a imagen reusa el MISMO modelo instalado que texto a imagen: los pesos
# son los mismos y solo cambia la clase de pipeline que los carga. Lo que NO es
# igual es la cobertura.
#
# Medido el 2026-07-29 contra ORTPipelineForImage2Image.ort_pipelines_mapping:
#   texto a imagen : latent-consistency, stable-diffusion, stable-diffusion-xl,
#                    stable-diffusion-3, flux, sana
#   imagen a imagen: latent-consistency, stable-diffusion, stable-diffusion-xl,
#                    stable-diffusion-3
#
# Falta flux y falta sana. Un modelo instalado y perfectamente usable para texto a
# imagen puede no servir para imagen a imagen, asi que el picker TIENE que filtrar
# por la clase declarada y no ofrecer todo lo instalado. Chequear que exista la
# clase no sirve como test -- ya paso al revés con sana, que optimum SI ejecuta
# aunque ORTSanaPipeline no exista: validate_table() cruza contra el mapping real
# de la versión instalada.
# ---------------------------------------------------------------------------

# Clase declarada en el model_index.json del pipeline instalado -> clase ORT de
# imagen a imagen que lo carga.
IMG2IMG_CLASS_NAMES: dict[str, str] = {
    "OnnxStableDiffusionPipeline": "ORTStableDiffusionImg2ImgPipeline",
    "ORTStableDiffusionPipeline": "ORTStableDiffusionImg2ImgPipeline",
    "StableDiffusionPipeline": "ORTStableDiffusionImg2ImgPipeline",
    "StableDiffusionXLPipeline": "ORTStableDiffusionXLImg2ImgPipeline",
    "ORTStableDiffusionXLPipeline": "ORTStableDiffusionXLImg2ImgPipeline",
    "StableDiffusion3Pipeline": "ORTStableDiffusion3Img2ImgPipeline",
    "ORTStableDiffusion3Pipeline": "ORTStableDiffusion3Img2ImgPipeline",
    "LatentConsistencyModelPipeline": "ORTLatentConsistencyModelImg2ImgPipeline",
    "ORTLatentConsistencyModelPipeline": "ORTLatentConsistencyModelImg2ImgPipeline",
}

# Las que cargan para texto a imagen pero NO tienen camino de imagen a imagen. Se
# nombran explicitamente para que el motivo del rechazo diga cual es el problema
# en vez de "clase no soportada".
TEXT_ONLY_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "FluxPipeline",
        "ORTFluxPipeline",
        "SanaPipeline",
        "ORTSanaPipeline",
    }
)


class Img2ImgUnsupportedError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Inpainting (edición con máscara).
#
# Cobertura MEDIDA el 2026-07-31 contra ORTPipelineForInpainting.
# ort_pipelines_mapping de optimum-onnx 0.1.0 (el pin del proyecto):
#   inpainting = {stable-diffusion, stable-diffusion-xl, stable-diffusion-3}
# Quedan FUERA (tienen otras rutas pero no inpainting en optimum-onnx):
#   - latent-consistency: tiene img2img pero NO inpainting.
#   - flux, sana: texto a imagen solamente (igual que en img2img).
# ---------------------------------------------------------------------------

# Mapea tanto el nombre diffusers (repo HF) como el nombre ORT (repo convertido
# por la app declara la clase ORT en su model_index.json).
INPAINT_CLASS_NAMES: dict[str, str] = {
    "StableDiffusionPipeline": "ORTStableDiffusionInpaintPipeline",
    "ORTStableDiffusionPipeline": "ORTStableDiffusionInpaintPipeline",
    "StableDiffusionImg2ImgPipeline": "ORTStableDiffusionInpaintPipeline",
    "StableDiffusionXLPipeline": "ORTStableDiffusionXLInpaintPipeline",
    "ORTStableDiffusionXLPipeline": "ORTStableDiffusionXLInpaintPipeline",
    "StableDiffusionXLImg2ImgPipeline": "ORTStableDiffusionXLInpaintPipeline",
    "StableDiffusion3Pipeline": "ORTStableDiffusion3InpaintPipeline",
    "ORTStableDiffusion3Pipeline": "ORTStableDiffusion3InpaintPipeline",
    # Checkpoints de inpainting DEDICADOS (unet de 9 canales): su model_index
    # ya declara la clase de inpainting. Son los que de verdad borran limpio —
    # el unet recibe la máscara como entrada en vez de adivinar — así que hay
    # que aceptarlos tal como se declaran, no solo los checkpoints normales.
    "StableDiffusionInpaintPipeline": "ORTStableDiffusionInpaintPipeline",
    "ORTStableDiffusionInpaintPipeline": "ORTStableDiffusionInpaintPipeline",
    "StableDiffusionXLInpaintPipeline": "ORTStableDiffusionXLInpaintPipeline",
    "ORTStableDiffusionXLInpaintPipeline": "ORTStableDiffusionXLInpaintPipeline",
    "StableDiffusion3InpaintPipeline": "ORTStableDiffusion3InpaintPipeline",
    "ORTStableDiffusion3InpaintPipeline": "ORTStableDiffusion3InpaintPipeline",
}

# Arquitecturas que la app conoce pero que NO tienen camino de inpainting en
# optimum-onnx: motivo de rechazo distinto al de una clase desconocida.
NO_INPAINT_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "FluxPipeline",
        "ORTFluxPipeline",
        "SanaPipeline",
        "ORTSanaPipeline",
        "LatentConsistencyModelPipeline",
        "ORTLatentConsistencyModelPipeline",
    }
)


class InpaintUnsupportedError(RuntimeError):
    pass


@dataclass(frozen=True, eq=False)
class PipelineMode:
    label: str
    table: dict[str, str]
    excluded: frozenset[str]
    error_class: type[RuntimeError]
    excluded_reason: str
    ort_base_class_name: str


IMG2IMG_MODE = PipelineMode(
    label="imagen a imagen",
    table=IMG2IMG_CLASS_NAMES,
    excluded=TEXT_ONLY_CLASS_NAMES,
    error_class=Img2ImgUnsupportedError,
    excluded_reason=(
        "{declared} sirve para texto a imagen pero optimum-onnx no tiene pipeline "
        "de imagen a imagen para esa arquitectura. Usa un modelo Stable Diffusion, "
        "SDXL, SD3 o Latent Consistency."
    ),
    ort_base_class_name="ORTPipelineForImage2Image",
)

INPAINT_MODE = PipelineMode(
    label="inpainting",
    table=INPAINT_CLASS_NAMES,
    excluded=NO_INPAINT_CLASS_NAMES,
    error_class=InpaintUnsupportedError,
    excluded_reason=(
        "El modelo ({declared}) genera imágenes pero optimum-onnx no tiene "
        "pipeline de inpainting para esa arquitectura. Usa un modelo Stable "
        "Diffusion, SDXL o SD3."
    ),
    ort_base_class_name="ORTPipelineForInpainting",
)


def class_for(mode: PipelineMode, declared_class_name: str) -> str:
    resolved = mode.table.get(declared_class_name)
    if resolved is not None:
        return resolved
    if declared_class_name in mode.excluded:
        raise mode.error_class(mode.excluded_reason.format(declared=declared_class_name))
    raise mode.error_class(
        f"Clase de pipeline desconocida para {mode.label}: {declared_class_name!r}."
    )


def supports(mode: PipelineMode, declared_class_name: str) -> bool:
    return declared_class_name in mode.table


def load_class(mode: PipelineMode, declared_class_name: str) -> Any:
    import optimum.onnxruntime as ort_module

    return getattr(ort_module, class_for(mode, declared_class_name))


def validate_table(mode: PipelineMode) -> None:
    """Falla ruidosamente si un upgrade de optimum rompe el mapeo.

    Sin esto, un rename de clase degradaria en silencio a "arquitectura no
    soportada" para todos los modelos.
    """
    import optimum.onnxruntime as ort_module

    base_class = getattr(ort_module, mode.ort_base_class_name)
    supported_classes = {cls.__name__ for cls in base_class.ort_pipelines_mapping.values()}
    targets = set(mode.table.values())

    unknown = sorted(targets - supported_classes)
    if unknown:
        raise RuntimeError(
            f"optimum-onnx ya no expone estas clases de {mode.label}: "
            + ", ".join(unknown)
            + f". Soportadas hoy: {sorted(supported_classes)}. Revisar la tabla del modo."
        )

    missing_attr = sorted(name for name in targets if not hasattr(ort_module, name))
    if missing_attr:
        raise RuntimeError(
            "Estas clases estan en el mapping de optimum pero no se pueden importar: "
            + ", ".join(missing_attr)
        )


def is_dedicated_inpaint_class(declared_class_name: str) -> bool:
    """Checkpoint de inpainting DEDICADO (unet 9ch): solo sabe editar con
    máscara — no puede generar texto a imagen ni imagen a imagen."""
    return "Inpaint" in declared_class_name and declared_class_name in INPAINT_CLASS_NAMES


def img2img_class_for(declared_class_name: str) -> str:
    return class_for(IMG2IMG_MODE, declared_class_name)


def supports_img2img(declared_class_name: str) -> bool:
    return supports(IMG2IMG_MODE, declared_class_name)


def load_img2img_class(declared_class_name: str) -> Any:
    return load_class(IMG2IMG_MODE, declared_class_name)


def validate_img2img_table() -> None:
    validate_table(IMG2IMG_MODE)


def inpaint_class_for(declared_class_name: str) -> str:
    return class_for(INPAINT_MODE, declared_class_name)


def supports_inpaint(declared_class_name: str) -> bool:
    return supports(INPAINT_MODE, declared_class_name)


def load_inpaint_class(declared_class_name: str) -> Any:
    return load_class(INPAINT_MODE, declared_class_name)


def validate_inpaint_table() -> None:
    validate_table(INPAINT_MODE)
