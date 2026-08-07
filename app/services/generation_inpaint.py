from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Inpainting (edición con máscara): misma tabla-módulo que generation_img2img.
#
# Cobertura MEDIDA el 2026-07-31 contra ORTPipelineForInpainting.
# ort_pipelines_mapping de optimum-onnx 0.1.0 (el pin del proyecto):
#   inpainting = {stable-diffusion, stable-diffusion-xl, stable-diffusion-3}
# Quedan FUERA (tienen otras rutas pero no inpainting en optimum-onnx):
#   - latent-consistency: tiene img2img pero NO inpainting.
#   - flux, sana: texto a imagen solamente (igual que en img2img).
# Chequear que exista la clase no sirve como test — misma lección que img2img
# (sana ejecuta aunque ORTSanaPipeline no exista): validate_inpaint_table()
# cruza contra el mapping real de la versión instalada.
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


def inpaint_class_for(declared_class_name: str) -> str:
    target = INPAINT_CLASS_NAMES.get(declared_class_name)
    if target is not None:
        return target
    if declared_class_name in NO_INPAINT_CLASS_NAMES:
        raise InpaintUnsupportedError(
            f"El modelo ({declared_class_name}) genera imágenes pero optimum-onnx "
            "no tiene pipeline de inpainting para esa arquitectura. Usa un modelo "
            "Stable Diffusion, SDXL o SD3."
        )
    raise InpaintUnsupportedError(
        f"Clase de pipeline desconocida para inpainting: {declared_class_name!r}"
    )


def supports_inpaint(declared_class_name: str) -> bool:
    return declared_class_name in INPAINT_CLASS_NAMES


def is_dedicated_inpaint_class(declared_class_name: str) -> bool:
    """Checkpoint de inpainting DEDICADO (unet 9ch): solo sabe editar con
    máscara — no puede generar texto a imagen ni imagen a imagen."""
    return "Inpaint" in declared_class_name and declared_class_name in INPAINT_CLASS_NAMES


def load_inpaint_class(declared_class_name: str) -> Any:
    import optimum.onnxruntime as ort_module

    return getattr(ort_module, inpaint_class_for(declared_class_name))


def validate_inpaint_table() -> None:
    import optimum.onnxruntime as ort_module
    from optimum.onnxruntime import ORTPipelineForInpainting

    exposed = {cls.__name__ for cls in ORTPipelineForInpainting.ort_pipelines_mapping.values()}
    for target in set(INPAINT_CLASS_NAMES.values()):
        if target not in exposed:
            raise AssertionError(
                f"{target} no está en ORTPipelineForInpainting.ort_pipelines_mapping "
                "de la versión instalada de optimum — la tabla quedó vieja tras un upgrade"
            )
        if not hasattr(ort_module, target):
            raise AssertionError(
                f"{target} está en el mapping pero optimum.onnxruntime no lo expone"
            )
