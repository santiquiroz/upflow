from __future__ import annotations

from typing import Any

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
# aunque ORTSanaPipeline no exista.

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


def img2img_class_for(declared_class_name: str) -> str:
    """Clase ORT de imagen a imagen para el pipeline instalado.

    Levanta Img2ImgUnsupportedError con un motivo distinto segun si el modelo
    simplemente no se conoce o si es uno que optimum SI puede correr para texto a
    imagen pero no para imagen a imagen.
    """
    resolved = IMG2IMG_CLASS_NAMES.get(declared_class_name)
    if resolved is not None:
        return resolved

    if declared_class_name in TEXT_ONLY_CLASS_NAMES:
        raise Img2ImgUnsupportedError(
            f"{declared_class_name} sirve para texto a imagen pero optimum-onnx no "
            "tiene pipeline de imagen a imagen para esa arquitectura. Usa un modelo "
            "Stable Diffusion, SDXL, SD3 o Latent Consistency."
        )

    raise Img2ImgUnsupportedError(
        f"Clase de pipeline desconocida para imagen a imagen: {declared_class_name!r}."
    )


def supports_img2img(declared_class_name: str) -> bool:
    return declared_class_name in IMG2IMG_CLASS_NAMES


def load_img2img_class(declared_class_name: str) -> Any:
    import optimum.onnxruntime as ort_module

    return getattr(ort_module, img2img_class_for(declared_class_name))


def validate_img2img_table() -> None:
    """Falla ruidosamente si un upgrade de optimum rompe el mapeo.

    Sin esto, un rename de clase degradaria en silencio a "arquitectura no
    soportada" para todos los modelos.
    """
    import optimum.onnxruntime as ort_module
    from optimum.onnxruntime import ORTPipelineForImage2Image

    supported_classes = {
        cls.__name__ for cls in ORTPipelineForImage2Image.ort_pipelines_mapping.values()
    }
    targets = set(IMG2IMG_CLASS_NAMES.values())

    unknown = sorted(targets - supported_classes)
    if unknown:
        raise RuntimeError(
            "optimum-onnx ya no expone estas clases de imagen a imagen: "
            + ", ".join(unknown)
            + f". Soportadas hoy: {sorted(supported_classes)}. Revisar IMG2IMG_CLASS_NAMES."
        )

    missing_attr = sorted(name for name in targets if not hasattr(ort_module, name))
    if missing_attr:
        raise RuntimeError(
            "Estas clases estan en el mapping de optimum pero no se pueden importar: "
            + ", ".join(missing_attr)
        )
