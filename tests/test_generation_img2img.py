from __future__ import annotations

import pytest

from app.services.generation_img2img import (
    IMG2IMG_CLASS_NAMES,
    TEXT_ONLY_CLASS_NAMES,
    Img2ImgUnsupportedError,
    img2img_class_for,
    load_img2img_class,
    supports_img2img,
    validate_img2img_table,
)


def test_the_table_matches_what_optimum_actually_exposes():
    # Avisa si un upgrade de optimum renombra una clase. Sin esto el mapeo
    # degradaria en silencio a "arquitectura no soportada" para todo.
    validate_img2img_table()


def test_the_cover_is_narrower_than_text_to_image():
    """La asimetria es el hecho central de esta capacidad.

    Medido: texto a imagen soporta seis model types, imagen a imagen cuatro. Falta
    flux y falta sana. Un modelo instalado y usable para texto a imagen puede no
    servir aca, asi que el picker tiene que filtrar.
    """
    from optimum.onnxruntime import ORTPipelineForImage2Image, ORTPipelineForText2Image

    text = set(ORTPipelineForText2Image.ort_pipelines_mapping)
    image = set(ORTPipelineForImage2Image.ort_pipelines_mapping)

    assert image < text
    assert {"flux", "sana"} <= text - image


def test_every_target_class_is_importable_from_optimum():
    for declared in IMG2IMG_CLASS_NAMES:
        assert load_img2img_class(declared) is not None


@pytest.mark.parametrize(
    "declared,expected",
    [
        ("OnnxStableDiffusionPipeline", "ORTStableDiffusionImg2ImgPipeline"),
        ("StableDiffusionXLPipeline", "ORTStableDiffusionXLImg2ImgPipeline"),
        ("ORTStableDiffusionXLPipeline", "ORTStableDiffusionXLImg2ImgPipeline"),
        ("StableDiffusion3Pipeline", "ORTStableDiffusion3Img2ImgPipeline"),
        ("LatentConsistencyModelPipeline", "ORTLatentConsistencyModelImg2ImgPipeline"),
    ],
)
def test_a_declared_class_resolves_to_its_img2img_class(declared: str, expected: str):
    assert img2img_class_for(declared) == expected


def test_the_already_ort_names_resolve_too():
    # Un repo convertido por esta app declara el nombre ORT en su model_index.json,
    # y un repo de HF declara el nombre de diffusers. Los dos tienen que entrar.
    assert img2img_class_for("ORTStableDiffusion3Pipeline") == (
        "ORTStableDiffusion3Img2ImgPipeline"
    )


@pytest.mark.parametrize("declared", sorted(TEXT_ONLY_CLASS_NAMES))
def test_a_text_only_architecture_is_rejected_saying_why(declared: str):
    # El motivo importa: "no soportada" a secas haria pensar que el modelo esta
    # roto, cuando en realidad funciona perfecto para texto a imagen.
    with pytest.raises(Img2ImgUnsupportedError) as exc_info:
        img2img_class_for(declared)

    message = str(exc_info.value)
    assert "texto a imagen" in message
    assert declared in message


def test_an_unknown_class_is_rejected_as_unknown():
    with pytest.raises(Img2ImgUnsupportedError) as exc_info:
        img2img_class_for("PipelineInventado")
    assert "desconocida" in str(exc_info.value)


def test_flux_and_sana_are_not_in_the_mapping():
    # Si alguna vez entran, tiene que ser porque optimum las soporta, y el test de
    # validacion de la tabla es el que lo verifica.
    for declared in TEXT_ONLY_CLASS_NAMES:
        assert declared not in IMG2IMG_CLASS_NAMES


def test_supports_img2img_answers_without_raising():
    assert supports_img2img("StableDiffusionXLPipeline") is True
    assert supports_img2img("FluxPipeline") is False
    assert supports_img2img("PipelineInventado") is False


def test_no_target_class_is_a_text_to_image_class():
    # Un error de copia y pega que apunte a la clase de texto a imagen cargaria el
    # pipeline equivocado y la imagen de entrada se ignoraria en silencio.
    for target in IMG2IMG_CLASS_NAMES.values():
        assert "Img2Img" in target
