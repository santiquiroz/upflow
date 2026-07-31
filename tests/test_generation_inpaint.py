from __future__ import annotations

import pytest

from app.services.generation_inpaint import (
    INPAINT_CLASS_NAMES,
    InpaintUnsupportedError,
    inpaint_class_for,
    supports_inpaint,
    validate_inpaint_table,
)

# ---------------------------------------------------------------------------
# Espejo de test_generation_img2img.py: la tabla se valida contra el optimum
# REAL instalado (cobertura medida 2026-07-31: SD, SDXL, SD3; fuera: LCM que
# sí tiene img2img, y flux/sana). Sin fakes.
# ---------------------------------------------------------------------------


def test_the_table_matches_what_optimum_actually_exposes() -> None:
    validate_inpaint_table()


def test_the_cover_is_narrower_than_img2img() -> None:
    from optimum.onnxruntime import ORTPipelineForImage2Image, ORTPipelineForInpainting

    inpaint = set(ORTPipelineForInpainting.ort_pipelines_mapping)
    img2img = set(ORTPipelineForImage2Image.ort_pipelines_mapping)
    assert inpaint < img2img
    assert "latent-consistency" in img2img - inpaint


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("StableDiffusionPipeline", "ORTStableDiffusionInpaintPipeline"),
        ("ORTStableDiffusionPipeline", "ORTStableDiffusionInpaintPipeline"),
        ("StableDiffusionXLPipeline", "ORTStableDiffusionXLInpaintPipeline"),
        ("ORTStableDiffusionXLPipeline", "ORTStableDiffusionXLInpaintPipeline"),
        ("StableDiffusion3Pipeline", "ORTStableDiffusion3InpaintPipeline"),
    ],
)
def test_known_classes_resolve_to_their_inpaint_pipeline(declared: str, expected: str) -> None:
    assert inpaint_class_for(declared) == expected


def test_a_no_inpaint_architecture_gets_its_own_reason() -> None:
    with pytest.raises(InpaintUnsupportedError, match="no tiene pipeline de inpainting"):
        inpaint_class_for("LatentConsistencyModelPipeline")
    with pytest.raises(InpaintUnsupportedError, match="no tiene pipeline de inpainting"):
        inpaint_class_for("FluxPipeline")


def test_an_unknown_class_gets_the_unknown_reason() -> None:
    with pytest.raises(InpaintUnsupportedError, match="desconocida"):
        inpaint_class_for("KandinskyV22Pipeline")


def test_supports_inpaint_never_raises() -> None:
    assert supports_inpaint("StableDiffusionXLPipeline") is True
    assert supports_inpaint("LatentConsistencyModelPipeline") is False
    assert supports_inpaint("QuienSabeQuePipeline") is False


def test_no_target_class_is_missing_the_inpaint_marker() -> None:
    for target in INPAINT_CLASS_NAMES.values():
        assert "Inpaint" in target


# --- clamp de strength para checkpoints 4ch (medido 2026-07-31) -------------


class _FakeUnet:
    class config:  # noqa: D106 - shape mínima del config de diffusers
        in_channels = 4


class _FakePipeline:
    unet = _FakeUnet()


def test_inpaint_strength_is_forced_to_one_for_4ch_unet() -> None:
    from app.services.engines.generation_onnx import _inpaint_strength

    assert _inpaint_strength(_FakePipeline(), 0.85) == 1.0
    assert _inpaint_strength(_FakePipeline(), 1.0) == 1.0


def test_inpaint_strength_is_respected_for_real_inpaint_checkpoints() -> None:
    from app.services.engines.generation_onnx import _inpaint_strength

    pipeline = _FakePipeline()
    pipeline.unet.config.in_channels = 9
    try:
        assert _inpaint_strength(pipeline, 0.85) == 0.85
    finally:
        pipeline.unet.config.in_channels = 4


def test_dedicated_inpaint_checkpoints_are_accepted() -> None:
    # Un checkpoint de inpainting real (unet 9ch) declara la clase de inpainting
    # en su model_index; si no la aceptamos, el modelo que MEJOR borra sería el
    # único rechazado.
    assert inpaint_class_for("StableDiffusionXLInpaintPipeline") == "ORTStableDiffusionXLInpaintPipeline"
    assert inpaint_class_for("StableDiffusionInpaintPipeline") == "ORTStableDiffusionInpaintPipeline"
    assert supports_inpaint("StableDiffusionXLInpaintPipeline") is True
