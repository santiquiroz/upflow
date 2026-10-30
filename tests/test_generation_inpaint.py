from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from app.services.generation_pipeline_modes import (
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


def test_dedicated_inpaint_classes_are_detected() -> None:
    from app.services.generation_pipeline_modes import is_dedicated_inpaint_class

    assert is_dedicated_inpaint_class("StableDiffusionXLInpaintPipeline") is True
    assert is_dedicated_inpaint_class("ORTStableDiffusionInpaintPipeline") is True
    # Un checkpoint base soporta inpaint por RUTEO pero no es dedicado.
    assert is_dedicated_inpaint_class("StableDiffusionXLPipeline") is False
    assert is_dedicated_inpaint_class("ClaseInventada") is False


@pytest.fixture
def auto_edit_context(tmp_path, monkeypatch):
    from app.config import Settings
    from app.services.engines.generation_onnx import GenerationEngine, GenerationRequest
    from app.services import generation_soft_inpaint

    base, mask, protect = [tmp_path / name for name in ("base.png", "mask.png", "protect.png")]
    Image.new("RGB", (64, 64), (180, 180, 180)).save(base)
    Image.new("L", (64, 64), 255).save(mask)
    protected = np.zeros((64, 64), dtype=np.uint8)
    protected[20:40, 20:40] = 255
    Image.fromarray(protected).save(protect)
    request = GenerationRequest(
        prompt="venue", negative_prompt="blur", steps=2, guidance=1.0,
        width=64, height=64, seed=None, init_image_path=base,
        mask_image_path=mask, protect_mask_path=protect, erase_strategy="auto",
        mask_dilate_px=8, mask_feather_px=8,
    )
    lama = Mock()
    lama.available.return_value = True
    engine = GenerationEngine(Settings(RUNTIME_DIR=str(tmp_path), _env_file=None), Mock(), lama)
    monkeypatch.setattr(generation_soft_inpaint, "build_soft_inpaint_callback", lambda *args: None)
    return engine, request, tmp_path / "result.png"


@pytest.mark.parametrize("channels,strengths", [(9, [0.5, 1.0]), (4, [1.0, 1.0])])
def test_auto_edit_injects_providers_and_preserves_strength_clamping(
    auto_edit_context, monkeypatch, channels, strengths
) -> None:
    from app.services import erase_auto, inpaint_pipeline
    from app.services.engines.generation_onnx import ERASE_AUTO_NEGATIVE_PROMPT

    engine, request, output = auto_edit_context
    pipeline = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (64, 64))]))
    pipeline.unet.config.in_channels = channels
    recorded_settings = []
    choices = [{"engine": "sdxl", "scores": {"sdxl": {"score": 0.0}}}]

    def masked_edit(base, mask, model, settings):
        recorded_settings.append(settings)
        return model(base, mask, *base.size)

    def orchestrate(base, mask, protect, lama_fill, sdxl_edit):
        assert lama_fill == engine.lama_eraser.fill_group
        assert protect[30, 30] and not protect[0, 0]
        assert protect.dtype == bool
        sdxl_edit(base, mask, 0.5)
        return sdxl_edit(base, mask, 1.0), choices

    monkeypatch.setattr(inpaint_pipeline, "run_masked_edit", masked_edit)
    monkeypatch.setattr(erase_auto, "auto_erase", orchestrate)
    result = engine._run_masked_edit_blocking(pipeline, request, output, Mock(), threading.Lock())

    assert result == output and output.is_file()
    assert [call.kwargs["strength"] for call in pipeline.call_args_list] == strengths
    assert all(s.dilate_px == s.feather_px == 0 for s in recorded_settings)
    assert pipeline.call_args.kwargs["negative_prompt"] == f"blur, {ERASE_AUTO_NEGATIVE_PROMPT}"
    assert request.erase_auto == choices
    engine.gpu_coordinator.acquire.assert_not_called()


@pytest.mark.parametrize("availability", [None, False])
def test_auto_edit_falls_back_when_lama_is_missing_or_unavailable(
    auto_edit_context, monkeypatch, caplog, availability
) -> None:
    from app.services import erase_auto, inpaint_pipeline

    engine, request, output = auto_edit_context
    if availability is None:
        engine.lama_eraser = None
    else:
        engine.lama_eraser.available.return_value = availability
    masked_edit = Mock(side_effect=lambda base, *args: base)
    auto = Mock(side_effect=AssertionError("LaMa no está disponible"))
    monkeypatch.setattr(inpaint_pipeline, "run_masked_edit", masked_edit)
    monkeypatch.setattr(erase_auto, "auto_erase", auto)

    engine._run_masked_edit_blocking(Mock(), request, output, Mock(), threading.Lock())

    masked_edit.assert_called_once()
    assert masked_edit.call_args.args[3].dilate_px == 8
    assert masked_edit.call_args.args[3].feather_px == 8
    auto.assert_not_called()
    assert "LaMa no disponible" in caplog.text
    assert request.erase_auto is None
    engine.gpu_coordinator.acquire.assert_not_called()
