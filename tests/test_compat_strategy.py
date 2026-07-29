from __future__ import annotations

import pytest

from app.services.compat_strategy import (
    GenerationCompatStrategy,
    UpscalerCompatStrategy,
    strategy_for,
)
from app.services.generation_compat import classify as classify_generation

SD15 = (
    "model_index.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/model.fp16.safetensors",
)
SDXL_ONNX = (
    "model_index.json",
    "unet/model.onnx",
    "vae_decoder/model.onnx",
    "text_encoder/model.onnx",
)


# ---------------------------------------------------------------------------
# La de generacion envuelve lo que ya existia, sin cambiarlo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filenames,gated",
    [
        (SD15, False),
        (SDXL_ONNX, False),
        (SD15, "auto"),
        (("config.json",), False),
        (("checkpoint.safetensors",), False),
    ],
)
def test_the_generation_strategy_delegates_without_changing_the_verdict(filenames, gated):
    # Los tests de generation_compat son el contrato de esa clasificacion: la
    # estrategia solo la envuelve, asi que tiene que dar exactamente lo mismo.
    assert GenerationCompatStrategy().classify(filenames, gated) == classify_generation(
        filenames, gated
    )


def test_the_generation_strategy_offers_the_precisions_the_repo_publishes():
    options = GenerationCompatStrategy().install_options(SD15)
    assert "fp16" in options.precisions


def test_the_generation_strategy_offers_the_root_checkpoints():
    options = GenerationCompatStrategy().install_options(
        ("pony.safetensors", "vae/model.safetensors", "README.md")
    )
    # Solo los de la RAIZ: un .safetensors dentro de una carpeta de componente es
    # parte de un pipeline, no un checkpoint suelto.
    assert options.checkpoints == ("pony.safetensors",)


# ---------------------------------------------------------------------------
# La de upscalers, que hasta ahora vivia implicita en el instalador
# ---------------------------------------------------------------------------


def test_an_onnx_upscaler_installs_direct():
    verdict, reason = UpscalerCompatStrategy().classify(("RealESRGAN_x4.onnx",), False)
    assert verdict == "ready_onnx"
    assert reason.key


def test_torch_weights_need_conversion():
    for name in ("model.safetensors", "model.pth"):
        verdict, reason = UpscalerCompatStrategy().classify((name,), False)
        assert verdict == "needs_conversion", name
        assert reason.key == "compat.upscaler.needsConversion"


def test_onnx_wins_over_torch_weights():
    # Mismo orden de preferencia que pick_weight_file, que es lo que el
    # instalador realmente hace: si hay .onnx no convierte nada.
    verdict, _reason = UpscalerCompatStrategy().classify(
        ("model.pth", "model.onnx", "model.safetensors"), False
    )
    assert verdict == "ready_onnx"


def test_a_repo_without_weights_is_incompatible():
    verdict, reason = UpscalerCompatStrategy().classify(
        ("README.md", "config.json", "preview.png"), False
    )
    assert verdict == "incompatible"
    assert reason.key


def test_an_empty_repo_is_incompatible():
    verdict, _reason = UpscalerCompatStrategy().classify((), False)
    assert verdict == "incompatible"


@pytest.mark.parametrize("gated", [True, "auto", "manual"])
def test_a_gated_repo_is_gated_no_matter_what_it_contains(gated):
    # Sin token no se puede leer nada mas del repo: cualquier otro veredicto
    # seria una conjetura.
    verdict, reason = UpscalerCompatStrategy().classify(("model.onnx",), gated)
    assert verdict == "gated"
    assert reason.key == "compat.gated"


def test_a_not_gated_repo_is_classified_by_its_content():
    for gated in (False, None):
        verdict, _reason = UpscalerCompatStrategy().classify(("model.onnx",), gated)
        assert verdict == "ready_onnx", gated


def test_upscalers_offer_no_install_options():
    # No es un caso degradado: el instalador elige el archivo de pesos solo con
    # pick_weight_file, asi que no hay nada que preguntar.
    options = UpscalerCompatStrategy().install_options(("model.onnx", "model.pth"))
    assert options.precisions == ()
    assert options.checkpoints == ()


def test_the_suffix_check_is_case_insensitive():
    verdict, _reason = UpscalerCompatStrategy().classify(("MODEL.ONNX",), False)
    assert verdict == "ready_onnx"


# ---------------------------------------------------------------------------
# Resolucion por dominio
# ---------------------------------------------------------------------------


def test_generate_resolves_to_the_generation_strategy():
    assert isinstance(strategy_for("generate"), GenerationCompatStrategy)


@pytest.mark.parametrize("domain", ["video", "image"])
def test_video_and_image_share_the_upscaler_strategy(domain):
    # Es el mismo motor: RealESRGAN sirve a los dos.
    assert isinstance(strategy_for(domain), UpscalerCompatStrategy)


def test_an_unknown_domain_is_rejected_loudly():
    # Devolver una estrategia por defecto clasificaria un dominio nuevo con las
    # reglas del equivocado, en silencio.
    with pytest.raises(ValueError):
        strategy_for("audio")


def test_every_strategy_satisfies_the_protocol():
    for strategy in (GenerationCompatStrategy(), UpscalerCompatStrategy()):
        assert callable(strategy.classify)
        assert callable(strategy.install_options)
