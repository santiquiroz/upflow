from __future__ import annotations

from app.services.generation_compat import classify

SD15 = (
    "model_index.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "text_encoder/model.safetensors",
)

SDXL_ONNX = (
    "model_index.json",
    "unet/model.onnx",
    "unet/diffusion_pytorch_model.safetensors",
    "vae_decoder/model.onnx",
    "vae_decoder/diffusion_pytorch_model.safetensors",
)


def test_repo_without_model_index_is_incompatible():
    # Caso real: wikeeyang/Flux2-Klein-9B-True-V2 (15 archivos, sin model_index).
    verdict, reason = classify(("config.json", "weights.safetensors"), False)
    assert verdict == "incompatible"
    assert "model_index.json" in reason


def test_torch_only_repo_needs_conversion():
    verdict, reason = classify(SD15, False)
    assert verdict == "needs_conversion"
    assert reason


def test_repo_with_onnx_for_every_torch_component_is_ready():
    verdict, _ = classify(SDXL_ONNX, False)
    assert verdict == "ready_onnx"


def test_repo_with_partial_onnx_needs_conversion():
    # Caso real documentado en el spike: stabilityai/sdxl-turbo publica ONNX
    # para unet pero solo pesos torch para vae. Bajar solo el ONNX deja un
    # pipeline parcial.
    files = SDXL_ONNX + ("vae/diffusion_pytorch_model.safetensors",)
    verdict, _ = classify(files, False)
    assert verdict == "needs_conversion"


def test_gated_repo_wins_over_everything_else():
    # black-forest-labs/FLUX.1-dev y stabilityai/stable-diffusion-3.5-medium
    # devuelven gated="auto" en la metadata publica.
    for gated in ("auto", "manual", True):
        verdict, reason = classify(SD15, gated)
        assert verdict == "gated"
        assert "token" in reason.lower()


def test_gated_wins_even_when_model_index_is_missing():
    # Sin token no se puede saber nada mas del repo, asi que gated gana.
    verdict, _ = classify(("config.json",), "auto")
    assert verdict == "gated"


def test_gated_false_and_none_are_not_gated():
    for gated in (False, None, ""):
        verdict, _ = classify(SD15, gated)
        assert verdict != "gated"
