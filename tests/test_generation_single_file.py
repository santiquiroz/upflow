from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.generation_single_file import (
    _ARCHITECTURE_TO_ORT,
    classify_checkpoint,
    supported_architecture,
    validate_architecture_table,
)

FIXTURES_PATH = Path(__file__).parent / "assets" / "single_file_fixtures.json"


def _fixtures() -> dict:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


def _header(name: str) -> dict:
    """Reconstruye un header de safetensors desde el fixture.

    Los fixtures guardan solo las claves que el detector de diffusers consulta
    mas una representativa por rol -- capturado de repos reales el 2026-07-29.
    """
    entry = _fixtures()[name]
    header: dict[str, dict] = {}
    for group in ("watched_keys", "role_sample_keys", "lora_sample_keys"):
        for key, shape in entry[group].items():
            header[key] = {"shape": shape, "dtype": "F16"}
    return header


# ---------------------------------------------------------------------------
# Tabla de arquitecturas
# ---------------------------------------------------------------------------


def test_architecture_table_matches_what_optimum_supports():
    # El test que avisa si un upgrade de optimum renombra una clave. Sin esto el
    # mapeo degradaria en silencio a "arquitectura no soportada".
    validate_architecture_table()


def test_supported_architecture_maps_every_table_entry():
    for detected, expected in _ARCHITECTURE_TO_ORT.items():
        assert supported_architecture(detected) == expected


@pytest.mark.parametrize(
    "detected",
    [
        "flux-2-dev",       # Flux2Pipeline existe en diffusers, no en optimum
        "z-image-turbo",    # idem ZImagePipeline
        "xl_refiner",       # img2img, no text-to-image
        "inpainting",       # inpaint, no text-to-image
        "animatediff_v1",   # video
        "invented-arch",
    ],
)
def test_unsupported_architectures_return_none(detected):
    assert supported_architecture(detected) is None


def test_supported_architecture_handles_none():
    assert supported_architecture(None) is None


# ---------------------------------------------------------------------------
# classify_checkpoint contra checkpoints reales
# ---------------------------------------------------------------------------


def test_real_sdxl_checkpoint_is_installable():
    # LyliaEngine/Pony_Diffusion_V6_XL, 183k descargas: SDXL fine-tuneado, el
    # caso que motiva toda la feature.
    verdict = classify_checkpoint(_header("pony_sdxl"))
    assert verdict.installable is True
    assert verdict.architecture == "xl_base"
    assert verdict.ort_model_type == "stable-diffusion-xl"


def test_standalone_vae_is_not_installable():
    # sdxl_vae.safetensors, al lado del checkpoint real en el mismo repo.
    verdict = classify_checkpoint(_header("vae_only"))
    assert verdict.installable is False
    assert verdict.reason


def test_standalone_vae_rejected_for_missing_roles_not_for_being_a_lora():
    verdict = classify_checkpoint(_header("vae_only"))
    assert "falta" in verdict.reason.lower()
    assert "lora" not in verdict.reason.lower()


@pytest.mark.parametrize("name", ["zimage_lora", "ipadapter_lora"])
def test_loras_are_not_installable(name):
    verdict = classify_checkpoint(_header(name))
    assert verdict.installable is False
    assert "lora" in verdict.reason.lower() or "adapter" in verdict.reason.lower()


def test_flux2_backbone_only_file_is_not_installable():
    # FLUX.2 se publica componente-por-repo: este archivo trae solo el backbone,
    # sin text encoder ni VAE.
    verdict = classify_checkpoint(_header("flux2_backbone"))
    assert verdict.installable is False


# ---------------------------------------------------------------------------
# Regresion del bug de v0.15.1: ausencia de senal leida como veredicto positivo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["vae_only", "zimage_lora", "ipadapter_lora"])
def test_files_the_detector_calls_v1_are_still_rejected(name):
    """El detector de diffusers devuelve "v1" para estos tres, con CERO claves
    consultadas -- es su default historico, no una identificacion. Si el gate
    fuera el detector, se instalarian como SD1.5 y el badge volveria a mentir,
    que es exactamente el bug de v0.15.1."""
    assert _fixtures()[name]["detector_says"] == "v1"
    assert _fixtures()[name]["watched_keys"] == {}
    assert classify_checkpoint(_header(name)).installable is False


def test_lora_check_runs_before_the_role_check():
    # El LoRA de Z-Image usa claves con prefijo `diffusion_model.`, que satisface
    # el rol de backbone. Si el orden se invirtiera, el motivo hablaria de roles
    # faltantes en vez de decir que es un LoRA.
    verdict = classify_checkpoint(_header("zimage_lora"))
    assert "lora" in verdict.reason.lower() or "adapter" in verdict.reason.lower()


def test_vae_role_requires_first_stage_model_prefix():
    # Un decoder./encoder. en la raiz significa que el archivo ES un VAE. Si se
    # aceptara como rol VAE, sdxl_vae.safetensors pasaria el gate.
    header = {
        "model.diffusion_model.input_blocks.0.0.weight": {"shape": [320, 4, 3, 3]},
        "conditioner.embedders.0.transformer.text_model.x.weight": {"shape": [768, 768]},
        "decoder.conv_in.bias": {"shape": [512]},
        "encoder.conv_in.bias": {"shape": [128]},
    }
    assert classify_checkpoint(header).installable is False


def test_complete_checkpoint_needs_all_three_roles():
    backbone = {"model.diffusion_model.input_blocks.0.0.weight": {"shape": [320, 4, 3, 3]}}
    text = {"cond_stage_model.transformer.x.weight": {"shape": [768, 768]}}
    vae = {"first_stage_model.decoder.conv_in.bias": {"shape": [512]}}

    assert classify_checkpoint(backbone | text).installable is False
    assert classify_checkpoint(backbone | vae).installable is False
    assert classify_checkpoint(text | vae).installable is False


def test_empty_header_is_not_installable():
    assert classify_checkpoint({}).installable is False
    assert classify_checkpoint({"__metadata__": {"format": "pt"}}).installable is False


def test_verdict_reason_is_always_populated():
    for name in _fixtures():
        assert classify_checkpoint(_header(name)).reason
