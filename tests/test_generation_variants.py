from __future__ import annotations

import pytest

from app.services.generation_variants import (
    available_precisions,
    canonical_weight_name,
    select_for_precision,
)
from app.services.hf_client import HfFile

MB = 1024 * 1024

# Listado real de stable-diffusion-v1-5/stable-diffusion-v1-5 (API de HF,
# 2026-07-28). Los .ckpt de la raiz se omiten: CONVERSION_SKIP_SUFFIXES ya los
# descarta y no aportan al caso que este test fija.
SD15_FILES = [
    HfFile(path="model_index.json", size=1024),
    HfFile(path="tokenizer/vocab.json", size=1 * MB),
    HfFile(path="unet/config.json", size=2048),
    HfFile(path="unet/diffusion_pytorch_model.safetensors", size=3278 * MB),
    HfFile(path="unet/diffusion_pytorch_model.non_ema.safetensors", size=3278 * MB),
    HfFile(path="unet/diffusion_pytorch_model.fp16.safetensors", size=1639 * MB),
    HfFile(path="unet/diffusion_pytorch_model.bin", size=3278 * MB),
    HfFile(path="safety_checker/model.safetensors", size=1159 * MB),
    HfFile(path="safety_checker/model.fp16.safetensors", size=579 * MB),
    HfFile(path="text_encoder/model.safetensors", size=469 * MB),
    HfFile(path="text_encoder/model.fp16.safetensors", size=234 * MB),
    HfFile(path="vae/diffusion_pytorch_model.safetensors", size=319 * MB),
    HfFile(path="vae/diffusion_pytorch_model.fp16.safetensors", size=159 * MB),
]

SD15_DECLARED = [
    "feature_extractor",
    "safety_checker",
    "scheduler",
    "text_encoder",
    "tokenizer",
    "unet",
    "vae",
]


def _total_mb(files: list[HfFile]) -> int:
    return sum(f.size for f in files) // MB


def test_available_precisions_detects_both_when_repo_publishes_both():
    assert available_precisions(SD15_FILES) == ("fp16", "fp32")


def test_available_precisions_returns_only_fp32_when_no_fp16_variant():
    # Caso real: Tongyi-MAI/Z-Image-Turbo publica solo la variante plana.
    files = [
        HfFile(path="model_index.json", size=1024),
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=500 * MB),
    ]
    assert available_precisions(files) == ("fp32",)


def test_available_precisions_ignores_non_ema_when_deciding():
    # .non_ema. es un checkpoint de entrenamiento: no es una precision elegible.
    files = [
        HfFile(path="model_index.json", size=1024),
        HfFile(path="unet/diffusion_pytorch_model.non_ema.safetensors", size=500 * MB),
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=500 * MB),
    ]
    assert available_precisions(files) == ("fp32",)


def test_select_fp16_picks_only_the_fp16_weight_per_component():
    selected = select_for_precision(SD15_FILES, SD15_DECLARED, "fp16")
    paths = {f.path for f in selected}
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in paths
    assert "unet/diffusion_pytorch_model.safetensors" not in paths
    assert "safety_checker/model.fp16.safetensors" in paths
    assert "text_encoder/model.fp16.safetensors" in paths
    assert "vae/diffusion_pytorch_model.fp16.safetensors" in paths


def test_select_fp32_picks_only_the_plain_weight_per_component():
    selected = select_for_precision(SD15_FILES, SD15_DECLARED, "fp32")
    paths = {f.path for f in selected}
    assert "unet/diffusion_pytorch_model.safetensors" in paths
    assert "unet/diffusion_pytorch_model.fp16.safetensors" not in paths


def test_non_ema_never_selected():
    for precision in ("fp16", "fp32"):
        selected = select_for_precision(SD15_FILES, SD15_DECLARED, precision)
        assert not any(".non_ema." in f.path for f in selected)


def test_bin_dropped_when_safetensors_sibling_exists():
    selected = select_for_precision(SD15_FILES, SD15_DECLARED, "fp32")
    assert not any(f.path.endswith(".bin") for f in selected)


def test_bin_kept_when_it_is_the_only_weight_in_the_component():
    files = [
        HfFile(path="model_index.json", size=1024),
        HfFile(path="unet/diffusion_pytorch_model.bin", size=500 * MB),
    ]
    selected = select_for_precision(files, ["unet"], "fp32")
    assert [f.path for f in selected] == ["unet/diffusion_pytorch_model.bin"]


def test_model_index_never_in_selection():
    # Se descarga aparte y antes que el resto, para conocer `declared`.
    selected = select_for_precision(SD15_FILES, SD15_DECLARED, "fp16")
    assert not any(f.path == "model_index.json" for f in selected)


def test_undeclared_component_dirs_excluded():
    files = SD15_FILES + [
        HfFile(path="controlnet/diffusion_pytorch_model.safetensors", size=999 * MB)
    ]
    selected = select_for_precision(files, SD15_DECLARED, "fp32")
    assert not any(f.path.startswith("controlnet/") for f in selected)


@pytest.mark.parametrize(
    ("precision", "expected_mb"),
    [("fp16", 2612), ("fp32", 5226)],
)
def test_selection_totals_match_the_measured_numbers(precision, expected_mb):
    # Totales de ESTE fixture, que es una reproduccion reducida: incluye los 10
    # archivos grandes del repo real pero omite varios sub-MB (configs de
    # componente, merges.txt, special_tokens_map...). Sobre el repo completo de
    # 19 archivos los numeros del spec son 2611 y 5232 MB; la diferencia son
    # justamente esos archivos chicos.
    #
    # Lo que importa y lo que este test fija: una sola variante de pesos por
    # componente. Antes la seleccion pesaba 11121 MB y reventaba el cap de 8192.
    total = _total_mb(select_for_precision(SD15_FILES, SD15_DECLARED, precision))
    assert total == expected_mb


def test_regression_selection_is_far_below_the_old_cap():
    # El bug original: 11121 MB > 8192 MB. Este test falla si vuelve a bajar
    # multiples variantes del mismo componente.
    assert _total_mb(select_for_precision(SD15_FILES, SD15_DECLARED, "fp16")) < 3072


def test_conversion_skip_suffixes_excluded():
    files = SD15_FILES + [
        HfFile(path="v1-5-pruned.ckpt", size=7700 * MB),
        HfFile(path="unet/model.onnx", size=100 * MB),
    ]
    selected = select_for_precision(files, SD15_DECLARED, "fp16")
    assert not any(f.path.endswith((".ckpt", ".onnx")) for f in selected)


def test_canonical_weight_name_strips_the_fp16_marker():
    assert (
        canonical_weight_name("unet/diffusion_pytorch_model.fp16.safetensors")
        == "unet/diffusion_pytorch_model.safetensors"
    )


def test_canonical_weight_name_leaves_plain_names_untouched():
    for path in (
        "unet/diffusion_pytorch_model.safetensors",
        "tokenizer/vocab.json",
        "model_index.json",
    ):
        assert canonical_weight_name(path) == path
