from __future__ import annotations

from app.services.asr_files import (
    is_required_asr_file,
    missing_required_onnx,
    select_asr_files,
)

# Capturado de onnx-community/whisper-tiny.en el 2026-07-29 (los .onnx completos;
# el repo real publica 28 archivos .onnx contando todas las variantes).
REPO_FILES = (
    ".gitattributes",
    "README.md",
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
    "quantize_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "onnx/encoder_model.onnx",
    "onnx/encoder_model_fp16.onnx",
    "onnx/encoder_model_int8.onnx",
    "onnx/encoder_model_quantized.onnx",
    "onnx/decoder_model.onnx",
    "onnx/decoder_model_fp16.onnx",
    "onnx/decoder_model_int8.onnx",
    "onnx/decoder_model_merged.onnx",
    "onnx/decoder_model_merged_fp16.onnx",
    "onnx/decoder_with_past_model.onnx",
    "onnx/decoder_with_past_model_fp16.onnx",
    "onnx/decoder_with_past_model_int8.onnx",
)

# El subconjunto que se MIDIO cargando de verdad, menos los archivos que el
# selector no necesita (.gitattributes y README.md no los pide nadie).
MEASURED_WORKING_SUBSET = frozenset(
    {
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "normalizer.json",
        "preprocessor_config.json",
        "quantize_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "onnx/encoder_model.onnx",
        "onnx/decoder_model.onnx",
        "onnx/decoder_with_past_model.onnx",
    }
)


def test_the_selection_matches_the_subset_that_was_measured_to_load():
    """Este es el test que importa.

    Se verifico a mano que con exactamente estos archivos el modelo carga y
    transcribe. Si la seleccion cambia, o se rompe la instalacion (falta algo) o se
    infla (sobra algo), y eso tiene que fallar aca.
    """
    assert set(select_asr_files(REPO_FILES)) == MEASURED_WORKING_SUBSET


def test_the_three_required_onnx_are_selected():
    selected = set(select_asr_files(REPO_FILES))
    assert "onnx/encoder_model.onnx" in selected
    # El par NO merged necesita las DOS mitades del decoder: use_merged=False no es
    # opcional porque con el merged DirectML devuelve basura.
    assert "onnx/decoder_model.onnx" in selected
    assert "onnx/decoder_with_past_model.onnx" in selected


def test_the_merged_decoder_is_never_selected():
    # Bajarlo seria peso muerto: el motor carga con use_merged=False.
    selected = set(select_asr_files(REPO_FILES))
    assert not any("merged" in name for name in selected)


def test_no_quantized_variant_is_selected():
    # El repo publica seis variantes por componente (fp16, int8, bnb4, q4, uint8,
    # quantized). Traerlas multiplicaria el peso sin que se usen.
    selected = set(select_asr_files(REPO_FILES))
    for marker in ("fp16", "int8", "quantized", "bnb4", "q4", "uint8"):
        assert not any(marker in name for name in selected), marker


def test_metadata_is_only_taken_from_the_root():
    # Un .json dentro de onnx/ es config de una variante que no se usa.
    assert is_required_asr_file("config.json") is True
    assert is_required_asr_file("onnx/config.json") is False


def test_files_that_nobody_needs_are_left_out():
    for name in (".gitattributes", "README.md", "model.safetensors", "onnx/foo.bin"):
        assert is_required_asr_file(name) is False, name


def test_windows_separators_are_understood():
    assert is_required_asr_file("onnx\\encoder_model.onnx") is True


def test_a_complete_repo_reports_nothing_missing():
    assert missing_required_onnx(REPO_FILES) == ()


def test_a_repo_without_the_encoder_reports_it_before_downloading():
    # Descubrirlo al final significaria dejar cientos de megas en disco para despues
    # fallar al cargar.
    incomplete = tuple(f for f in REPO_FILES if f != "onnx/encoder_model.onnx")
    assert missing_required_onnx(incomplete) == ("onnx/encoder_model.onnx",)


def test_a_repo_without_the_with_past_decoder_reports_it():
    incomplete = tuple(f for f in REPO_FILES if f != "onnx/decoder_with_past_model.onnx")
    assert missing_required_onnx(incomplete) == ("onnx/decoder_with_past_model.onnx",)


def test_a_repo_with_only_the_merged_decoder_is_reported_incomplete():
    # Un repo asi existe: publica el merged y no el par. No sirve para este motor.
    only_merged = (
        "config.json",
        "onnx/encoder_model.onnx",
        "onnx/decoder_model_merged.onnx",
    )
    missing = missing_required_onnx(only_merged)
    assert "onnx/decoder_model.onnx" in missing
    assert "onnx/decoder_with_past_model.onnx" in missing


def test_an_empty_repo_reports_every_required_file():
    assert len(missing_required_onnx(())) == 3
