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


def test_repo_without_model_index_and_root_safetensors_is_single_file_candidate():
    # Caso real: wikeeyang/Flux2-Klein-9B-True-V2 (15 archivos, sin model_index).
    verdict, reason = classify(("config.json", "weights.safetensors"), False)
    assert verdict == "single_file"
    assert reason.key == "compat.singleFile"


def test_repo_without_model_index_or_root_safetensors_is_incompatible():
    verdict, reason = classify(
        ("config.json", "nested/weights.safetensors"),
        False,
    )
    assert verdict == "incompatible"
    assert reason.params["filename"] == "model_index.json"


def test_torch_only_repo_needs_conversion():
    verdict, reason = classify(SD15, False)
    assert verdict == "needs_conversion"
    assert reason.key


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
        assert reason.key == "compat.gated"


def test_gated_wins_even_when_model_index_is_missing():
    # Sin token no se puede saber nada mas del repo, asi que gated gana.
    verdict, _ = classify(("config.json",), "auto")
    assert verdict == "gated"


def test_gated_wins_over_single_file_candidate():
    verdict, _ = classify(("checkpoint.safetensors",), "auto")
    assert verdict == "gated"


def test_gated_false_and_none_are_not_gated():
    for gated in (False, None, ""):
        verdict, _ = classify(SD15, gated)
        assert verdict != "gated"


# ---------------------------------------------------------------------------
# Bug reportado 2026-07-28: un repo de checkpoints sueltos (formato single-file,
# estilo Civitai) se clasificaba ready_onnx y el install fallaba despues de no
# descargar nada. Causa: torch_dirs y onnx_dirs son AMBOS vacios cuando los
# pesos estan en la raiz, y la resta de conjuntos vacios es vacuamente
# verdadera -> "todo componente torch tiene ONNX".
# Caso real: Manjushri/pornworks-...-sdxl-and-pony-chekpoint (7 archivos, todos
# en la raiz, 4 checkpoints de 6616 MB, model_index.json declarando 9
# componentes que no existen como carpetas).
# ---------------------------------------------------------------------------

SINGLE_FILE_CHECKPOINTS = (
    ".gitattributes",
    "README.md",
    "model_index.json",
    "pornworksRealPornPhoto_ponyV04.safetensors",
    "pornworksRealPornPhoto_v03.safetensors",
    "pornworksRealPorn_Illustrious_v4_04.safetensors",
)


def test_repo_with_weights_only_at_root_is_incompatible():
    verdict, reason = classify(SINGLE_FILE_CHECKPOINTS, False)
    assert verdict == "incompatible"
    assert reason.key


def test_incompatible_reason_distinguishes_the_single_file_layout():
    _verdict, reason = classify(SINGLE_FILE_CHECKPOINTS, False)
    # El motivo tiene que distinguirse del "falta model_index.json": aca SI
    # esta, el problema es otro y el usuario necesita saber cual. Con claves eso
    # se verifica directo en vez de buscar palabras en una oracion.
    assert reason.key == "compat.weightsAtRoot"
    assert reason.key != "compat.noModelIndex"


def test_repo_with_no_weights_at_all_is_incompatible():
    # Solo metadata, sin un solo peso: tampoco es instalable.
    verdict, _ = classify(("model_index.json", "README.md"), False)
    assert verdict == "incompatible"


def test_a_real_pipeline_layout_is_still_classified_normally():
    # Guard contra sobre-corregir: SD1.5 tiene pesos DENTRO de carpetas de
    # componente y tiene que seguir siendo needs_conversion.
    verdict, _ = classify(SD15, False)
    assert verdict == "needs_conversion"


def test_a_real_onnx_pipeline_is_still_ready():
    verdict, _ = classify(SDXL_ONNX, False)
    assert verdict == "ready_onnx"


def test_gated_still_wins_over_the_single_file_layout():
    verdict, _ = classify(SINGLE_FILE_CHECKPOINTS, "auto")
    assert verdict == "gated"


# --- por que un repo CON ONNX igual se convierte (estado: SIN resolver) ------
#
# Reportado: "todos los modelos piden conversion". Parte es inevitable: la
# mayoria de los repos de HuggingFace publican solo pesos PyTorch. Pero los mas
# populares -- stabilityai/stable-diffusion-xl-base-1.0, sdxl-turbo,
# SimianLuo/LCM_Dreamshaper_v7 -- SI traen un pipeline ONNX completo (unet,
# text encoders y vae_decoder/vae_encoder) y hoy se convierten igual, porque la
# comparacion es carpeta-a-carpeta y la carpeta torch `vae` nunca tiene gemela:
# al exportar, el VAE se parte en encoder y decoder (lo mismo que ya contempla
# _COMPONENT_ALIASES en generation_installer, del lado de la validacion).
#
# Se probo habilitarlos y quedo A MEDIAS, sin veredicto:
#   - La deteccion anda: sdxl-turbo paso de "converting" a "downloading" y bajo
#     los 7 GB de ONNX sin convertir (A/B con el mismo repo).
#   - La validacion posterior fallo con un DmlCommandRecorder 80004005, PERO la
#     prueba estaba contaminada: habia un job de generacion del usuario corriendo
#     en la misma GPU y fallo con el MISMO error en el mismo minuto. O sea fue
#     contencion de VRAM, no una prueba valida de que el fp32 del repo no entre.
#   - LCM fallo por su safety_checker, que el model_index declara y el pipeline
#     exige; excluirlo del veredicto lo rompe ("NoneType has no attribute decode").
#
# Queda revertido a proposito hasta re-probarlo con la GPU libre. Si entonces
# carga bien, el cambio ahorra ~40 min por modelo en los repos mas usados.


def test_a_repo_that_ships_fp32_onnx_still_needs_conversion() -> None:
    verdict, _ = classify(
        (
            "model_index.json",
            "unet/model.onnx",
            "unet/diffusion_pytorch_model.safetensors",
            "text_encoder/model.onnx",
            "vae_decoder/model.onnx",
            "vae_encoder/model.onnx",
            "vae/diffusion_pytorch_model.safetensors",
        ),
        gated=False,
    )
    assert verdict == "needs_conversion"


def test_a_pure_onnx_repo_is_still_ready() -> None:
    # El caso que si funciona tal cual: sin pesos torch al lado (amd/*_amdgpu).
    verdict, _ = classify(
        (
            "model_index.json",
            "unet/model.onnx",
            "text_encoder/model.onnx",
            "vae_decoder/model.onnx",
            "vae_encoder/model.onnx",
        ),
        gated=False,
    )
    assert verdict == "ready_onnx"
