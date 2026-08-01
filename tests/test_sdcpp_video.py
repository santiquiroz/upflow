from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.engines.sdcpp_video import (
    VIDEO_MODEL_PREFIX,
    SdcppVideoEngine,
    VideoRequest,
    list_video_models,
    normalize_frames,
    resolve_video_model,
)


def make_video_dir(tmp_path: Path, *diffusion_names: str, with_tae: bool = False) -> Path:
    video = tmp_path / "models" / "video"
    video.mkdir(parents=True)
    (video / "Wan2.2_VAE.safetensors").write_bytes(b"vae")
    (video / "umt5-xxl-encoder-Q5_K_M.gguf").write_bytes(b"te")
    if with_tae:
        (video / "taew2_2.safetensors").write_bytes(b"tae")
    for name in diffusion_names:
        (video / name).write_bytes(b"dit")
    return video


def make_settings(tmp_path: Path, enabled: bool = True) -> Settings:
    binary = tmp_path / "sd-cli.exe"
    binary.write_bytes(b"exe")
    return Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ENABLE_SDCPP=enabled,
        SDCPP_BINARY=str(binary),
        SDCPP_MODEL="",
        SDCPP_MODELS_DIR=str(tmp_path / "models"),
    )


def make_request(**overrides) -> VideoRequest:
    base = dict(prompt="un zorro corriendo", negative_prompt=None, steps=None,
                guidance=None, width=704, height=704, seed=None, frames=17, fps=16,
                init_image=None)
    base.update(overrides)
    return VideoRequest(**base)


# --- descubrimiento de modelos -------------------------------------------------

def test_lists_only_diffusion_checkpoints_not_vae_or_encoder(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf", "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    ids = [m.id for m in list_video_models(make_settings(tmp_path))]
    assert ids == [
        f"{VIDEO_MODEL_PREFIX}Wan2.2-TI2V-5B-Q8_0",
        f"{VIDEO_MODEL_PREFIX}Wan2_2-TI2V-5B-Turbo-Q8_0",
    ]


def test_no_models_when_vae_or_encoder_missing(tmp_path: Path) -> None:
    video = tmp_path / "models" / "video"
    video.mkdir(parents=True)
    (video / "Wan2.2-TI2V-5B-Q8_0.gguf").write_bytes(b"dit")
    assert list_video_models(make_settings(tmp_path)) == []


def test_resolve_never_returns_a_path_outside_the_video_dir(tmp_path: Path) -> None:
    """Propiedad de seguridad: el id llega por la API; ningun id puede sacar un
    archivo de fuera de la carpeta, ni siquiera uno que exista en disco."""
    video = make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    # Señuelos a un nivel y a dos: un `..` solo sube uno, y el test tiene que
    # cazar igual al que sube uno.
    vecino = video.parent / "secreto.gguf"
    vecino.write_bytes(b"privado")
    lejano = tmp_path / "secreto.gguf"
    lejano.write_bytes(b"privado")
    settings = make_settings(tmp_path)

    hostiles = [
        f"{VIDEO_MODEL_PREFIX}../secreto",
        f"{VIDEO_MODEL_PREFIX}..\\secreto",
        f"{VIDEO_MODEL_PREFIX}../../secreto",
        f"{VIDEO_MODEL_PREFIX}{lejano.with_suffix('')}",
        f"{VIDEO_MODEL_PREFIX}../../../../Windows/System32/config/SAM",
        "sdcpp:Wan2.2-TI2V-5B-Q8_0",
        "Wan2.2-TI2V-5B-Q8_0",
    ]
    for model_id in hostiles:
        resolved = resolve_video_model(model_id, settings)
        assert resolved is None or resolved.diffusion.resolve().parent == video.resolve(), model_id


def test_resolve_returns_the_model_for_a_legitimate_id(tmp_path: Path) -> None:
    video = make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    resolved = resolve_video_model(f"{VIDEO_MODEL_PREFIX}Wan2.2-TI2V-5B-Q8_0", settings)
    assert resolved is not None
    assert resolved.diffusion == video / "Wan2.2-TI2V-5B-Q8_0.gguf"


def test_turbo_is_detected_from_filename(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf", "Wan2.2-TI2V-5B-Q8_0.gguf")
    models = {m.name: m for m in list_video_models(make_settings(tmp_path))}
    assert models["Wan2_2-TI2V-5B-Turbo-Q8_0"].turbo is True
    assert models["Wan2.2-TI2V-5B-Q8_0"].turbo is False


# --- normalizacion de frames ---------------------------------------------------

@pytest.mark.parametrize(
    "pedidos,esperado",
    [(1, 1), (17, 17), (18, 17), (20, 17), (21, 21), (33, 33), (34, 33), (0, 1), (-5, 1)],
)
def test_frames_normalize_to_4n_plus_1(pedidos: int, esperado: int) -> None:
    assert normalize_frames(pedidos) == esperado


# --- construccion del comando --------------------------------------------------

def test_command_forces_text_encoder_to_cpu(tmp_path: Path) -> None:
    """El buffer unico de Vulkan en AMD topea en 4 GiB y umt5 pide 4.2 GB."""
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    command = SdcppVideoEngine(settings).build_command(make_request(), tmp_path / "o.webm", model)
    assert "--backend" in command
    assert command[command.index("--backend") + 1] == "te=cpu,diffusion=vulkan0,vae=vulkan0"


def test_command_always_uses_flash_attention_and_vae_tiling(tmp_path: Path) -> None:
    """Sin flash attention, Vulkan produce artefactos verdes; el VAE sin tiling no entra."""
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    command = SdcppVideoEngine(settings).build_command(make_request(), tmp_path / "o.webm", model)
    assert "--diffusion-fa" in command
    assert "--vae-tiling" in command


def test_command_wires_the_three_pack_files(tmp_path: Path) -> None:
    video = make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    command = SdcppVideoEngine(settings).build_command(make_request(), tmp_path / "o.webm", model)
    assert command[command.index("--diffusion-model") + 1] == str(video / "Wan2.2-TI2V-5B-Q8_0.gguf")
    assert command[command.index("--vae") + 1] == str(video / "Wan2.2_VAE.safetensors")
    assert command[command.index("--t5xxl") + 1] == str(video / "umt5-xxl-encoder-Q5_K_M.gguf")
    assert command[1] == "-M" and command[2] == "vid_gen"


def test_turbo_defaults_to_four_steps_and_cfg_one(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    command = SdcppVideoEngine(settings).build_command(make_request(), tmp_path / "o.webm", model)
    assert command[command.index("--steps") + 1] == "4"
    assert command[command.index("--cfg-scale") + 1] == "1.0"


def test_base_model_defaults_to_twenty_steps_and_cfg_five(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    command = SdcppVideoEngine(settings).build_command(make_request(), tmp_path / "o.webm", model)
    assert command[command.index("--steps") + 1] == "20"
    assert command[command.index("--cfg-scale") + 1] == "5.0"


def test_explicit_steps_and_guidance_win_over_defaults(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    request = make_request(steps=12, guidance=3.5)
    command = SdcppVideoEngine(settings).build_command(request, tmp_path / "o.webm", model)
    assert command[command.index("--steps") + 1] == "12"
    assert command[command.index("--cfg-scale") + 1] == "3.5"


def test_turbo_omits_negative_prompt_because_cfg_one_ignores_it(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    request = make_request(negative_prompt="borroso")
    command = SdcppVideoEngine(settings).build_command(request, tmp_path / "o.webm", model)
    assert "-n" not in command


def test_base_model_keeps_negative_prompt(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    request = make_request(negative_prompt="borroso")
    command = SdcppVideoEngine(settings).build_command(request, tmp_path / "o.webm", model)
    assert command[command.index("-n") + 1] == "borroso"


def test_frames_are_normalized_in_the_command(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    request = make_request(frames=34)
    command = SdcppVideoEngine(settings).build_command(request, tmp_path / "o.webm", model)
    assert command[command.index("--video-frames") + 1] == "33"


def test_init_image_switches_to_image_to_video(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    init = tmp_path / "inicio.png"
    init.write_bytes(b"png")
    command = SdcppVideoEngine(settings).build_command(
        make_request(init_image=init), tmp_path / "o.webm", model
    )
    assert command[command.index("-i") + 1] == str(init)


def test_dimensions_are_snapped_to_multiples_of_thirty_two(tmp_path: Path) -> None:
    """Hay fallos no lineales por alineacion: 287 anda y 288 revienta."""
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    request = make_request(width=700, height=500)
    command = SdcppVideoEngine(settings).build_command(request, tmp_path / "o.webm", model)
    assert command[command.index("-W") + 1] == "704"
    assert command[command.index("-H") + 1] == "512"


def test_output_must_be_webm(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    out = tmp_path / "salida.webm"
    command = SdcppVideoEngine(settings).build_command(make_request(), out, model)
    assert command[command.index("-o") + 1] == str(out)


# --- decodificador chico (TAE) -------------------------------------------------

def test_tae_is_used_instead_of_the_full_vae_when_present(tmp_path: Path) -> None:
    """Medido: el decode del VAE de Wan tarda 85,8 s y el del TAE 5,4 s."""
    video = make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf", with_tae=True)
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    command = SdcppVideoEngine(settings).build_command(make_request(), tmp_path / "o.webm", model)
    assert command[command.index("--tae") + 1] == str(video / "taew2_2.safetensors")
    assert "--vae" not in command
    # Con el TAE el decode ya no es el cuello: el tiling solo agrega costura.
    assert "--vae-tiling" not in command


def test_falls_back_to_the_full_vae_when_there_is_no_tae(tmp_path: Path) -> None:
    video = make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    settings = make_settings(tmp_path)
    model = list_video_models(settings)[0]
    command = SdcppVideoEngine(settings).build_command(make_request(), tmp_path / "o.webm", model)
    assert command[command.index("--vae") + 1] == str(video / "Wan2.2_VAE.safetensors")
    assert "--tae" not in command
    assert "--vae-tiling" in command


def test_tae_is_not_mistaken_for_the_vae_or_a_diffusion_model(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf", with_tae=True)
    settings = make_settings(tmp_path)
    models = list_video_models(settings)
    assert [m.name for m in models] == ["Wan2.2-TI2V-5B-Q8_0"]
    assert models[0].vae.name == "Wan2.2_VAE.safetensors"
    assert models[0].tae is not None and models[0].tae.name == "taew2_2.safetensors"


# --- disponibilidad ------------------------------------------------------------

def test_engine_unavailable_without_models(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    assert SdcppVideoEngine(make_settings(tmp_path)).available() is False


def test_engine_unavailable_when_flag_off(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    assert SdcppVideoEngine(make_settings(tmp_path, enabled=False)).available() is False


def test_engine_available_with_a_full_pack(tmp_path: Path) -> None:
    make_video_dir(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf")
    assert SdcppVideoEngine(make_settings(tmp_path)).available() is True
