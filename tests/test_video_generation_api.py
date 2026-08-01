from __future__ import annotations

from pathlib import Path

import pytest

from app.api.routes import _video_aware_sampling, video_generation_capabilities
from app.config import Settings
from app.schemas import CreateGenerationJobRequest
from app.services.engines.sdcpp_video import VIDEO_MODEL_PREFIX, SdcppVideoEngine


def make_pack(tmp_path: Path, *diffusion_names: str) -> Settings:
    binary = tmp_path / "sd-cli.exe"
    binary.write_bytes(b"exe")
    video = tmp_path / "models" / "video"
    video.mkdir(parents=True)
    (video / "Wan2.2_VAE.safetensors").write_bytes(b"vae")
    (video / "umt5-xxl-encoder-Q5_K_M.gguf").write_bytes(b"te")
    (video / "taew2_2.safetensors").write_bytes(b"tae")
    for name in diffusion_names:
        (video / name).write_bytes(b"dit")
    return Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ENABLE_SDCPP=True,
        SDCPP_BINARY=str(binary),
        SDCPP_MODEL="",
        SDCPP_MODELS_DIR=str(tmp_path / "models"),
    )


def make_payload(**overrides) -> CreateGenerationJobRequest:
    base = {"prompt": "un zorro", "modelId": f"{VIDEO_MODEL_PREFIX}Wan2_2-TI2V-5B-Turbo-Q8_0"}
    base.update(overrides)
    return CreateGenerationJobRequest.model_validate(base)


@pytest.mark.asyncio
async def test_capabilities_are_empty_without_the_pack(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    binary = tmp_path / "sd-cli.exe"
    binary.write_bytes(b"exe")
    settings = Settings(
        _env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), ENABLE_SDCPP=True,
        SDCPP_BINARY=str(binary), SDCPP_MODEL="", SDCPP_MODELS_DIR=str(tmp_path / "models"),
    )
    response = await video_generation_capabilities(settings=settings)
    assert response.available is False
    assert response.models == []


@pytest.mark.asyncio
async def test_capabilities_flag_the_fast_model_and_expose_its_defaults(tmp_path: Path) -> None:
    settings = make_pack(tmp_path, "Wan2.2-TI2V-5B-Q8_0.gguf", "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    response = await video_generation_capabilities(settings=settings)
    assert response.available is True
    por_nombre = {m.name: m for m in response.models}
    turbo = por_nombre["Wan2_2-TI2V-5B-Turbo-Q8_0 (Vulkan)"]
    base = por_nombre["Wan2.2-TI2V-5B-Q8_0 (Vulkan)"]
    assert (turbo.fast, turbo.default_steps, turbo.default_guidance) == (True, 4, 1.0)
    assert (base.fast, base.default_steps, base.default_guidance) == (False, 20, 5.0)


def test_video_model_gets_its_own_sampling_defaults_not_the_image_ones(tmp_path: Path) -> None:
    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    steps, guidance = _video_aware_sampling(make_payload(), settings)
    assert (steps, guidance) == (4, 1.0)


def test_explicit_sampling_wins_even_on_a_video_model(tmp_path: Path) -> None:
    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    steps, guidance = _video_aware_sampling(make_payload(steps=12, guidance=3.0), settings)
    assert (steps, guidance) == (12, 3.0)


def test_explicit_sampling_equal_to_the_image_default_is_still_honoured(tmp_path: Path) -> None:
    """Pedir 25 pasos a propósito no puede confundirse con no haber pedido nada."""
    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    steps, guidance = _video_aware_sampling(make_payload(steps=25, guidance=7.5), settings)
    assert (steps, guidance) == (25, 7.5)


def test_image_models_keep_the_image_defaults(tmp_path: Path) -> None:
    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    payload = make_payload(modelId="gen--stabilityai--sd15")
    assert _video_aware_sampling(payload, settings) == (25, 7.5)


@pytest.mark.asyncio
async def test_job_writes_a_webm_and_passes_frames_through(tmp_path: Path, monkeypatch) -> None:
    from app.models import GenerationJob
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    settings.outputs_path.mkdir(parents=True, exist_ok=True)
    capturado: dict = {}

    async def fake_run(request, output_path, model):
        capturado["request"] = request
        capturado["output"] = output_path
        capturado["model"] = model
        output_path.write_bytes(b"webm")
        return output_path

    engine = SdcppVideoEngine(settings)
    monkeypatch.setattr(engine, "run", fake_run)
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=engine,
    )
    job = GenerationJob(
        prompt="un zorro", model_id=f"{VIDEO_MODEL_PREFIX}Wan2_2-TI2V-5B-Turbo-Q8_0",
        # Distintos del default a propósito: si el motor los ignorara y usara
        # los suyos, este test no lo notaría.
        steps=4, guidance=1.0, width=832, height=480, frames=49, fps=24,
    )
    await manager._run_video(job)

    assert capturado["output"].suffix == ".webm"
    assert job.output_path == capturado["output"]
    assert (capturado["request"].frames, capturado["request"].fps) == (49, 24)
    assert capturado["model"].turbo is True


@pytest.mark.asyncio
async def test_job_falls_back_to_the_default_clip_length(tmp_path: Path, monkeypatch) -> None:
    from app.models import GenerationJob
    from app.services.generation_job_manager import (
        DEFAULT_VIDEO_FPS,
        DEFAULT_VIDEO_FRAMES,
        GenerationJobManager,
    )

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    settings.outputs_path.mkdir(parents=True, exist_ok=True)
    capturado: dict = {}

    async def fake_run(request, output_path, model):
        capturado["request"] = request
        output_path.write_bytes(b"webm")
        return output_path

    engine = SdcppVideoEngine(settings)
    monkeypatch.setattr(engine, "run", fake_run)
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=engine,
    )
    job = GenerationJob(
        prompt="un zorro", model_id=f"{VIDEO_MODEL_PREFIX}Wan2_2-TI2V-5B-Turbo-Q8_0",
    )
    await manager._run_video(job)
    assert capturado["request"].frames == DEFAULT_VIDEO_FRAMES
    assert capturado["request"].fps == DEFAULT_VIDEO_FPS


def test_unknown_video_model_is_rejected_at_creation(tmp_path: Path) -> None:
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=SdcppVideoEngine(settings),
    )
    with pytest.raises(ValueError, match="video"):
        manager._validate_generation_model(f"{VIDEO_MODEL_PREFIX}no-existe")


def test_video_model_is_accepted_at_creation(tmp_path: Path) -> None:
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=SdcppVideoEngine(settings),
    )
    manager._validate_generation_model(f"{VIDEO_MODEL_PREFIX}Wan2_2-TI2V-5B-Turbo-Q8_0")
