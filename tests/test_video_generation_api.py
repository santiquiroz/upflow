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


def test_response_marks_a_video_job_so_the_ui_can_render_a_player() -> None:
    from app.api.routes import generation_job_to_response
    from app.models import GenerationJob, JobStatus

    video = GenerationJob(prompt="x", model_id=f"{VIDEO_MODEL_PREFIX}Wan", status=JobStatus.completed)
    imagen = GenerationJob(prompt="x", model_id="gen--sd15", status=JobStatus.completed)
    assert generation_job_to_response(video).is_video is True
    assert generation_job_to_response(imagen).is_video is False


@pytest.mark.asyncio
async def test_download_serves_webm_as_video_not_as_png(tmp_path: Path) -> None:
    """Servido como image/png, el navegador se baja el archivo en vez de reproducirlo."""
    from app.api.routes import download_generation_job
    from app.models import GenerationJob, JobStatus
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=SdcppVideoEngine(settings),
    )
    salida = tmp_path / "clip.webm"
    salida.write_bytes(b"webm")
    job = GenerationJob(
        prompt="x", model_id=f"{VIDEO_MODEL_PREFIX}Wan",
        status=JobStatus.completed, output_path=salida,
    )
    manager.jobs[job.id] = job
    response = await download_generation_job(job.id, generation_jobs=manager)
    assert response.media_type == "video/webm"


@pytest.mark.asyncio
async def test_download_still_serves_images_as_png(tmp_path: Path) -> None:
    from app.api.routes import download_generation_job
    from app.models import GenerationJob, JobStatus
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=SdcppVideoEngine(settings),
    )
    salida = tmp_path / "imagen.png"
    salida.write_bytes(b"png")
    job = GenerationJob(
        prompt="x", model_id="gen--sd15", status=JobStatus.completed, output_path=salida,
    )
    manager.jobs[job.id] = job
    response = await download_generation_job(job.id, generation_jobs=manager)
    assert response.media_type == "image/png"


def test_video_accepts_the_cinematic_aspect_ratio(tmp_path: Path) -> None:
    """832x480 es la relación con la que se entrenó Wan y la que se midió mejor.
    480 no es múltiplo de 64, que es la regla de las imágenes."""
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=SdcppVideoEngine(settings),
    )
    manager._validate_params(
        "un zorro", 4, 832, 480, model_id=f"{VIDEO_MODEL_PREFIX}Wan2_2-TI2V-5B-Turbo-Q8_0"
    )


def test_images_still_require_multiples_of_sixty_four(tmp_path: Path) -> None:
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=SdcppVideoEngine(settings),
    )
    with pytest.raises(ValueError, match="64"):
        manager._validate_params("un zorro", 25, 832, 480, model_id="gen--sd15")


def test_video_still_rejects_dimensions_off_the_thirty_two_grid(tmp_path: Path) -> None:
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=SdcppVideoEngine(settings),
    )
    with pytest.raises(ValueError, match="32"):
        manager._validate_params(
            "un zorro", 4, 833, 480, model_id=f"{VIDEO_MODEL_PREFIX}Wan2_2-TI2V-5B-Turbo-Q8_0"
        )


def test_video_reserves_the_gpu_so_it_does_not_run_next_to_another_gpu_job(tmp_path: Path) -> None:
    """Dos difusiones a la vez en la misma placa se degradan mutuamente (medido:
    24 s/it -> 60 s/it y despues hambreado). Sin device, el job de video caía en
    un bucket propio y corría en paralelo con un upscale en dml:0."""
    from app.models import GenerationJob
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=SdcppVideoEngine(settings),
    )
    sin_device = GenerationJob(prompt="x", model_id=f"{VIDEO_MODEL_PREFIX}Wan")
    assert manager._reservation_device(sin_device) == settings.default_device


def test_an_explicit_device_still_wins_for_video(tmp_path: Path) -> None:
    from app.models import GenerationJob
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=SdcppVideoEngine(settings),
    )
    job = GenerationJob(prompt="x", model_id=f"{VIDEO_MODEL_PREFIX}Wan", device="dml:1")
    assert manager._reservation_device(job) == "dml:1"


def test_image_jobs_keep_reserving_exactly_what_they_asked_for(tmp_path: Path) -> None:
    from app.models import GenerationJob
    from app.services.generation_job_manager import GenerationJobManager

    settings = make_pack(tmp_path, "Wan2_2-TI2V-5B-Turbo-Q8_0.gguf")
    manager = GenerationJobManager(
        settings, engine=None, device_semaphores=None, registry=None,
        upscale_engine=None, video_engine=SdcppVideoEngine(settings),
    )
    job = GenerationJob(prompt="x", model_id="gen--sd15", device=None)
    assert manager._reservation_device(job) is None


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
