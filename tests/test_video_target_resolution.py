from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.video_upscaler import VideoUpscaler


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


def make_upscaler(tmp_path: Path) -> VideoUpscaler:
    # Los dos metodos bajo prueba son puros: solo leen el job y su metadata, asi que
    # no hace falta ni motor ni media_tools de verdad.
    return VideoUpscaler(make_settings(tmp_path), None, None, None)  # type: ignore[arg-type]


def make_job(
    source: tuple[int, int], scale: int, target_height: int | None = None
) -> VideoUpscaleJob:
    job = VideoUpscaleJob(
        source_path=Path("in.mp4"),
        original_filename="in.mp4",
        model_name="realesrgan-x4plus",
        scale=scale,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=17,
        keep_audio=True,
        target_height=target_height,
    )
    job.metadata["sourceWidth"], job.metadata["sourceHeight"] = source
    return job


# ---------------------------------------------------------------------------
# El filtro de redimensionado
# ---------------------------------------------------------------------------


def test_no_target_means_no_resize_filter(tmp_path: Path):
    # El camino viejo del multiplicador queda intacto: ni una pasada de ffmpeg extra.
    upscaler = make_upscaler(tmp_path)
    assert upscaler._resize_filter_args(make_job((1920, 1080), 4)) == []


def test_a_target_that_the_integer_scale_hits_exactly_needs_no_filter(tmp_path: Path):
    # 540 x2 = 1080 justo: redimensionar seria una pasada al vicio.
    upscaler = make_upscaler(tmp_path)
    assert upscaler._resize_filter_args(make_job((960, 540), 2, target_height=1080)) == []


def test_a_target_below_what_the_scale_produces_adds_a_lanczos_downscale(tmp_path: Path):
    # 600 x2 = 1200, y se pidio 1080: hay que bajar a la medida exacta.
    upscaler = make_upscaler(tmp_path)
    args = upscaler._resize_filter_args(make_job((1067, 600), 2, target_height=1080))

    assert args[0] == "-vf"
    assert args[1].startswith("scale=")
    assert args[1].endswith(":flags=lanczos")
    assert ":1080:" in args[1]


def test_the_case_that_motivated_this_downscales_4k_to_1080p(tmp_path: Path):
    """El caso real: un 4K con escala x4 pedia 15360x8640 y tardaba 2,8 horas.

    Pidiendo 1080p el archivo sale en 1920x1080.
    """
    upscaler = make_upscaler(tmp_path)
    args = upscaler._resize_filter_args(make_job((3840, 2160), 1, target_height=1080))

    assert args == ["-vf", "scale=1920:1080:flags=lanczos"]


def test_missing_source_dimensions_produce_no_filter(tmp_path: Path):
    # Sin las dimensiones de la fuente no se puede calcular nada, y adivinar seria
    # peor que no redimensionar.
    upscaler = make_upscaler(tmp_path)
    job = make_job((1920, 1080), 2, target_height=1080)
    job.metadata.pop("sourceWidth")

    assert upscaler._resize_filter_args(job) == []


# ---------------------------------------------------------------------------
# Las dimensiones que se reportan
# ---------------------------------------------------------------------------


def test_without_a_target_the_reported_size_is_source_times_scale(tmp_path: Path):
    upscaler = make_upscaler(tmp_path)
    assert upscaler._final_output_dims(make_job((1920, 1080), 4)) == (7680, 4320)


def test_with_a_target_the_reported_size_is_the_target(tmp_path: Path):
    # Reportar sourceWidth*scale mentiria en la UI y en la cola de jobs.
    upscaler = make_upscaler(tmp_path)
    dims = upscaler._final_output_dims(make_job((1067, 600), 2, target_height=1080))

    assert dims[1] == 1080
    assert dims != (1067 * 2, 600 * 2)


def test_the_aspect_ratio_survives_in_the_reported_size(tmp_path: Path):
    upscaler = make_upscaler(tmp_path)
    width, height = upscaler._final_output_dims(make_job((1440, 1080), 2, target_height=2160))

    assert abs(width / height - 1440 / 1080) < 0.01


# ---------------------------------------------------------------------------
# Validacion del alto pedido
# ---------------------------------------------------------------------------


def make_manager(tmp_path: Path):
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.video_job_manager import VideoJobManager

    settings = make_settings(tmp_path)
    # _validate_target_height es puro: no toca ni el upscaler ni media_tools.
    return VideoJobManager(
        settings, None, None, DeviceSemaphores(settings)  # type: ignore[arg-type]
    )


def test_no_target_height_is_valid(tmp_path: Path):
    # El camino del multiplicador sigue siendo valido.
    make_manager(tmp_path)._validate_target_height(None)


def test_a_reasonable_target_height_is_valid(tmp_path: Path):
    manager = make_manager(tmp_path)
    for height in (720, 1080, 1440, 2160):
        manager._validate_target_height(height)


def test_an_odd_target_height_is_rejected(tmp_path: Path):
    # yuv420p no acepta impares y el encode falla al FINAL del job, despues de todo el
    # trabajo: rechazarlo al crear es mucho mas barato.
    import pytest

    with pytest.raises(ValueError, match="even"):
        make_manager(tmp_path)._validate_target_height(1081)


def test_an_absurd_target_height_is_rejected(tmp_path: Path):
    # Sin techo, un 100000 reproduciria el problema del multiplicador ciego: un pedido
    # enorme que nada advierte.
    import pytest

    manager = make_manager(tmp_path)
    for height in (2, 100, 100000):
        with pytest.raises(ValueError):
            manager._validate_target_height(height)


def test_eight_k_is_the_ceiling_and_is_allowed(tmp_path: Path):
    make_manager(tmp_path)._validate_target_height(8640)
