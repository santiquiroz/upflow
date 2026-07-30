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


# ---------------------------------------------------------------------------
# El objetivo manda sobre la escala del modelo
#
# Sin esto, pedir 1080p sobre una fuente 4K corria el modelo al x4 del perfil
# (15360x8640, horas) y DESPUES reducia a 1920x1080: el mismo costo en GPU que el
# multiplicador ciego, y encima tirando el trabajo. Lo peor de los dos mundos.
# ---------------------------------------------------------------------------


def probe_for(width: int, height: int) -> dict:
    return {"streams": [{"codec_type": "video", "width": width, "height": height}]}


def test_without_a_target_the_requested_scale_is_untouched(tmp_path: Path):
    manager = make_manager(tmp_path)
    assert manager._scale_for_target(probe_for(1920, 1080), None, "realesrgan-x4plus", 4) == 4


def test_a_target_lowers_the_scale_when_the_model_offers_more_than_one(tmp_path: Path):
    """La ganancia real: 600p a 1080p alcanza con x2 en vez de x4.

    El costo por frame va con el CUADRADO de la escala, asi que x2 en vez de x4 es 4
    veces menos trabajo, y ese trabajo extra se tiraba en el redimensionado.

    Solo aplica a modelos multi-escala: realesr-animevideov3 ofrece [2,3,4].
    """
    manager = make_manager(tmp_path)
    scale = manager._scale_for_target(
        probe_for(1067, 600), 1080, "realesr-animevideov3", 4
    )

    assert scale == 2


def test_a_single_scale_model_cannot_be_made_cheaper_by_the_target(tmp_path: Path):
    """Limite honesto que hay que conocer: la mayoria de los modelos son de una sola
    escala.

    realesrgan-x4plus solo ofrece [4], asi que pedir 1080p NO baja el costo en GPU --
    el modelo hace x4 igual. Lo unico que cambia es el tamaño del archivo de salida.
    Prometer velocidad ahi seria mentir.
    """
    manager = make_manager(tmp_path)
    scale = manager._scale_for_target(probe_for(1067, 600), 1080, "realesrgan-x4plus", 4)

    assert scale == 4


def test_a_target_the_source_already_exceeds_is_refused_with_a_clear_reason(tmp_path: Path):
    """El caso 4K a 1080p: no necesita modelo, solo un redimensionado.

    Ese camino todavia no existe en el pipeline, asi que se RECHAZA diciendo por que,
    en vez de correr el modelo al x4 y quemar horas para despues tirar los pixeles.
    """
    import pytest

    manager = make_manager(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        manager._scale_for_target(probe_for(3840, 2160), 1080, "realesrgan-x4plus", 4)

    message = str(exc_info.value)
    assert "2160p" in message
    assert "redimensionado" in message


def test_the_scale_is_clamped_to_what_the_model_supports(tmp_path: Path):
    # Pedir una escala que el modelo no tiene haria fallar la resolucion del modelo
    # mas adelante, ya con el job encolado.
    manager = make_manager(tmp_path)
    option = manager.settings.get_model_option("realesrgan-x4plus")
    assert option is not None

    scale = manager._scale_for_target(probe_for(480, 270), 2160, "realesrgan-x4plus", 2)
    assert scale in option["scales"]


def test_missing_source_dimensions_leave_the_request_alone(tmp_path: Path):
    # Sin dimensiones no se puede planificar, y adivinar seria peor que respetar lo
    # que se pidio.
    manager = make_manager(tmp_path)
    assert manager._scale_for_target({"streams": []}, 1080, "realesrgan-x4plus", 3) == 3


def test_the_measured_saving_for_the_reported_case(tmp_path: Path):
    """Cuantifica la mejora del caso que se puede arreglar hoy.

    El costo por frame va con el cuadrado de la escala.
    """
    from app.services.target_resolution import megapixels_per_frame, plan_for_scale

    manager = make_manager(tmp_path)
    source = (1067, 600)
    profile_scale = 4
    planned = manager._scale_for_target(
        probe_for(*source), 1080, "realesr-animevideov3", profile_scale
    )

    blind = plan_for_scale(*source, profile_scale)
    targeted = plan_for_scale(*source, planned)
    blind_mp = megapixels_per_frame(blind.output_width, blind.output_height)
    targeted_mp = megapixels_per_frame(targeted.output_width, targeted.output_height)

    assert blind_mp / targeted_mp > 3.9
