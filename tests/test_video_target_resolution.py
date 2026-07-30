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
    """600p a 1080p alcanza con x2 en vez de x4.

    Que se ahorra, con precision: los PIXELES DE SALIDA bajan a la cuarta parte, y con
    ellos los PNG intermedios, el disco y el encode. La inferencia NO baja igual -- los
    bloques RRDB corren a resolucion de entrada y no cambian con la escala. Prometer
    "4 veces mas rapido" seria falso.

    Solo aplica a modelos multi-escala: realesr-animevideov3 ofrece [2,3,4].
    """
    manager = make_manager(tmp_path)
    scale = manager._scale_for_target(
        probe_for(1067, 600), 1080, "realesr-animevideov3", 4
    )

    assert scale == 2


def test_the_general_model_can_now_be_made_cheaper_by_the_target(tmp_path: Path):
    """El limite que este test documentaba ya no existe.

    realesrgan-x4plus ofrecia SOLO [4], asi que pedir 1080p desde 600p corria x4 igual y
    solo cambiaba el tamaño del archivo. Ahora ofrece [2,3,4]: el binario ncnn resuelve
    2x/3x con -s sobre el mismo modelo (medido: 64x48 -> 128x96 y 192x144) y hay exports
    ONNX derivados para el camino CPU. Asi que el objetivo SI baja la escala.

    Sigue valiendo la precision de siempre sobre QUE se ahorra: los pixeles de salida
    bajan a la cuarta parte y con ellos los PNG intermedios, el disco y el encode. La
    inferencia no baja igual -- el cuerpo RRDB corre a resolucion de entrada.
    """
    manager = make_manager(tmp_path)
    scale = manager._scale_for_target(probe_for(1067, 600), 1080, "realesrgan-x4plus", 4)

    assert scale == 2


def test_a_genuinely_single_scale_model_is_still_not_cheapened(tmp_path: Path):
    """El limite sigue existiendo donde es real.

    realesr-animevideov3-x4 es un model_id POR escala: pedir 1080p no puede bajarlo,
    porque el modelo elegido solo hace x4.
    """
    manager = make_manager(tmp_path)
    scale = manager._scale_for_target(
        probe_for(1067, 600), 1080, "realesr-animevideov3-x4", 4
    )

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
    """Cuantifica la mejora en PIXELES DE SALIDA, que es lo que se puede afirmar.

    Los pixeles de salida gobiernan los PNG intermedios, el disco y el encode. La
    inferencia esta dominada por la resolucion de ENTRADA y casi no cambia, asi que este
    numero NO es un factor de aceleracion total.
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


# ---------------------------------------------------------------------------
# El objetivo tambien manda en el camino streaming
#
# El -vf solo se inyectaba en el camino PNG. Los tres caminos de streaming ignoraban
# target_height por completo: el archivo salia en fuente*escala mientras
# _final_output_dims reportaba el objetivo, o sea que la metadata, la API y la vista
# previa del frontend mentian sobre el resultado entregado.
# ---------------------------------------------------------------------------


def test_the_rawpipe_encode_applies_the_target_resize(tmp_path: Path):
    upscaler = make_upscaler(tmp_path)
    job = make_job((1920, 1080), 2, target_height=1440)

    cmd = upscaler._build_rawpipe_command(
        3840, 2160, "24", None, [], tmp_path / "out.mp4", job, "libx264"
    )

    assert "-vf" in cmd, "el camino streaming descartaba el objetivo en silencio"
    assert "scale=2560:1440:flags=lanczos" in cmd


def test_the_rawpipe_encode_is_untouched_without_a_target(tmp_path: Path):
    # El camino del multiplicador no debe ganar una pasada de ffmpeg.
    upscaler = make_upscaler(tmp_path)
    job = make_job((1920, 1080), 2)

    cmd = upscaler._build_rawpipe_command(
        3840, 2160, "24", None, [], tmp_path / "out.mp4", job, "libx264"
    )

    assert "-vf" not in cmd


def test_what_the_rawpipe_produces_matches_what_the_job_reports(tmp_path: Path):
    """La invariante que el bug rompia: lo reportado y lo entregado son lo mismo."""
    upscaler = make_upscaler(tmp_path)
    job = make_job((1920, 1080), 2, target_height=1440)

    reported = upscaler._final_output_dims(job)
    cmd = upscaler._build_rawpipe_command(
        3840, 2160, "24", None, [], tmp_path / "out.mp4", job, "libx264"
    )
    filter_arg = cmd[cmd.index("-vf") + 1]

    assert filter_arg.startswith(f"scale={reported[0]}:{reported[1]}")
