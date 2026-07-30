from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.classic_upscalers import (
    CLASSIC_CATEGORY,
    CLASSIC_SCALES,
    CLASSIC_UPSCALERS,
    catalog_entries,
    is_classic_upscaler,
    swscale_flag_for,
)
from app.services.model_registry import ModelKind
from app.services.video_upscaler import VideoUpscaler

# Flags verificados contra el ffmpeg vendorizado. Si alguno dejara de existir el encode
# fallaria al FINAL del job, con todo el trabajo ya hecho.
VENDORED_SWSCALE_FLAGS = {
    "lanczos",
    "spline",
    "bicubic",
    "bilinear",
    "area",
    "neighbor",
    "gauss",
    "sinc",
}


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


def make_upscaler(tmp_path: Path) -> VideoUpscaler:
    return VideoUpscaler(make_settings(tmp_path), None, None, None)  # type: ignore[arg-type]


def make_job(
    source: tuple[int, int],
    scale: int,
    model_id: str = "classic-lanczos",
    target_height: int | None = None,
) -> VideoUpscaleJob:
    job = VideoUpscaleJob(
        source_path=Path("in.mp4"),
        original_filename="in.mp4",
        model_name=swscale_flag_for(model_id),
        scale=scale,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=17,
        keep_audio=True,
        target_height=target_height,
        model_id=model_id,
    )
    job.metadata["sourceWidth"], job.metadata["sourceHeight"] = source
    return job


# ---------------------------------------------------------------------------
# El catalogo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("upscaler", CLASSIC_UPSCALERS, ids=lambda u: u.key)
def test_every_flag_exists_in_the_vendored_ffmpeg(upscaler):
    assert upscaler.swscale_flag in VENDORED_SWSCALE_FLAGS


def test_the_label_says_out_loud_that_there_is_no_ai():
    # Es el punto de la funcion: alguien que elige esto tiene que saber que no hay IA,
    # no descubrirlo por el resultado.
    for upscaler in CLASSIC_UPSCALERS:
        assert "no AI" in upscaler.label


def test_scale_one_is_offered_so_pure_resizing_is_possible():
    # Es lo que habilita 4K a 1080p. Ningun modelo puede ofrecerlo: no tendria sentido
    # correr una red para no cambiar el tamaño.
    assert 1 in CLASSIC_SCALES


def test_the_classic_entries_are_in_the_catalog(tmp_path: Path):
    settings = make_settings(tmp_path)
    for upscaler in CLASSIC_UPSCALERS:
        assert upscaler.key in settings.model_keys


def test_the_ai_models_still_come_first_in_the_catalog(tmp_path: Path):
    # El clasico es una herramienta valida, no el default: si apareciera primero
    # empujaria a la gente a la opcion sin IA sin haberla pedido.
    catalog = make_settings(tmp_path).model_catalog
    categories = [item["category"] for item in catalog]
    assert categories[0] != CLASSIC_CATEGORY
    first_classic = next(i for i, c in enumerate(categories) if c == CLASSIC_CATEGORY)
    assert first_classic == len(catalog) - len(CLASSIC_UPSCALERS)


def test_the_catalog_entries_have_the_same_shape_as_a_model(tmp_path: Path):
    # Si les faltara una clave, el selector de la UI y la validacion necesitarian un
    # camino aparte para estas entradas.
    reference = set(make_settings(tmp_path).get_model_option("realesrgan-x4plus").keys())
    for entry in catalog_entries():
        assert set(entry.keys()) == reference


def test_the_engine_name_carries_the_swscale_flag():
    for entry in catalog_entries():
        assert entry["engine_name"] in VENDORED_SWSCALE_FLAGS


def test_only_the_classic_keys_are_recognized_as_classic():
    assert is_classic_upscaler("classic-lanczos") is True
    assert is_classic_upscaler("realesrgan-x4plus") is False
    assert is_classic_upscaler(None) is False


def test_an_unknown_key_falls_back_to_a_flag_that_exists():
    # Un flag inventado haria fallar el encode al final del job. Ante la duda, lanczos.
    assert swscale_flag_for("no-existe") in VENDORED_SWSCALE_FLAGS


# ---------------------------------------------------------------------------
# El filtro es TODO el trabajo en el camino clasico
# ---------------------------------------------------------------------------


def test_the_multiplier_path_scales_with_the_filter(tmp_path: Path):
    """Sin modelo, los frames siguen en resolucion de fuente.

    Si el filtro no se emitiera, un x2 clasico devolveria el video del mismo tamaño: el
    escalado se perderia en silencio.
    """
    args = make_upscaler(tmp_path)._resize_filter_args(make_job((1920, 1080), 2))

    assert args == ["-vf", "scale=3840:2160:flags=lanczos"]


def test_the_chosen_algorithm_is_the_one_used(tmp_path: Path):
    # Elegir "vecino mas cercano" y recibir lanczos convertiria la opcion en decorado.
    args = make_upscaler(tmp_path)._resize_filter_args(
        make_job((640, 480), 2, model_id="classic-neighbor")
    )

    assert args[1].endswith(":flags=neighbor")


def test_downscaling_4k_to_1080p_needs_no_model_at_all(tmp_path: Path):
    """El caso que motivo todo esto, ahora sin correr nada.

    Un 4K a 1080p con x4 tardaba 2,8 horas. Por el camino clasico es una pasada de
    swscale dentro del encode que ya iba a correr igual.
    """
    args = make_upscaler(tmp_path)._resize_filter_args(
        make_job((3840, 2160), 1, target_height=1080)
    )

    assert args == ["-vf", "scale=1920:1080:flags=lanczos"]


def test_scale_one_without_a_target_emits_no_filter(tmp_path: Path):
    # No se pidio cambiar el tamaño: escalar a la misma medida seria trabajo al vicio.
    assert make_upscaler(tmp_path)._resize_filter_args(make_job((1920, 1080), 1)) == []


def test_missing_source_dimensions_produce_no_filter(tmp_path: Path):
    # Sin las dimensiones no se puede calcular nada, y adivinar seria peor.
    job = make_job((1920, 1080), 2)
    job.metadata.pop("sourceWidth")

    assert make_upscaler(tmp_path)._resize_filter_args(job) == []


def test_an_ai_model_without_a_target_is_untouched(tmp_path: Path):
    # El camino del multiplicador con IA no debe ganar una pasada de ffmpeg: ahi el
    # modelo YA entrego los pixeles finales.
    job = make_job((1920, 1080), 4, model_id="realesrgan-x4plus")

    assert make_upscaler(tmp_path)._resize_filter_args(job) == []


# ---------------------------------------------------------------------------
# Las dimensiones que se reportan y con las que se elige el encoder
# ---------------------------------------------------------------------------


def test_the_reported_size_is_the_real_one_in_the_classic_path(tmp_path: Path):
    upscaler = make_upscaler(tmp_path)
    assert upscaler._final_output_dims(make_job((1920, 1080), 2)) == (3840, 2160)


def test_the_encoder_is_probed_against_the_target_not_the_multiplier(tmp_path: Path):
    """El encoder se elige probando que soporte la resolucion de salida.

    Pasarle fuente*escala descartaria encoders por hardware capaces de la salida REAL
    (o al reves, aceptaria uno que no llega).
    """
    upscaler = make_upscaler(tmp_path)
    dims = upscaler._expected_output_dims(make_job((3840, 2160), 1, target_height=1080))

    assert dims == (1920, 1080)


def test_unknown_source_dimensions_still_report_zero(tmp_path: Path):
    job = make_job((1920, 1080), 2, target_height=1080)
    job.metadata["sourceWidth"] = 0

    assert make_upscaler(tmp_path)._expected_output_dims(job) == (0, 0)


# ---------------------------------------------------------------------------
# La resolucion del "modelo"
# ---------------------------------------------------------------------------


def make_manager(tmp_path: Path):
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.video_job_manager import VideoJobManager

    settings = make_settings(tmp_path)
    return VideoJobManager(
        settings, None, None, DeviceSemaphores(settings)  # type: ignore[arg-type]
    )


def test_a_classic_upscaler_resolves_without_any_installed_model(tmp_path: Path):
    resolution = make_manager(tmp_path)._resolve_model("classic-lanczos", 2, None)

    assert resolution.kind == ModelKind.classic
    assert resolution.engine_model_name == "lanczos"
    assert resolution.scale == 2


def test_the_classic_path_accepts_the_cpu_device(tmp_path: Path):
    """swscale corre en CPU: exigir GPU excluiria a quien no tiene una.

    Los builtin ncnn SI rechazan 'cpu' porque necesitan Vulkan -- por eso el kind
    aparte, y no reusar builtin-ncnn.
    """
    resolution = make_manager(tmp_path)._resolve_model("classic-lanczos", 2, "cpu")

    assert resolution.kind == ModelKind.classic


def test_a_scale_the_classic_path_does_not_offer_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="scales"):
        make_manager(tmp_path)._resolve_model("classic-lanczos", 9, None)


def test_the_classic_kind_does_not_require_a_gpu():
    from app.services.device_router import has_compatible_device

    cpu_only = [{"id": "cpu", "name": "CPU", "kind": "cpu"}]

    assert has_compatible_device(cpu_only, ModelKind.classic) is True
    # Contraste que da sentido al kind nuevo: ncnn en la misma maquina no puede correr.
    assert has_compatible_device(cpu_only, ModelKind.builtin_ncnn) is False


# ---------------------------------------------------------------------------
# El objetivo en el camino clasico
# ---------------------------------------------------------------------------


def probe_for(width: int, height: int) -> dict:
    return {"streams": [{"codec_type": "video", "width": width, "height": height}]}


def test_a_target_makes_the_classic_scale_irrelevant(tmp_path: Path):
    # swscale va a cualquier medida en una sola pasada: no hay pasos enteros que elegir.
    scale = make_manager(tmp_path)._scale_for_target(
        probe_for(3840, 2160), 1080, "classic-lanczos", 4
    )

    assert scale == 1


def test_a_target_the_source_already_exceeds_is_allowed_for_classic(tmp_path: Path):
    """El mismo pedido que con un modelo se RECHAZA.

    Con IA, 4K a 1080p no tiene camino (los modelos solo amplian) y se rechaza
    explicando. Por el clasico es justamente el caso principal, asi que pasa.
    """
    manager = make_manager(tmp_path)

    with pytest.raises(ValueError, match="redimensionado"):
        manager._scale_for_target(probe_for(3840, 2160), 1080, "realesrgan-x4plus", 4)

    assert manager._scale_for_target(probe_for(3840, 2160), 1080, "classic-lanczos", 4) == 1


def test_without_a_target_the_classic_multiplier_is_respected(tmp_path: Path):
    scale = make_manager(tmp_path)._scale_for_target(
        probe_for(1920, 1080), None, "classic-lanczos", 2
    )

    assert scale == 2


# ---------------------------------------------------------------------------
# El motor NO corre
#
# Es la garantia central: si el pipeline entrara al escalado, un job clasico correria
# ncnn (el fallback por default cuando el modelo no esta en onnx) y devolveria un
# resultado con IA que nadie pidio -- o fallaria buscando pesos que no existen.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_classic_path_never_runs_the_upscale_engine(tmp_path: Path, monkeypatch):
    upscaler = make_upscaler(tmp_path)
    frames_in = tmp_path / "frames_in"
    frames_in.mkdir()
    frames_out = tmp_path / "frames_out"

    called: list[str] = []

    async def fail_if_called(*args, **kwargs):
        called.append("upscale")

    async def no_interpolation(job, src, fps, multiplier, target_fps):
        return src, fps

    async def would_stream(job):
        called.append("stream")
        return True

    monkeypatch.setattr(upscaler, "_upscale_frames", fail_if_called)
    monkeypatch.setattr(upscaler, "_maybe_interpolate", no_interpolation)
    monkeypatch.setattr(upscaler, "_should_stream", would_stream)

    encode_dir, encode_fps = await upscaler._interpolate_and_upscale(
        make_job((3840, 2160), 1, target_height=1080),
        frames_in,
        frames_out,
        "24",
        1,
        tmp_path / "out.mp4",
        None,
        [],
    )

    assert called == []
    # Los frames de entrada SON los del encode: sin copia y sin segundo directorio.
    assert encode_dir == frames_in
    assert not frames_out.exists()
    assert encode_fps == "24"


# ---------------------------------------------------------------------------
# Solo video
#
# El reescalado clasico vive en el encode de video. El pipeline de imagen no tiene ese
# paso, asi que aceptar el job lo encolaria para fallar en el motor ncnn buscando un
# modelo llamado "lanczos" -- despues de haber dicho que si.
# ---------------------------------------------------------------------------


def make_image_manager(tmp_path: Path):
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.job_manager import JobManager

    settings = make_settings(tmp_path)
    return JobManager(settings, None, DeviceSemaphores(settings))  # type: ignore[arg-type]


def test_an_image_job_refuses_a_classic_upscaler_with_the_reason(tmp_path: Path):
    with pytest.raises(ValueError, match="only available for video"):
        make_image_manager(tmp_path)._resolve_model(
            model_id="classic-lanczos", scale=2, output_format="png", device=None
        )


def test_an_image_job_still_accepts_a_real_model(tmp_path: Path):
    # Que el rechazo no se lleve puesto el camino normal.
    resolution = make_image_manager(tmp_path)._resolve_model(
        model_id="realesrgan-x4plus", scale=4, output_format="png", device=None
    )

    assert resolution.model_id == "realesrgan-x4plus"


# ---------------------------------------------------------------------------
# La guarda contra el fallo silencioso de ncnn
#
# realesrgan-x4plus ofrece 2x y 3x, pero SOLO por ONNX. El binario ncnn devuelve las
# dimensiones pedidas con el contenido roto y sale con codigo 0, asi que sin esta guarda
# el usuario recibiria un video destruido sin un solo error.
# ---------------------------------------------------------------------------


def make_manager_without_onnx_exports(tmp_path: Path):
    """Manager apuntando a un directorio de exports VACIO.

    Hace falta apuntarlo a proposito: builtin_onnx_path se resuelve contra la raiz del
    proyecto, no contra el runtime, asi que un tmp_path no lo deja vacio -- ve los
    exports reales del repo. Un test que no lo hiciera pasaria por la razon equivocada.
    """
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.video_job_manager import VideoJobManager

    settings = Settings(
        RUNTIME_DIR=str(tmp_path),
        BUILTIN_ONNX_DIR=str(tmp_path / "sin-exports"),
        _env_file=None,
    )
    return VideoJobManager(settings, None, None, DeviceSemaphores(settings))  # type: ignore[arg-type]


def test_a_scale_only_onnx_can_do_is_refused_when_the_export_is_missing(tmp_path: Path):
    manager = make_manager_without_onnx_exports(tmp_path)

    with pytest.raises(ValueError, match="ONNX"):
        manager._resolve_model("realesrgan-x4plus", 2, None)


def test_the_same_model_at_its_native_scale_needs_no_onnx(tmp_path: Path):
    # 4x es nativo en ncnn: anda incluso sin ningun export ONNX en disco.
    manager = make_manager_without_onnx_exports(tmp_path)
    resolution = manager._resolve_model("realesrgan-x4plus", 4, None)

    assert resolution.scale == 4


def test_the_scale_is_allowed_once_the_onnx_export_exists(tmp_path: Path):
    manager = make_manager_without_onnx_exports(tmp_path)
    onnx_dir = manager.settings.builtin_onnx_path
    onnx_dir.mkdir(parents=True, exist_ok=True)
    (onnx_dir / "realesrgan-x4plus-x2-uint8.onnx").write_bytes(b"grafo")

    resolution = manager._resolve_model("realesrgan-x4plus", 2, None)

    assert resolution.scale == 2


def test_a_per_scale_model_is_unaffected_by_the_guard(tmp_path: Path):
    # animevideov3 trae archivos ncnn por escala: 2x es nativo y no necesita ONNX.
    resolution = make_manager(tmp_path)._resolve_model("realesr-animevideov3-x2", 2, None)

    assert resolution.scale == 2


def test_the_runtime_is_forced_to_onnx_for_a_scale_ncnn_would_corrupt(tmp_path: Path):
    """El ruteo no puede caer a ncnn como fallback cuando ncnn corrompe.

    La regla auto normalmente prefiere ncnn si el export ONNX no esta; aca eso entregaria
    un video roto, asi que la escala manda sobre la preferencia.
    """
    from app.services.backend_registry import UpscaleBackend

    upscaler = make_upscaler(tmp_path)
    job = make_job((1920, 1080), 2, model_id="realesrgan-x4plus")
    job.model_name = "realesrgan-x4plus"

    assert upscaler._resolve_builtin_backend(job) == UpscaleBackend.onnx
