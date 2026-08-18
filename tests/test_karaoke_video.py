"""Karaoke completo: instrumental + letra sincronizada, en un solo trabajo.

Las tres piezas ya existian sueltas —separar, transcribir con tiempos, quemar
subtitulos—. Lo que estas pruebas fijan es el EMPALME, que es lo que ninguna de
las tres hace sola: que la imagen termine con la pista instrumental y no con la
original, que la letra salga del audio completo, y que un mp3 con caratula no se
confunda con un video.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.models import TranscribeJob
from app.services.karaoke_video import (
    BACKGROUND_COLOR,
    build_karaoke_command,
    build_picture_probe_command,
    has_real_picture,
)
from app.services.transcribe_job_manager import OUTPUT_MODES, TranscribeJobManager


def test_el_karaoke_es_un_modo_de_salida_mas() -> None:
    assert "karaoke" in OUTPUT_MODES


# ---------------------------------------------------------------------------
# Que cuenta como imagen de fondo
# ---------------------------------------------------------------------------


def test_un_video_de_verdad_sirve_de_fondo() -> None:
    assert has_real_picture({"streams": [{"codec_type": "video", "disposition": {}}]})


def test_la_caratula_de_un_mp3_no_es_un_video() -> None:
    # ES un stream de video para ffprobe, pero de UN cuadro: usarla de fondo da
    # un video de un cuadro con la cancion cortada.
    caratula = {"streams": [{"codec_type": "video", "disposition": {"attached_pic": 1}}]}

    assert not has_real_picture(caratula)


def test_un_audio_pelado_no_tiene_imagen() -> None:
    assert not has_real_picture({"streams": []})
    assert not has_real_picture({})


def test_el_probe_de_imagen_pregunta_por_la_caratula() -> None:
    comando = build_picture_probe_command(Path("ffprobe"), Path("a.mp3"))

    # Sin pedir `attached_pic` la respuesta no alcanza para distinguir los dos
    # casos de arriba.
    assert "attached_pic" in " ".join(comando)
    assert "v:0" in comando


# ---------------------------------------------------------------------------
# El comando que arma el video
# ---------------------------------------------------------------------------


def comando(**extra):
    base = dict(
        ffmpeg="ffmpeg",
        picture=None,
        duration_seconds=210.5,
        instrumental=Path("inst.wav"),
        subtitles=Path("letra.srt"),
        destination=Path("out.mp4"),
    )
    base.update(extra)
    return build_karaoke_command(**base)


def test_el_audio_sale_del_instrumental_y_no_del_video() -> None:
    args = comando(picture=Path("clip.mp4"))

    # ESTE es el punto del karaoke. Sin el map explicito ffmpeg toma la pista
    # del video, o sea la voz original, y el resultado es el tema con letra.
    assert "1:a" in args
    assert args.index("-map") < args.index("1:a")


def test_sin_imagen_el_fondo_lo_genera_ffmpeg() -> None:
    args = comando()

    assert "lavfi" in args
    assert BACKGROUND_COLOR in " ".join(args)
    # La duracion va en la ENTRADA: `color` es una fuente infinita y sin cortarla
    # ahi ffmpeg genera cuadros para siempre.
    assert args.index("-t") < args.index("-i")
    assert "210.5" in args


def test_con_imagen_no_se_genera_ningun_fondo() -> None:
    args = comando(picture=Path("clip.mp4"))

    assert "lavfi" not in args
    assert "clip.mp4" in args


def test_la_letra_se_quema_en_la_imagen() -> None:
    args = comando()

    filtro = args[args.index("-filter_complex") + 1]
    # Quemada y no muxeada: un karaoke con la letra en una pista que el
    # reproductor puede no mostrar no es un karaoke.
    assert "subtitles=" in filtro
    assert "letra.srt" in filtro


def test_el_corte_lo_pone_el_mas_corto_de_los_dos() -> None:
    # El video original puede durar mas o menos que el instrumental: sin esto
    # queda cola muda o audio sin imagen.
    assert "-shortest" in comando(picture=Path("clip.mp4"))


def test_no_se_escribe_encima_del_original() -> None:
    with pytest.raises(ValueError):
        comando(picture=Path("out.mp4"), destination=Path("out.mp4"))


# ---------------------------------------------------------------------------
# El paso completo dentro del manager
# ---------------------------------------------------------------------------


class SeparadorFalso:
    def __init__(self) -> None:
        self.corridas: list[tuple] = []

    async def run(self, decoded, stem_wavs, device, **kwargs):
        self.corridas.append((decoded, list(stem_wavs), device, kwargs))
        for wav in stem_wavs:
            wav.write_bytes(b"stem")


def manager(tmp_path: Path, separador) -> TranscribeJobManager:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    return TranscribeJobManager(
        settings,
        engine=None,
        device_semaphores=None,
        registry=None,
        separators={"mdx": separador, "vr": separador, "roformer": separador},
    )


@pytest.fixture
def ffmpeg_falso(monkeypatch):
    llamadas: list[list[str]] = []

    async def correr(command, _timeout, **_kwargs):
        llamadas.append(command)
        if "attached_pic" in " ".join(command):
            return json.dumps({"streams": []}).encode(), b"", 0
        if "-show_entries" in command:
            return json.dumps({"format": {"duration": "200.0"}}).encode(), b"", 0
        Path(command[-1]).write_bytes(b"salida")
        return b"", b"", 0

    for modulo in ("transcribe_job_manager", "audio_excerpt"):
        monkeypatch.setattr(f"app.services.{modulo}.run_guarded_process", correr)
    return llamadas


def job_de_karaoke(tmp_path: Path) -> TranscribeJob:
    fuente = tmp_path / "cancion.mp3"
    fuente.write_bytes(b"x")
    job = TranscribeJob(
        source_path=fuente,
        original_filename="cancion.mp3",
        model_id="whisper-base",
        output_mode="karaoke",
    )
    job.segments = []
    return job


def test_el_karaoke_separa_y_arma_el_video(tmp_path: Path, ffmpeg_falso) -> None:
    separador = SeparadorFalso()
    gestor = manager(tmp_path, separador)
    job = job_de_karaoke(tmp_path)

    salida = asyncio.run(gestor._build_karaoke_video(job))

    assert salida.exists()
    assert len(separador.corridas) == 1
    # A 44100 estereo: entrarle otra cosa al grafo lo hace remuestrear adentro.
    decode = next(c for c in ffmpeg_falso if "pcm_s16le" in c)
    assert decode[decode.index("-ar") + 1] == "44100"
    assert decode[decode.index("-ac") + 1] == "2"


def test_el_instrumental_no_queda_ocupando_disco(tmp_path: Path, ffmpeg_falso) -> None:
    gestor = manager(tmp_path, SeparadorFalso())
    job = job_de_karaoke(tmp_path)

    asyncio.run(gestor._build_karaoke_video(job))

    # Un wav sin comprimir por karaoke, cuando ya quedo pintado dentro del mp4.
    assert not (gestor.settings.outputs_path / f"{job.id}.karaoke").exists()


def test_sin_motor_de_separacion_falla_diciendolo(tmp_path: Path, ffmpeg_falso) -> None:
    gestor = manager(tmp_path, SeparadorFalso())
    gestor.separators = {}
    job = job_de_karaoke(tmp_path)

    with pytest.raises(RuntimeError) as error:
        asyncio.run(gestor._build_karaoke_video(job))

    # Pedir karaoke y recibir un video con la voz intacta seria peor que fallar.
    assert "separacion" in str(error.value)


def test_un_armado_fallido_no_deja_el_wav_ocupando_disco(tmp_path: Path, monkeypatch) -> None:
    gestor = manager(tmp_path, SeparadorFalso())
    job = job_de_karaoke(tmp_path)

    async def falla_al_armar(command, _timeout, **_kwargs):
        if "-filter_complex" in command:
            return b"", b"ffmpeg exploto", 1
        if "attached_pic" in " ".join(command):
            return json.dumps({"streams": []}).encode(), b"", 0
        if "-show_entries" in command:
            return json.dumps({"format": {"duration": "200.0"}}).encode(), b"", 0
        Path(command[-1]).write_bytes(b"salida")
        return b"", b"", 0

    for modulo in ("transcribe_job_manager", "audio_excerpt"):
        monkeypatch.setattr(f"app.services.{modulo}.run_guarded_process", falla_al_armar)

    with pytest.raises(RuntimeError):
        asyncio.run(gestor._build_karaoke_video(job))

    # El wav sin comprimir es el archivo mas pesado que produce el trabajo: un
    # fallo no puede costar disco para siempre.
    assert not (gestor.settings.outputs_path / f"{job.id}.karaoke").exists()


def test_una_separacion_fallida_tampoco_deja_la_carpeta(tmp_path: Path, ffmpeg_falso) -> None:
    class Explota(SeparadorFalso):
        async def run(self, *args, **kwargs):
            raise RuntimeError("el modelo reviento")

    gestor = manager(tmp_path, Explota())
    job = job_de_karaoke(tmp_path)

    with pytest.raises(RuntimeError):
        asyncio.run(gestor._build_karaoke_video(job))

    assert not (gestor.settings.outputs_path / f"{job.id}.karaoke").exists()


def test_el_principal_lo_decide_el_catalogo_y_no_la_posicion(tmp_path: Path, ffmpeg_falso) -> None:
    from app.services.engines.separation_models import (
        DEFAULT_SEPARATION_MODEL,
        SEPARATION_MODELS,
    )

    gestor = manager(tmp_path, SeparadorFalso())
    job = job_de_karaoke(tmp_path)

    salida = asyncio.run(gestor._separate_instrumental(job))

    esperado = SEPARATION_MODELS[DEFAULT_SEPARATION_MODEL].main_stem.id
    assert salida.name == f"{esperado}.wav"
