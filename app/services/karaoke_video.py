"""El video de karaoke: instrumental + letra sincronizada pintada en la imagen.

Las tres piezas ya existian sueltas —separar, transcribir con tiempos, quemar
subtitulos— y armar el karaoke a mano era hacer tres viajes y ademas resolver el
empalme: pegarle al video la pista instrumental EN VEZ de la original, que es lo
que ninguno de los tres pasos hace solo.

Un solo ffmpeg para todo: el filtro de subtitulos y el reemplazo de audio en la
misma pasada. En dos pasadas el video se re-encodearia dos veces, y el
re-encodeado es justo lo caro.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.subtitle_burn import (
    DEFAULT_CRF,
    DEFAULT_PRESET,
    escape_subtitles_filter_path,
)

# Un fondo sobrio y oscuro: la letra va en blanco con borde, y sobre un fondo
# claro no se lee. No es negro puro para que no parezca un video roto.
BACKGROUND_COLOR = "0x14141a"
BACKGROUND_SIZE = "1280x720"
# 24 fps sobre un fondo fijo: nada se mueve salvo el texto, y mas cuadros solo
# agrandan el archivo.
BACKGROUND_FPS = 24


def build_picture_probe_command(ffprobe_path: Path, source_path: Path) -> list[str]:
    return [
        str(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type:stream_disposition=attached_pic",
        "-of",
        "json",
        str(source_path),
    ]


def has_real_picture(probe_json: dict[str, Any]) -> bool:
    """Si hay imagen que sirva de fondo, o solo la caratula del mp3.

    La caratula ES un stream de video para ffprobe, pero de UN cuadro: usarla de
    fondo da un video de un cuadro con la letra encima y la cancion cortada.
    """
    streams = probe_json.get("streams") or []
    if not streams:
        return False
    stream = streams[0]
    if str(stream.get("codec_type")) != "video":
        return False
    disposition = stream.get("disposition") or {}
    return not int(disposition.get("attached_pic") or 0)


# Que puede ser el fondo del karaoke. "source" es el video original del que
# salio el audio; "image" y "video" son archivos que el usuario eligio; y
# "generated" es el fondo liso que fabrica ffmpeg cuando no hay nada mejor.
BACKGROUND_KINDS = ("source", "image", "video", "generated")


def _background_input(
    kind: str, background: Path | None, duration_seconds: float
) -> list[str]:
    if kind == "generated":
        return [
            "-f",
            "lavfi",
            # La duracion va en la ENTRADA y no al final: `color` es una fuente
            # infinita, y sin cortarla aca ffmpeg genera cuadros para siempre.
            "-t",
            str(duration_seconds),
            "-i",
            f"color=c={BACKGROUND_COLOR}:s={BACKGROUND_SIZE}:r={BACKGROUND_FPS}",
        ]
    if background is None:
        raise ValueError(f"El fondo {kind!r} necesita un archivo y no llego ninguno.")
    if kind == "image":
        # Una imagen fija no tiene duracion: `-loop 1` la vuelve una fuente
        # continua y `-t` la corta al largo del instrumental.
        return [
            "-loop",
            "1",
            "-framerate",
            str(BACKGROUND_FPS),
            "-t",
            str(duration_seconds),
            "-i",
            str(background),
        ]
    if kind == "video":
        # Un video de fondo mas corto que la cancion se repite (`-stream_loop
        # -1` = infinito); el `-shortest` de salida corta en el instrumental.
        return ["-stream_loop", "-1", "-i", str(background)]
    if kind == "source":
        return ["-i", str(background)]
    raise ValueError(
        f"Fondo desconocido: {kind!r}. Validos: {', '.join(BACKGROUND_KINDS)}."
    )


def build_practice_mix_command(
    *,
    ffmpeg: str,
    instrumental: Path,
    vocals: Path,
    destination: Path,
) -> list[str]:
    """La pista de ensayo: instrumental + voces (ya gateadas) en un FLAC.

    `normalize=0` porque amix por default divide el nivel por la cantidad de
    entradas: la mezcla de practica tiene que sonar al nivel de la cancion,
    no a la mitad. `duration=longest` para no cortar la cola del instrumental
    si las voces gateadas terminan antes.
    """
    return [
        str(ffmpeg),
        "-y",
        "-i",
        str(instrumental),
        "-i",
        str(vocals),
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0[a]",
        "-map",
        "[a]",
        "-c:a",
        "flac",
        str(destination),
    ]


def build_karaoke_command(
    *,
    ffmpeg: str,
    background_kind: str = "generated",
    background: Path | None = None,
    duration_seconds: float,
    instrumental: Path,
    subtitles: Path,
    destination: Path,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
) -> list[str]:
    """Una sola pasada: fondo + letra quemada + instrumental.

    El fondo puede ser el video original, una imagen o video elegidos, o el
    liso que genera ffmpeg mismo (`lavfi`), que evita escribir un archivo de
    video liso a disco solo para volver a leerlo.
    """
    if background is not None and Path(destination) == Path(background):
        raise ValueError("El destino no puede ser el mismo archivo de origen.")

    entrada_video = _background_input(background_kind, background, duration_seconds)

    return [
        str(ffmpeg),
        "-y",
        *entrada_video,
        "-i",
        str(instrumental),
        "-filter_complex",
        f"[0:v]subtitles='{escape_subtitles_filter_path(subtitles)}'[v]",
        "-map",
        "[v]",
        # El audio sale de la ENTRADA 1: el punto del karaoke es que la voz
        # original no este. Sin este map explicito ffmpeg toma la del video.
        "-map",
        "1:a",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        # El video original puede durar mas o menos que el instrumental (un
        # video con intro larga, un corte). Sin esto queda cola muda o audio sin
        # imagen.
        "-shortest",
        str(destination),
    ]
