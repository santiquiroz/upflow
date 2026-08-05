"""Agrega una pista de subtitulos al contenedor SIN re-encodear el video.

Muxeo suave: el video y el audio se copian tal cual y solo se suma la pista de
texto. Es instantaneo y sin perdida. Quemar los subtitulos en la imagen es la
otra opcion, pero obliga a recodificar el video entero — mas lento y con
perdida, asi que no puede ser el default.

Los invariantes de ffmpeg que este repo ya pago caro (ver
tests/test_audio_subtitle_mux.py) valen igual aca: TODOS los `-i` van antes de
cualquier `-map`, y el video se mapea explicito o desaparece sin aviso.
"""

from __future__ import annotations

from pathlib import Path

# mp4 no puede llevar SubRip: necesita `mov_text`. Pedirle `-c:s srt` a un mp4
# falla con "Subtitle codec 0x94a13 is not supported".
SUBTITLE_CODEC_BY_CONTAINER: dict[str, str] = {
    "mkv": "srt",
    "mp4": "mov_text",
    "mov": "mov_text",
}

_DEFAULT_SUBTITLE_CODEC = "srt"


def subtitle_codec_for(container: str) -> str:
    return SUBTITLE_CODEC_BY_CONTAINER.get(container.lower(), _DEFAULT_SUBTITLE_CODEC)


def build_subtitle_mux_command(
    *,
    ffmpeg: str,
    video: Path,
    subtitles: Path,
    destination: Path,
    container: str,
    language: str | None,
) -> list[str]:
    if not container.strip():
        raise ValueError("Hace falta un contenedor para elegir el codec de subtitulos.")

    command = [
        str(ffmpeg),
        # Sin `-y` ffmpeg espera confirmacion y el job cuelga.
        "-y",
        "-i",
        str(video),
        "-i",
        str(subtitles),
        # Todos los -map DESPUES de todos los -i, o ffmpeg responde
        # "Option map cannot be applied to input url".
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-map",
        "1:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-c:s",
        subtitle_codec_for(container),
    ]
    if language:
        # Sin esto el reproductor muestra la pista como "Track 1" y no como el
        # idioma que es.
        #
        # Medido con ffmpeg real (2026-08-04): la etiqueta QUEDA en mkv
        # (`ffprobe` devuelve `subrip,es`) pero NO en mp4, que reporta la pista
        # sin idioma. Es una limitacion del contenedor, no del comando: mkv es
        # el que conserva bien la metadata de pista.
        command += ["-metadata:s:s:0", f"language={language}"]
    command.append(str(destination))
    return command
