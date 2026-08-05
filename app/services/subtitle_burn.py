"""Quema los subtitulos EN la imagen, re-encodeando el video.

Es la alternativa al muxeo suave (`subtitle_mux.py`): ahi el texto viaja como
pista aparte y el reproductor decide si mostrarlo; aca queda pintado en los
pixeles y se ve en cualquier lado, incluidas las redes sociales, que descartan
las pistas de texto. El costo es real: re-encodear el video entero, con la
perdida que eso implica. Por eso el muxeo suave sigue siendo el default.

El punto delicado es la RUTA del .srt: viaja DENTRO de la cadena del filtro,
donde `:` separa opciones, `\\` escapa y `'` delimita. Una ruta de Windows tal
cual rompe el parser de ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

# 18 da una copia visualmente cercana al original sin inflar el archivo. Dejar
# que ffmpeg elija su default (23) se nota en los degradados.
DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"


def escape_subtitles_filter_path(path: Path) -> str:
    """Deja la ruta como la entiende el filtro `subtitles`.

    Tres reemplazos, en este orden: las barras invertidas pasan a barras
    normales (ffmpeg las acepta en Windows y asi no hay escapes que interpretar),
    los dos puntos de la unidad se escapan porque separan opciones del filtro, y
    las comillas simples se escapan porque cierran la cadena.
    """
    texto = str(path).replace("\\", "/")
    texto = texto.replace(":", r"\:")
    return texto.replace("'", r"\'")


def build_subtitle_burn_command(
    *,
    ffmpeg: str,
    video: Path,
    subtitles: Path,
    destination: Path,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
) -> list[str]:
    if Path(destination) == Path(video):
        raise ValueError("El destino no puede ser el mismo archivo de origen.")

    filtro = f"subtitles='{escape_subtitles_filter_path(subtitles)}'"
    return [
        str(ffmpeg),
        # Sin `-y` ffmpeg espera confirmacion y el job cuelga.
        "-y",
        "-i",
        str(video),
        "-vf",
        filtro,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        # El audio no se toca: solo cambia la imagen.
        "-c:a",
        "copy",
        str(destination),
    ]
