"""Un pedazo del medio de un audio, para probar algo sin pagar el tema entero.

Del MEDIO y no del arranque: las intros suelen ser instrumentales, y juzgar un
separador de voces en una parte sin voces no dice nada. El medio de una cancion
es donde estan las dos cosas a la vez, que es justo el caso dificil.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# 30 s alcanzan para oir la diferencia entre dos separadores y son ~8x mas
# baratos que un tema de 4 minutos, que es el punto de probar antes.
DEFAULT_EXCERPT_SECONDS = 30


def build_duration_probe_command(ffprobe_path: Path, source_path: Path) -> list[str]:
    return [
        str(ffprobe_path),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(source_path),
    ]


def parse_duration_seconds(probe_json: dict[str, Any]) -> float | None:
    try:
        duration = float(probe_json.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        return None
    return duration if duration > 0 else None


def centered_offset(duration_seconds: float | None, excerpt_seconds: int) -> float:
    """Donde empezar para que el fragmento quede centrado.

    Sin duracion —un stream, un contenedor raro— se arranca de cero: es peor
    lugar para juzgar, pero inventar un offset mas alla del final da un archivo
    vacio, que no es peor sino inservible.
    """
    if duration_seconds is None or duration_seconds <= excerpt_seconds:
        return 0.0
    return round((duration_seconds - excerpt_seconds) / 2, 3)


def build_excerpt_command(
    *,
    ffmpeg: Path,
    source: Path,
    destination: Path,
    offset_seconds: float,
    excerpt_seconds: int,
) -> list[str]:
    return [
        str(ffmpeg),
        "-y",
        # `-ss` ANTES de `-i`: asi ffmpeg salta al punto en vez de decodificar
        # todo lo anterior y tirarlo. En un tema de 4 minutos es la diferencia
        # entre cortar al instante y cortar en segundos.
        "-ss",
        str(offset_seconds),
        "-i",
        str(source),
        "-t",
        str(excerpt_seconds),
        # Sin video: la fuente puede ser un mp4 bajado de una URL.
        "-vn",
        # WAV y no el formato de origen: el fragmento es una entrada intermedia
        # que se descarta, y recomprimirlo agregaria un artefacto a lo que
        # justamente se esta comparando.
        "-acodec",
        "pcm_s16le",
        str(destination),
    ]
