"""Decodificado a WAV para las rutas que leen audio con `soundfile`.

`soundfile` (libsndfile) entiende WAV/FLAC/OGG/MP3 y varios mas, pero NO
contenedores de video ni AAC/M4A. La ruta de transcripcion aceptaba un .mp4 sin
chistar, lo encolaba, tomaba el semaforo del dispositivo y recien ahi reventaba
con un error crudo de libsndfile.

ffmpeg ya viaja con la app y `audio_pipeline._decode_to_wav` ya prueba este
camino; aca se extrae para que la transcripcion lo use tal cual.
"""

from __future__ import annotations

from pathlib import Path

# Lo que libsndfile lee directo. Se decodifica todo lo demas — incluido lo
# desconocido: mandarlo a ffmpeg le da una chance de andar, rechazarlo no.
SOUNDFILE_READABLE = frozenset({".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".aiff", ".aif", ".w64", ".caf"})


def needs_decoding(source: Path) -> bool:
    return source.suffix.lower() not in SOUNDFILE_READABLE


def build_decode_to_wav_command(
    *, ffmpeg: str, source: Path, destination: Path, sample_rate: int
) -> list[str]:
    return [
        str(ffmpeg),
        # Sin `-y` ffmpeg se queda esperando confirmacion y el job cuelga.
        "-y",
        "-i",
        str(source),
        # El modelo toma un solo canal a 16 kHz; convertir aca evita que el
        # remuestreo lo haga numpy despues, mas lento y con peor calidad.
        "-vn",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(destination),
    ]
