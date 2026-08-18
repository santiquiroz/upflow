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


# Los separadores trabajan a 44100 estereo. No es negociable como el 16k mono de
# la transcripcion: entrarle otra cosa al grafo lo hace remuestrear por dentro.
SEPARATION_SAMPLE_RATE = 44100
SEPARATION_CHANNELS = 2


def build_decode_to_wav_command(
    *, ffmpeg: str, source: Path, destination: Path, sample_rate: int, channels: int = 1
) -> list[str]:
    return [
        str(ffmpeg),
        # Sin `-y` ffmpeg se queda esperando confirmacion y el job cuelga.
        "-y",
        "-i",
        str(source),
        # Convertir aca y no despues: el remuestreo en numpy es mas lento y de
        # peor calidad, y ffmpeg tiene el downmix canonico del surround.
        "-vn",
        "-ac",
        str(channels),
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        str(destination),
    ]
