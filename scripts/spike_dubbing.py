"""Spike: doblaje automático de punta a punta, con modelos reales.

Uso:
    .venv\\Scripts\\python scripts\\spike_dubbing.py

Las unidades ya están probadas con dobles. Esto mide la CADENA real: traducir
con OPUS-MT, sintetizar con Kokoro, acomodar cada línea en su hueco y muxear con
ffmpeg. Un doblaje puede fallar de tres maneras que un test con dobles no ve:

  - la pista sale en silencio (el modelo devolvió algo, pero nada suena)
  - la voz aparece en el momento equivocado (todo el sentido del doblaje)
  - el video sale con una sola pista de audio (el muxeo se comió el original)

Por eso el veredicto sale de MEDIR el archivo producido, no de que ffmpeg
devuelva cero.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.config import get_settings  # noqa: E402
from app.services.dub_mux import build_dub_mux_command  # noqa: E402
from app.services.dubbing_pipeline import DubbingPipeline  # noqa: E402
from app.services.engines.tts_kokoro import SAMPLE_RATE as TTS_SAMPLE_RATE  # noqa: E402
from app.services.engines.tts_kokoro import KokoroTtsEngine  # noqa: E402
from app.services.phonemize import text_to_phonemes  # noqa: E402
from app.services.subtitles import TranscriptSegment  # noqa: E402
from app.services.translate import TranslationEngine  # noqa: E402
from app.services.engines.tts_kokoro import available_voices  # noqa: E402
from app.services.vendor_paths import kokoro_dir, translation_dir  # noqa: E402

# Dos líneas separadas por un hueco de silencio: si el doblaje respeta los
# tiempos, el silencio del medio tiene que seguir estando.
SEGMENTS = [
    TranscriptSegment(start=0.5, end=3.0, text="The quick brown fox jumps over the lazy dog."),
    TranscriptSegment(start=7.0, end=9.5, text="This model runs on your own graphics card."),
]
TOTAL_SECONDS = 11.0


def rms(bloque: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(bloque)))) if bloque.size else 0.0


def main() -> int:
    settings = get_settings()
    ffmpeg = str(settings.ffmpeg_binary_path)

    pipeline = DubbingPipeline(
        translation=TranslationEngine(translation_dir(settings)),
        tts=KokoroTtsEngine(settings),
        tts_model_dir=kokoro_dir(settings),
        phonemize=text_to_phonemes,
        sample_rate=TTS_SAMPLE_RATE,
        available_voices=available_voices(kokoro_dir(settings)),
    )
    print(f"voces instaladas: {available_voices(kokoro_dir(settings))}")

    print("Traduciendo y sintetizando ...")
    resultado = pipeline.build_track(
        SEGMENTS, source_language="en", target_language="es", total_seconds=TOTAL_SECONDS
    )
    pista = resultado.track
    print(f"pista de {len(pista) / TTS_SAMPLE_RATE:.1f} s, {resultado.overflowing} línea(s) fuera de hueco")

    if rms(pista) < 1e-4:
        print("\nLA PISTA SALIO EN SILENCIO: el doblaje no dice nada.")
        return 1

    # ¿Cada línea cayó donde tenía que caer?
    def rms_entre(desde: float, hasta: float) -> float:
        return rms(pista[int(desde * TTS_SAMPLE_RATE) : int(hasta * TTS_SAMPLE_RATE)])

    voz_1 = rms_entre(0.5, 3.0)
    silencio = rms_entre(4.0, 6.5)
    voz_2 = rms_entre(7.0, 9.5)
    print(f"\nRMS donde va la primera línea : {voz_1:.4f}")
    print(f"RMS en el hueco de silencio    : {silencio:.4f}")
    print(f"RMS donde va la segunda línea  : {voz_2:.4f}")

    if voz_1 < 0.01 or voz_2 < 0.01:
        print("\nHAY VOZ, PERO NO DONDE VA: alguna línea no cayó en su hueco.")
        return 1
    if silencio > min(voz_1, voz_2) / 4:
        print("\nEL SILENCIO DEL MEDIO NO ESTA: la voz se corrió de sus tiempos.")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        carpeta = Path(tmp)
        video = carpeta / "origen.mp4"
        voz = carpeta / "doblaje.wav"
        salida = carpeta / "doblado.mkv"

        import soundfile

        soundfile.write(voz, pista, TTS_SAMPLE_RATE, format="WAV", subtype="PCM_16")
        subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i",
             f"color=c=blue:s=320x240:d={TOTAL_SECONDS}:r=10", "-f", "lavfi", "-i",
             f"sine=frequency=200:duration={TOTAL_SECONDS}", "-c:v", "libx264", "-c:a",
             "aac", str(video)],
            check=True,
        )

        comando = build_dub_mux_command(
            ffmpeg=ffmpeg, video=video, dubbed_audio=voz, destination=salida, language="es"
        )
        proceso = subprocess.run(comando, capture_output=True)
        if proceso.returncode != 0:
            print("\nEL MUXEO FALLO:", proceso.stderr.decode("utf-8", "replace")[-500:])
            return 1

        sonda = subprocess.run(
            [str(Path(ffmpeg).with_name("ffprobe.exe")), "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index:stream_tags=language:disposition=default",
             "-of", "json", str(salida)],
            capture_output=True, check=True,
        )
        pistas = json.loads(sonda.stdout)["streams"]
        print(f"\npistas de audio en el archivo final: {len(pistas)}")
        for i, pista_info in enumerate(pistas):
            idioma = (pista_info.get("tags") or {}).get("language", "sin idioma")
            por_defecto = (pista_info.get("disposition") or {}).get("default", 0)
            print(f"  #{i}: idioma={idioma} por_defecto={por_defecto}")

        if len(pistas) < 2:
            print("\nSE PERDIO EL AUDIO ORIGINAL: el doblaje lo reemplazó en vez de sumarse.")
            return 1
        if not (pistas[0].get("disposition") or {}).get("default", 0):
            print("\nLA PISTA DOBLADA NO QUEDO POR DEFECTO: el reproductor arrancaría con el original.")
            return 1

    print("\n================ VEREDICTO ================")
    print("DOBLA DE VERDAD: hay voz en su hueco, silencio donde iba el silencio,")
    print("y el archivo final conserva las dos pistas con la doblada por defecto.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
