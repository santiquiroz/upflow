"""Spike: ¿la voz doblada DICE lo traducido, o solo suena a algo?

Uso:
    .venv\Scripts\python scripts\spike_dub_content.py

Medido el 2026-08-05: 99% de parecido entre lo que se le pidió decir y lo que
Whisper escuchó de vuelta.

Los chequeos de RMS del spike de doblaje prueban que hay voz y que cae en su
hueco. No prueban que diga palabras: un fonemizador aplicando reglas del idioma
equivocado produce ruido con la energia y la duracion correctas.

Vuelta completa: texto ingles -> traduccion -> voz -> Whisper multilingue ->
texto. Si lo que vuelve se parece a la traduccion, la voz dice lo que tiene que
decir.
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.config import get_settings  # noqa: E402
from app.services.dubbing import voice_for_language  # noqa: E402
from app.services.engines.tts_kokoro import SAMPLE_RATE, KokoroTtsEngine, available_voices  # noqa: E402
from app.services.phonemize import text_to_phonemes  # noqa: E402
from app.services.translate import LanguagePair, TranslationEngine  # noqa: E402
from app.services.vendor_paths import kokoro_dir, translation_dir  # noqa: E402

ASR = "onnx-community/whisper-small"
FRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "This model runs on your own graphics card.",
]


def normalizar(texto: str) -> str:
    return re.sub(r"[^a-záéíóúñü ]", "", texto.lower()).strip()


def main() -> int:
    settings = get_settings()
    traductor = TranslationEngine(translation_dir(settings))
    traducidas = traductor.translate(FRASES, LanguagePair(source="en", target="es"))

    voces = available_voices(kokoro_dir(settings))
    voz = voice_for_language(voces, "es")
    print(f"voces instaladas: {voces}")
    print(f"voz elegida para el espanol: {voz}")

    tts = KokoroTtsEngine(settings)
    audios = [
        np.asarray(
            tts.synthesize(
                model_dir=kokoro_dir(settings),
                phonemes=text_to_phonemes(t, "es"),
                voice=voz,
            ),
            dtype=np.float32,
        )
        for t in traducidas
    ]

    print(f"\nCargando {ASR} (multilingue) ...")
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(ASR)
    # use_merged=False: el decoder merged ya devolvio texto fluido y equivocado
    # en este repo, sin fallar.
    modelo = ORTModelForSpeechSeq2Seq.from_pretrained(ASR, use_merged=False)

    puntajes = []
    for esperado, audio in zip(traducidas, audios):
        from scipy.signal import resample_poly

        a16 = resample_poly(audio, 16000, SAMPLE_RATE).astype(np.float32)
        features = processor(a16, sampling_rate=16000, return_tensors="pt").input_features
        ids = modelo.generate(features, language="es", task="transcribe", max_new_tokens=100)
        vuelto = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        parecido = difflib.SequenceMatcher(
            None, normalizar(esperado), normalizar(vuelto)
        ).ratio()
        puntajes.append(parecido)
        print(f"\n  se le pidio decir : {esperado}")
        print(f"  Whisper escucho   : {vuelto}")
        print(f"  parecido          : {parecido:.0%}")

    promedio = sum(puntajes) / len(puntajes)
    print("\n================ VEREDICTO ================")
    print(f"parecido promedio entre lo pedido y lo escuchado: {promedio:.0%}")
    if promedio >= 0.7:
        print("LA VOZ DICE LO TRADUCIDO.")
        return 0
    if promedio >= 0.4:
        print("SE ENTIENDE A MEDIAS: revisar fonemizador o voz antes de confiar.")
        return 1
    print("NO DICE LO QUE TIENE QUE DECIR: suena a algo, pero no a eso.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
