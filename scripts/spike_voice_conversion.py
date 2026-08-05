"""Spike: conversión de voz con SpeechT5 (MIT), con verificación de ida y vuelta.

Uso:
    .venv\\Scripts\\python scripts\\spike_voice_conversion.py

Corrige un juicio apurado del 2026-08-05: se dijo que no habia modelo servible
con licencia limpia despues de mirar cuatro repos. `microsoft/speecht5_vc` es
MIT, hace conversion de voz de verdad y transformers lo corre nativo.

La verificacion correcta para CONVERSION no es la misma que para TTS. En TTS se
pregunta "¿dice lo que le pedi?". Aca hay DOS preguntas, y las dos importan:
  1. ¿El CONTENIDO sobrevive? (las palabras tienen que ser las mismas)
  2. ¿La VOZ cambio? (si no cambia, no convirtio nada)

Un modelo que devuelve el audio de entrada intacto pasaria la primera y fallaria
la segunda. Por eso se miden las dos.
"""

from __future__ import annotations

import difflib
import re
import sys
import tempfile
import time
from pathlib import Path

VC_REPO = "microsoft/speecht5_vc"
VOCODER_REPO = "microsoft/speecht5_hifigan"
ASR_REPO = "onnx-community/whisper-tiny.en"
FRASE = "The quick brown fox jumps over the lazy dog"
SAMPLE_RATE = 16000


def normalizar(texto: str) -> str:
    return re.sub(r"[^a-z ]", "", texto.lower()).strip()


def transcribir(audio, rate: int) -> str:
    import numpy as np
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(ASR_REPO)
    asr = ORTModelForSpeechSeq2Seq.from_pretrained(ASR_REPO, use_merged=False)
    if rate != 16000:
        objetivo = int(len(audio) * 16000 / rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, objetivo), np.arange(len(audio)), audio
        ).astype("float32")
    features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    ids = asr.generate(features)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main() -> int:
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from transformers import (
            SpeechT5ForSpeechToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )
    except ImportError as exc:
        print(f"FALTA UNA DEPENDENCIA: {exc}")
        return 2

    work = Path(tempfile.mkdtemp(prefix="vc-spike-"))

    print("--- 1) Generando la voz de origen con Kokoro ---")
    from app.services.engines.tts_kokoro import KokoroTtsEngine
    from app.services.phonemize import text_to_phonemes
    from app.config import Settings

    kokoro_dir = Path("vendor/kokoro")
    if not (kokoro_dir / "model.onnx").exists():
        print(f"Falta Kokoro en {kokoro_dir}. Corre scripts/download-kokoro-tts.ps1")
        return 2
    engine = KokoroTtsEngine(Settings(_env_file=None))
    origen = engine.synthesize(
        model_dir=kokoro_dir, phonemes=text_to_phonemes(FRASE, "en"), voice="af_heart"
    )
    # SpeechT5 trabaja a 16 kHz; Kokoro entrega 24 kHz.
    objetivo = int(len(origen) * SAMPLE_RATE / 24000)
    origen16 = np.interp(
        np.linspace(0, len(origen) - 1, objetivo), np.arange(len(origen)), origen
    ).astype("float32")
    sf.write(str(work / "origen.wav"), origen16, SAMPLE_RATE)
    print(f"origen: {len(origen16) / SAMPLE_RATE:.2f} s")

    print(f"\n--- 2) Cargando {VC_REPO} ---")
    try:
        processor = SpeechT5Processor.from_pretrained(VC_REPO)
        model = SpeechT5ForSpeechToSpeech.from_pretrained(VC_REPO)
        vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_REPO)
    except Exception as exc:  # noqa: BLE001
        print(f"NO CARGA: {type(exc).__name__}: {exc}")
        return 1

    # El "estilo" de la voz destino es un x-vector de 512. Para el spike alcanza
    # uno sintetico y estable: la pregunta es si la conversion CORRE y CAMBIA la
    # voz, no si clona a una persona concreta.
    generador = torch.Generator().manual_seed(7)
    speaker = torch.randn(1, 512, generator=generador)
    speaker = speaker / speaker.norm()

    print("\n--- 3) Convirtiendo ---")
    entrada = processor(audio=origen16, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    started = time.perf_counter()
    with torch.no_grad():
        convertido = model.generate_speech(
            entrada["input_values"], speaker, vocoder=vocoder
        )
    elapsed = time.perf_counter() - started
    convertido = convertido.cpu().numpy()
    sf.write(str(work / "convertido.wav"), convertido, SAMPLE_RATE)
    duracion = len(convertido) / SAMPLE_RATE
    print(f"convertido: {duracion:.2f} s en {elapsed:.2f} s ({elapsed / max(duracion, 0.001):.2f}x)")

    if not np.all(np.isfinite(convertido)):
        print("VEREDICTO: el audio convertido trae NaN. No sirve.")
        return 1

    print("\n--- 4) ¿Sobrevivio el CONTENIDO? ---")
    vuelta = transcribir(convertido, SAMPLE_RATE)
    print(f"lo que volvio: {vuelta!r}")
    parecido = difflib.SequenceMatcher(None, normalizar(FRASE), normalizar(vuelta)).ratio()
    print(f"parecido con lo pedido: {parecido:.0%}")

    print("\n--- 5) ¿CAMBIO la voz? ---")
    # Si devolviera la entrada intacta, la diferencia seria ~0. Se compara el
    # espectro promedio, que es donde vive el timbre.
    n = min(len(origen16), len(convertido))
    esp_origen = np.abs(np.fft.rfft(origen16[:n]))
    esp_conv = np.abs(np.fft.rfft(convertido[:n]))
    esp_origen /= esp_origen.sum() or 1
    esp_conv /= esp_conv.sum() or 1
    diferencia = float(np.abs(esp_origen - esp_conv).sum())
    print(f"diferencia espectral: {diferencia:.3f} (0 = audio identico)")

    print("\n================ VEREDICTO ================")
    print(f"origen:     {work / 'origen.wav'}")
    print(f"convertido: {work / 'convertido.wav'}")
    contenido_ok = parecido >= 0.6
    voz_cambio = diferencia > 0.1
    if contenido_ok and voz_cambio:
        print("CONVIERTE: el contenido sobrevive y el timbre cambio. Licencia MIT.")
        return 0
    if not voz_cambio:
        print("NO CONVIRTIO: el audio salio practicamente igual al de entrada.")
        return 1
    print("CONVIRTIO PERO SE COMIO EL CONTENIDO: la voz cambio y las palabras se perdieron.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
