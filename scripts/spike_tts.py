"""Spike: ¿se puede generar voz desde texto en esta máquina, y suena bien?

Uso:
    .venv\\Scripts\\python scripts\\spike_tts.py

La pregunta NO es "existe la clase". Este repo ya se quemó con eso: Whisper
sobre DirectML cargaba bien y devolvía texto basura sin fallar.

La verificación fuerte es de IDA Y VUELTA: se genera habla desde una frase
conocida y se transcribe de vuelta con el Whisper que la app ya usa. Si lo que
vuelve se parece a lo que se pidió, el audio es habla de verdad y no ruido con
la duración correcta.

Preguntas que decide:
  1. ¿Corre un modelo de TTS acá, sin phonemizador externo (espeak-ng)?
  2. ¿El audio que sale es HABLA INTELIGIBLE, medido por transcripción?
  3. ¿Cuánto tarda? Un TTS que tarda más que el audio que produce no sirve
     para doblaje.
"""

from __future__ import annotations

import difflib
import re
import sys
import tempfile
import time
from pathlib import Path

# Char-level para inglés: NO necesita espeak-ng, a diferencia de Piper y Kokoro.
TTS_REPO = "facebook/mms-tts-eng"
ASR_REPO = "onnx-community/whisper-tiny.en"
FRASE = "The quick brown fox jumps over the lazy dog"


def normalizar(texto: str) -> str:
    return re.sub(r"[^a-z ]", "", texto.lower()).strip()


def main() -> int:
    try:
        import soundfile as sf
        import torch
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import AutoTokenizer, VitsModel, WhisperProcessor
    except ImportError as exc:
        print(f"FALTA UNA DEPENDENCIA: {exc}")
        return 2

    work = Path(tempfile.mkdtemp(prefix="tts-spike-"))
    wav_path = work / "generado.wav"

    print(f"--- 1) Cargando {TTS_REPO} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TTS_REPO)
        model = VitsModel.from_pretrained(TTS_REPO)
    except Exception as exc:  # noqa: BLE001
        print(f"NO CARGA: {type(exc).__name__}: {exc}")
        return 1

    print(f"--- 2) Generando: {FRASE!r} ---")
    inputs = tokenizer(FRASE, return_tensors="pt")
    started = time.perf_counter()
    with torch.no_grad():
        output = model(**inputs).waveform
    elapsed = time.perf_counter() - started

    audio = output.squeeze().cpu().numpy()
    sample_rate = model.config.sampling_rate
    sf.write(str(wav_path), audio, sample_rate)
    duracion = len(audio) / sample_rate
    print(f"audio: {duracion:.2f} s a {sample_rate} Hz, generado en {elapsed:.2f} s")
    print(f"factor de tiempo real: {elapsed / max(duracion, 0.001):.2f}x (menos de 1 es mas rapido que reproducirlo)")

    if duracion < 0.5:
        print("VEREDICTO: el audio es demasiado corto para ser esa frase. Algo salio mal.")
        return 1

    print(f"\n--- 3) Verificacion de ida y vuelta con {ASR_REPO} ---")
    processor = WhisperProcessor.from_pretrained(ASR_REPO)
    asr = ORTModelForSpeechSeq2Seq.from_pretrained(ASR_REPO, use_merged=False)

    leido, rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
    mono = leido.mean(axis=1)
    if rate != 16000:
        import numpy as np

        objetivo = int(len(mono) * 16000 / rate)
        mono = np.interp(
            np.linspace(0, len(mono) - 1, objetivo), np.arange(len(mono)), mono
        ).astype("float32")

    features = processor(mono, sampling_rate=16000, return_tensors="pt").input_features
    ids = asr.generate(features)
    vuelta = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    print(f"lo que volvio: {vuelta!r}")

    parecido = difflib.SequenceMatcher(None, normalizar(FRASE), normalizar(vuelta)).ratio()
    print(f"parecido con lo pedido: {parecido:.0%}")

    print("\n================ VEREDICTO ================")
    print(f"wav: {wav_path}")
    if parecido >= 0.8:
        print("ES HABLA INTELIGIBLE. El camino de TTS sirve.")
        return 0
    if parecido >= 0.4:
        print("Habla, pero se entiende a medias. Sirve como base; hay que revisar calidad.")
        return 0
    print("NO es habla reconocible. El audio sale, pero no dice lo que se pidio.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
