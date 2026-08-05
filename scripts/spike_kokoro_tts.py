"""Spike: Kokoro (Apache 2.0, ONNX) + espeak-ng, con verificación de ida y vuelta.

Uso:
    .venv\\Scripts\\python scripts\\spike_kokoro_tts.py

Por qué este y no el anterior: `spike_tts.py` probó que generar voz acá funciona
y es rápido (0,17x tiempo real, 100% de coincidencia al transcribir de vuelta),
pero el modelo que usaba es CC-BY-NC-4.0 — NO COMERCIAL. Kokoro es Apache 2.0 y
ya viene en ONNX, que es el camino que usa el resto de la app.

El costo de Kokoro es que espera FONEMAS y no texto: su vocabulario son 115
símbolos con IPA adentro. De ahí espeak-ng.

Lo que decide este spike:
  1. ¿El ONNX de Kokoro carga y corre con onnxruntime a secas?
  2. ¿espeak-ng produce los fonemas que su vocabulario acepta?
  3. ¿El audio es HABLA INTELIGIBLE, medido transcribiéndolo de vuelta?
  4. ¿Cuánto tarda?
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import tempfile
import time
from pathlib import Path

REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
# MEDIDO 2026-08-04 con onnxruntime 1.24.4, y las dos formas de fallar son
# distintas y ninguna avisa bien:
#   - `model_q8f16.onnx` REVIENTA el proceso (segfault) al crear la sesion. No
#     tira excepcion: se lleva el proceso entero.
#   - `model_fp16.onnx` carga y corre, devuelve audio de la duracion correcta,
#     y ese audio es TODO NaN. Es el modo de fallo mas caro: nada falla.
#   - `model.onnx` (fp32) devuelve audio real: rms 0.069, pico 0.52.
ONNX_FILE = "onnx/model.onnx"
VOICE_FILE = "voices/af_heart.bin"
ASR_REPO = "onnx-community/whisper-tiny.en"
FRASE = "The quick brown fox jumps over the lazy dog"
SAMPLE_RATE = 24000


def normalizar(texto: str) -> str:
    return re.sub(r"[^a-z ]", "", texto.lower()).strip()


def fonemizar(texto: str) -> str:
    import espeakng_loader
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    backend = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True)
    return backend.phonemize([texto])[0].strip()


def main() -> int:
    try:
        import numpy as np
        import onnxruntime as ort
        import soundfile as sf
        from huggingface_hub import hf_hub_download
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperProcessor
    except ImportError as exc:
        print(f"FALTA UNA DEPENDENCIA: {exc}")
        return 2

    print("--- 1) Fonemas con espeak-ng ---")
    try:
        fonemas = fonemizar(FRASE)
    except Exception as exc:  # noqa: BLE001
        print(f"espeak-ng fallo: {type(exc).__name__}: {exc}")
        return 1
    print(f"{FRASE!r}\n  -> {fonemas!r}")

    print(f"\n--- 2) Bajando {REPO} ---")
    model_path = hf_hub_download(REPO, ONNX_FILE)
    voice_path = hf_hub_download(REPO, VOICE_FILE)
    tokenizer_path = hf_hub_download(REPO, "tokenizer.json")
    vocab = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))["model"]["vocab"]

    desconocidos = sorted({ch for ch in fonemas if ch not in vocab})
    print(f"simbolos fuera del vocabulario: {desconocidos or 'ninguno'}")
    tokens = [vocab[ch] for ch in fonemas if ch in vocab]
    print(f"tokens: {len(tokens)}")

    print("\n--- 3) Corriendo el ONNX ---")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    entradas = {i.name: i.shape for i in session.get_inputs()}
    print(f"entradas del modelo: {entradas}")

    # El modelo declara `style` con forma [1, 256]: una fila por longitud de
    # secuencia, y se elige la que corresponde a este texto.
    voces = np.fromfile(voice_path, dtype=np.float32).reshape(-1, 256)
    style = voces[len(tokens) : len(tokens) + 1]

    # Kokoro espera la secuencia entre tokens de borde (0).
    input_ids = np.array([[0, *tokens, 0]], dtype=np.int64)
    started = time.perf_counter()
    audio = session.run(
        None,
        {
            "input_ids": input_ids,
            "style": style,
            "speed": np.array([1.0], dtype=np.float32),
        },
    )[0]
    elapsed = time.perf_counter() - started

    audio = np.asarray(audio).squeeze()
    duracion = len(audio) / SAMPLE_RATE
    work = Path(tempfile.mkdtemp(prefix="kokoro-spike-"))
    wav_path = work / "kokoro.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    print(f"audio: {duracion:.2f} s, generado en {elapsed:.2f} s")
    print(f"factor de tiempo real: {elapsed / max(duracion, 0.001):.2f}x")

    if duracion < 0.5:
        print("VEREDICTO: audio demasiado corto para esa frase. Algo salio mal.")
        return 1

    print(f"\n--- 4) Verificacion de ida y vuelta con {ASR_REPO} ---")
    processor = WhisperProcessor.from_pretrained(ASR_REPO)
    asr = ORTModelForSpeechSeq2Seq.from_pretrained(ASR_REPO, use_merged=False)
    objetivo = int(len(audio) * 16000 / SAMPLE_RATE)
    remuestreado = np.interp(
        np.linspace(0, len(audio) - 1, objetivo), np.arange(len(audio)), audio
    ).astype("float32")
    features = processor(remuestreado, sampling_rate=16000, return_tensors="pt").input_features
    ids = asr.generate(features)
    vuelta = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    print(f"lo que volvio: {vuelta!r}")

    parecido = difflib.SequenceMatcher(None, normalizar(FRASE), normalizar(vuelta)).ratio()
    print(f"parecido con lo pedido: {parecido:.0%}")

    print("\n================ VEREDICTO ================")
    print(f"wav: {wav_path}")
    if parecido >= 0.8:
        print("KOKORO SIRVE. Licencia Apache 2.0, corre por ONNX, habla inteligible.")
        return 0
    print("El audio sale pero no dice lo que se pidio. Revisar tokenizado o estilo.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
