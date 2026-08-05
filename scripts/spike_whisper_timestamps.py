"""Spike: ¿este camino de Whisper devuelve MARCAS DE TIEMPO?

Uso:
    .venv\\Scripts\\python scripts\\spike_whisper_timestamps.py

Por qué existe: sin timestamps no hay subtítulos, solo transcripción. El
docstring de `split_into_chunks` dice que `return_timestamps=True` no devuelve
marcas en este camino (medido 2026-07-29), y esa sola línea decide si la
capacidad `video.subtitles` es construible o no.

Esa medición es de hace meses y de otras versiones. Antes de dar por muerta la
feature se vuelve a medir, con el MISMO modelo verificado en el spike anterior
(`onnx-community/whisper-tiny.en`) y con voz real, no un tono.

La pregunta NO es "existe el parámetro". Es:
  1. ¿`generate(return_timestamps=True)` devuelve tokens de tiempo?
  2. Si se decodifica con `decode(..., decode_with_timestamps=True)`, ¿aparecen
     las marcas `<|x.xx|>` en el texto?
  3. ¿El procesador puede convertirlas en offsets con `chunk_length_s`?

Cualquiera de las tres que funcione alcanza para construir subtítulos.
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_REPO = "onnx-community/whisper-tiny.en"
# Misma muestra de voz real que uso el spike anterior: transcripcion conocida.
SAMPLE_URL = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"


def main() -> int:
    try:
        import numpy as np
        import soundfile as sf
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperProcessor
    except ImportError as exc:
        print(f"FALTA UNA DEPENDENCIA: {exc}")
        return 2

    work = Path(tempfile.mkdtemp(prefix="whisper-ts-"))
    audio_path = work / "sample.flac"
    print(f"Bajando la muestra de voz real a {audio_path} ...")
    try:
        urllib.request.urlretrieve(SAMPLE_URL, audio_path)
    except Exception as exc:  # noqa: BLE001 - el spike reporta, no maneja
        print(f"NO SE PUDO BAJAR LA MUESTRA: {exc}")
        return 2

    audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    print(f"Muestra: {len(mono) / sample_rate:.1f} s a {sample_rate} Hz")

    print(f"Cargando {MODEL_REPO} (use_merged=False, CPU) ...")
    processor = WhisperProcessor.from_pretrained(MODEL_REPO)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(MODEL_REPO, use_merged=False)

    features = processor(
        mono, sampling_rate=sample_rate, return_tensors="pt"
    ).input_features

    print("\n--- 1) generate(return_timestamps=True) ---")
    ids = model.generate(features, return_timestamps=True)
    plain = processor.batch_decode(ids, skip_special_tokens=True)[0]
    print(f"texto: {plain!r}")

    print("\n--- 2) decode_with_timestamps=True ---")
    with_marks = processor.batch_decode(
        ids, skip_special_tokens=False, decode_with_timestamps=True
    )[0]
    print(f"crudo: {with_marks!r}")
    has_marks = "<|" in with_marks and any(
        ch.isdigit() for ch in with_marks.split("<|", 1)[-1][:6]
    )
    print(f"¿trae marcas de tiempo?: {has_marks}")

    print("\n--- 3) offsets por el procesador ---")
    try:
        decoded = processor.batch_decode(
            ids, skip_special_tokens=True, output_offsets=True
        )[0]
        offsets = decoded.get("offsets") if isinstance(decoded, dict) else None
        print(f"offsets: {offsets}")
        usable = bool(offsets)
    except Exception as exc:  # noqa: BLE001
        print(f"output_offsets fallo: {exc}")
        usable = False

    print("\n================ VEREDICTO ================")
    if has_marks or usable:
        print("HAY TIMESTAMPS. La capacidad de subtitulos es construible en este camino.")
        return 0
    print("SIN TIMESTAMPS. Confirmado lo medido el 2026-07-29.")
    print("Subtitulos con tiempos necesitaria otro runtime o una pasada de alineado.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
