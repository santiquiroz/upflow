"""Spike de viabilidad: separacion de stems con MDX-Net sobre ONNX Runtime.

Uso:
    .venv\\Scripts\\python scripts\\spike_stems_onnx.py
    .venv\\Scripts\\python scripts\\spike_stems_onnx.py --separate

Ya se midio que el modelo CARGA con onnxruntime crudo, en DirectML y en CPU (ver
docs/.../2026-07-29-stems-onnx-spike-findings.md). Cargar NO es funcionar: con
whisper, DirectML cargaba, corria sin excepcion y devolvia basura. Este spike
responde la pregunta que falta.

Tres controles, en orden. Cada uno invalida al siguiente si falla:

  1. ROUND-TRIP de la STFT, SIN el modelo. Si mi STFT seguida de su inversa no
     reconstruye el audio, cualquier resultado del modelo no significa nada porque
     no se sabe si el error es del modelo o mio. Es el mismo tipo de control que
     con `bert` atrapo el falso negativo del exportador de whisper.
  2. La sesion corre y devuelve un tensor de la forma esperada.
  3. CALIDAD sobre audio real. Se usa una grabacion de VOZ SOLA con un modelo que
     apunta a la voz: si funciona, la salida tiene que parecerse mucho al original
     (correlacion alta), porque no hay instrumental que sacar. Si devuelve basura,
     la correlacion se derrumba. Es una prueba falsable, no un "parece que anda".

Todo lo que descarga va a %TEMP% y se borra al terminar.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any

# Parametros de la familia MDX-Net. Salen de la firma del grafo
# ([batch, 4, 3072, 256]) mas la implementacion de UVR; el round-trip del control 1
# es lo que verifica que sean los correctos y no una conjetura.
N_FFT = 6144
DIM_F = 3072
DIM_T = 256
HOP = 1024
SAMPLE_RATE = 44100

MODEL_REPO = "masszhou/mdxnet"
MODEL_FILE = "UVR-MDX-NET-Voc_FT.onnx"

SPEECH_REPO = "Narsil/asr_dummy"
SPEECH_FILE = "i-know-kung-fu.mp3"


def _stft(audio: Any) -> Any:
    """STFT por canal -> [4, DIM_F, frames] (real e imaginario por canal)."""
    import numpy as np

    window = np.hanning(N_FFT).astype(np.float32)
    channels = []
    for channel in audio:
        frames = []
        for start in range(0, len(channel) - N_FFT + 1, HOP):
            frames.append(np.fft.rfft(channel[start : start + N_FFT] * window))
        spec = np.stack(frames, axis=-1)[:DIM_F]
        channels.append(spec)
    stacked = np.stack(channels)  # [2, DIM_F, frames] complejo
    return np.concatenate([stacked.real, stacked.imag]).astype(np.float32)


def _istft(spec: Any, length: int) -> Any:
    """Inversa de _stft con overlap-add."""
    import numpy as np

    real, imag = spec[:2], spec[2:]
    complex_spec = real + 1j * imag
    window = np.hanning(N_FFT).astype(np.float32)
    out = np.zeros((complex_spec.shape[0], length), dtype=np.float32)
    weight = np.zeros(length, dtype=np.float32)

    frames = complex_spec.shape[-1]
    for index in range(frames):
        start = index * HOP
        end = start + N_FFT
        if end > length:
            break
        for channel in range(complex_spec.shape[0]):
            padded = np.zeros(N_FFT // 2 + 1, dtype=np.complex64)
            padded[:DIM_F] = complex_spec[channel, :, index]
            out[channel, start:end] += np.fft.irfft(padded, n=N_FFT).astype(np.float32) * window
        weight[start:end] += window**2

    # Normalizacion de overlap-add: sin esto la amplitud queda escalada por el
    # solape y la correlacion mentiria.
    nonzero = weight > 1e-8
    out[:, nonzero] /= weight[nonzero]
    return out


def _correlation(a: Any, b: Any) -> float:
    import numpy as np

    length = min(a.shape[-1], b.shape[-1])
    x = a[:, :length].flatten()
    y = b[:, :length].flatten()
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _load_stereo_44k(cache_dir: pathlib.Path) -> Any:
    import numpy as np
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=SPEECH_REPO,
        filename=SPEECH_FILE,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        target = int(len(mono) * SAMPLE_RATE / rate)
        mono = np.interp(
            np.linspace(0, len(mono), target, endpoint=False),
            np.arange(len(mono)),
            mono,
        ).astype("float32")
    # El modelo espera estereo: se duplica el canal.
    return np.stack([mono, mono])


def _chunk_samples() -> int:
    return HOP * (DIM_T - 1) + N_FFT


def control_1_round_trip(audio: Any) -> dict[str, Any]:
    """Sin el modelo. Si esto falla, nada de lo que siga significa algo."""
    import numpy as np

    size = _chunk_samples()
    chunk = audio[:, :size]
    if chunk.shape[-1] < size:
        chunk = np.pad(chunk, ((0, 0), (0, size - chunk.shape[-1])))

    spec = _stft(chunk)
    back = _istft(spec, size)
    correlation = _correlation(chunk, back)
    return {
        "forma_del_espectrograma": list(spec.shape),
        "coincide_con_la_firma": list(spec.shape[:2]) == [4, DIM_F],
        "frames": int(spec.shape[-1]),
        "correlacion_round_trip": round(correlation, 4),
        # Un round-trip sano da >0.99. Por debajo de eso mi STFT esta mal y el
        # resultado del modelo seria ininterpretable.
        "ok": correlation > 0.99,
    }


def control_2_y_3(audio: Any, provider: str, cache_dir: pathlib.Path) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    try:
        model_path = hf_hub_download(MODEL_REPO, MODEL_FILE, cache_dir=str(cache_dir))
        session = ort.InferenceSession(model_path, providers=[provider])
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    size = _chunk_samples()
    chunk = audio[:, :size]
    if chunk.shape[-1] < size:
        chunk = np.pad(chunk, ((0, 0), (0, size - chunk.shape[-1])))

    spec = _stft(chunk)[:, :, :DIM_T]
    if spec.shape[-1] < DIM_T:
        spec = np.pad(spec, ((0, 0), (0, 0), (0, DIM_T - spec.shape[-1])))

    try:
        output = session.run(None, {"input": spec[None, ...]})[0][0]
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    separated = _istft(output, size)
    correlation = _correlation(chunk, separated)

    original_rms = float(np.sqrt(np.mean(chunk**2)))
    separated_rms = float(np.sqrt(np.mean(separated**2)))

    return {
        "ok": True,
        "forma_de_salida": list(output.shape),
        "rms_original": round(original_rms, 5),
        "rms_separado": round(separated_rms, 5),
        "correlacion_con_el_original": round(correlation, 4),
        # La entrada es VOZ SOLA y el modelo apunta a la voz: si funciona, la salida
        # tiene que parecerse mucho al original. Si devuelve basura, se derrumba.
        "calidad_plausible": correlation > 0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Descarga el modelo y separa audio real. Es lo unico que dice algo de calidad.",
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "parametros": {
            "n_fft": N_FFT,
            "dim_f": DIM_F,
            "dim_t": DIM_T,
            "hop": HOP,
            "sample_rate": SAMPLE_RATE,
            "muestras_por_trozo": _chunk_samples(),
        }
    }

    if args.separate:
        cache_dir = pathlib.Path(tempfile.mkdtemp(prefix="upflow-stems-spike-"))
        try:
            audio = _load_stereo_44k(cache_dir)
            report["control_1_round_trip_sin_modelo"] = control_1_round_trip(audio)
            if report["control_1_round_trip_sin_modelo"]["ok"]:
                report["separacion"] = {
                    provider: control_2_y_3(audio, provider, cache_dir)
                    for provider in ("DmlExecutionProvider", "CPUExecutionProvider")
                }
            else:
                report["separacion"] = "no se corrio: el control 1 fallo"
        finally:
            shutil.rmtree(cache_dir, ignore_errors=True)

    pathlib.Path(os.environ.get("TEMP", ".") + "/stems-spike.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
