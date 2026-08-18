"""¿Un repo con ONNX completo bajado DIRECTO realmente funciona?

Ésta es la prueba que quedó sin veredicto. El arreglo de la detección se había
hecho antes y se revirtió porque el pipeline bajado directo falló al validar con
un `DmlCommandRecorder 80004005` — pero esa prueba estaba contaminada: había un
job de generación del usuario en la misma GPU que falló con el MISMO error en el
mismo minuto. O sea que midió contención de VRAM, no si el repo sirve.

Si acá carga y genera, el arreglo ahorra cerca de cuarenta minutos por modelo en
los repos más usados. Si falla con la GPU libre, se revierte otra vez — y esta
vez con veredicto.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODELO = "stabilityai/sdxl-turbo"
DESTINO = REPO / "runtime" / "temp" / "spike-onnx-directo"

# Lo que el instalador bajaría: el pipeline ONNX, sin los pesos PyTorch.
PATRONES = [
    "model_index.json",
    "*/config.json",
    "scheduler/*",
    "tokenizer*/*",
    "text_encoder*/model.onnx",
    "unet/model.onnx",
    "unet/*.onnx_data",
    "vae_decoder/model.onnx",
    "vae_encoder/model.onnx",
]


def main() -> int:
    from huggingface_hub import snapshot_download

    if DESTINO.exists():
        shutil.rmtree(DESTINO, ignore_errors=True)

    print(f"Bajando el ONNX de {MODELO} (sin los pesos PyTorch)...", flush=True)
    arranque = time.perf_counter()
    snapshot_download(MODELO, local_dir=str(DESTINO), allow_patterns=PATRONES)
    bajada = time.perf_counter() - arranque
    peso = sum(f.stat().st_size for f in DESTINO.rglob("*") if f.is_file())
    print(f"  bajado en {bajada:.0f}s, {peso / 1e9:.1f} GB")
    print(f"  carpetas: {sorted(d.name for d in DESTINO.iterdir() if d.is_dir())}")

    print("\nCargando el pipeline y generando una imagen de verdad...", flush=True)
    arranque = time.perf_counter()
    try:
        from optimum.onnxruntime import ORTDiffusionPipeline

        pipeline = ORTDiffusionPipeline.from_pretrained(str(DESTINO), provider="DmlExecutionProvider")
        salida = pipeline(prompt="a red apple", num_inference_steps=1, width=512, height=512)
        imagen = salida.images[0]
        print(f"  OK: generó {imagen.size} en {time.perf_counter() - arranque:.0f}s")
        extremos = imagen.convert("L").getextrema()
        print(f"  rango de la imagen: {extremos}")
        if extremos[0] == extremos[1]:
            print("  PERO la imagen salió PLANA: carga pero no sirve.")
            return 1
    except Exception as exc:  # noqa: BLE001 - cualquier fallo es el veredicto
        print(f"  FALLO: {type(exc).__name__}: {str(exc)[:400]}")
        return 1
    finally:
        shutil.rmtree(DESTINO, ignore_errors=True)

    print("\n=== VEREDICTO: el repo con ONNX completo sirve bajado directo ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
