"""Como conseguir un ONNX fp16 sin pagar el export fp16 en CPU.

Medido el 2026-08-06 (SD1.4, este equipo): exportar con `dtype="fp16"` tarda mas
de 23 minutos y no habia terminado cuando se corto; el mismo export sin dtype
tarda 58 SEGUNDOS. PyTorch no tiene kernels fp16 nativos en CPU, asi que cada
operacion del trazado convierte a fp32, calcula y vuelve — la emulacion que el
propio aviso de la libreria anuncia.

`fp16` es el default de la app, o sea que TODA conversion toma el camino lento.

Se comparan tres caminos, con el reloj y el tamano de salida:
  A) fp32 puro
  B) fp16 en el export (lo que se hace hoy)
  C) fp32 en el export + conversion del GRAFO a fp16 despues
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.services.generation_converter import _export_with_optimum  # noqa: E402

MODELO = "CompVis/stable-diffusion-v1-4"
RAIZ = REPO / "runtime" / "temp" / "spike-precision"
TECHO_FP16_S = 900


def peso_gb(carpeta: Path) -> float:
    return sum(f.stat().st_size for f in carpeta.rglob("*") if f.is_file()) / 1e9


def exportar(dtype: str | None, destino: Path) -> float:
    if destino.exists():
        shutil.rmtree(destino, ignore_errors=True)
    destino.mkdir(parents=True, exist_ok=True)
    arranque = time.perf_counter()
    _export_with_optimum(MODELO, destino, lambda _n: None, dtype, None)
    return time.perf_counter() - arranque


def a_fp16_el_grafo(origen: Path) -> tuple[float, str]:
    """Convierte los .onnx ya exportados a fp16, sin volver a trazar el modelo.

    Se usa la variante que trabaja sobre el ARCHIVO: la que carga el grafo en
    memoria muere con `Failed to serialize proto` en cuanto el modelo pasa los
    2 GB del limite de protobuf, que es el caso de cualquier UNet real.
    """
    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        return 0.0, f"falta {exc.name}"

    arranque = time.perf_counter()
    for archivo in sorted(origen.rglob("*.onnx")):
        try:
            float16.convert_float_to_float16_model_path(
                str(archivo), keep_io_types=True
            )
        except Exception as exc:  # noqa: BLE001 - cual falla es el dato
            return time.perf_counter() - arranque, f"{archivo.parent.name}: {type(exc).__name__}"
    return time.perf_counter() - arranque, "ok"


def main() -> int:
    RAIZ.mkdir(parents=True, exist_ok=True)

    print("=== A) fp32 puro ===", flush=True)
    a = RAIZ / "fp32"
    t_a = exportar(None, a)
    print(f"  {t_a:.0f}s   {peso_gb(a):.2f} GB")

    print("\n=== C) fp32 + grafo a fp16 ===", flush=True)
    c = RAIZ / "fp32-luego-fp16"
    t_c1 = exportar(None, c)
    t_c2, estado = a_fp16_el_grafo(c)
    if estado != "ok":
        print(f"  export {t_c1:.0f}s, conversion del grafo NO se pudo: {estado}")
    else:
        print(f"  export {t_c1:.0f}s + grafo {t_c2:.0f}s = {t_c1 + t_c2:.0f}s   {peso_gb(c):.2f} GB")

    print(f"\n=== B) fp16 en el export (lo de hoy) — techo {TECHO_FP16_S}s ===", flush=True)
    b = RAIZ / "fp16"
    arranque = time.perf_counter()
    try:
        import threading

        resultado: dict = {}

        def correr():
            try:
                resultado["t"] = exportar("fp16", b)
            except Exception as exc:  # noqa: BLE001
                resultado["error"] = f"{type(exc).__name__}: {exc}"

        hilo = threading.Thread(target=correr, daemon=True)
        hilo.start()
        hilo.join(TECHO_FP16_S)
        if hilo.is_alive():
            print(f"  SIGUE CORRIENDO pasados {TECHO_FP16_S}s ({TECHO_FP16_S / 60:.0f} min) — se corta acá")
        elif "error" in resultado:
            print(f"  FALLO: {resultado['error'][:150]}")
        else:
            print(f"  {resultado['t']:.0f}s   {peso_gb(b):.2f} GB")
    finally:
        print(f"  (reloj de pared: {time.perf_counter() - arranque:.0f}s)")

    print("\n================ VEREDICTO ================")
    print(f"  A) fp32 puro                {t_a:.0f}s")
    if estado == "ok":
        print(f"  C) fp32 + grafo a fp16      {t_c1 + t_c2:.0f}s  <- mismo resultado, sin emular fp16 en CPU")
    print(f"  B) fp16 en el export        >{TECHO_FP16_S}s (lo que hace la app hoy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
