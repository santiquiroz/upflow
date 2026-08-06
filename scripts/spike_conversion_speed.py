"""¿Por qué tarda tanto convertir un modelo, y se puede acelerar?

Reportado el 2026-08-06: la conversión demora mucho en las máquinas de otros
usuarios.

Hipótesis a medir: le pasamos `dtype="fp16"` a un export que corre en CPU, y
PyTorch no tiene kernels fp16 nativos para CPU — los emula operación por
operación. Si es así, exportar en fp32 y convertir el ONNX a fp16 después sería
más rápido, o directamente exportar en fp32.

No se razona sobre esto: se corre el MISMO camino de producción con y sin fp16 y
se comparan los relojes.
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


def medir(dtype: str | None, destino: Path) -> tuple[float, dict[str, float]]:
    if destino.exists():
        shutil.rmtree(destino, ignore_errors=True)
    destino.mkdir(parents=True, exist_ok=True)

    por_componente: dict[str, float] = {}
    ultimo = {"nombre": None, "t": time.perf_counter()}

    def on_component(name: str) -> None:
        ahora = time.perf_counter()
        if ultimo["nombre"] is not None:
            por_componente[ultimo["nombre"]] = ahora - ultimo["t"]
        ultimo["nombre"] = name
        ultimo["t"] = ahora

    arranque = time.perf_counter()
    _export_with_optimum(MODELO, destino, on_component, dtype, None)
    total = time.perf_counter() - arranque
    if ultimo["nombre"] is not None:
        por_componente[ultimo["nombre"]] = time.perf_counter() - ultimo["t"]
    return total, por_componente


def main() -> int:
    temporal = REPO / "runtime" / "temp" / "spike-conv"
    resultados = {}
    for etiqueta, dtype in (("fp32 (sin dtype)", None), ("fp16", "fp16")):
        print(f"\n=== {etiqueta} ===", flush=True)
        try:
            total, partes = medir(dtype, temporal / etiqueta.split()[0])
        except Exception as exc:  # noqa: BLE001 - cualquier fallo es un dato
            print(f"  FALLO: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        resultados[etiqueta] = total
        print(f"  TOTAL {total:.0f}s")
        for nombre, segundos in sorted(partes.items(), key=lambda x: -x[1]):
            print(f"    {nombre:32} {segundos:7.1f}s  ({segundos / total * 100:.0f}%)")

    print("\n================ VEREDICTO ================")
    for etiqueta, total in resultados.items():
        print(f"  {etiqueta:20} {total:.0f}s")
    if len(resultados) == 2:
        a, b = resultados.values()
        print(f"\n  fp16 es {b / a:.2f}x el tiempo de fp32")
    shutil.rmtree(temporal, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
