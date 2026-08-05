"""Spike: ¿el verificador de malla dice la verdad, y aguanta el tamaño real?

Uso:
    .venv\\Scripts\\python scripts\\spike_mesh_inspect.py

Los tests unitarios usan un tetraedro de cuatro caras. Eso prueba la lógica, no
que sirva. Acá se mide contra dos cosas que el código no puede fingir:

  1. **Volumen conocido de antemano.** Una esfera UV converge a 4/3·π·r³ desde
     abajo. Si el volumen medido no converge a esa fórmula, la medición está mal
     y ningún test propio lo iba a notar.
  2. **El tamaño real.** Un comentario de la investigación de mercado se quejaba
     de que un generador anuncia "2 millones de polígonos" como si fuera bueno.
     Si el verificador tarda minutos a esa escala, no sirve dentro de un job.

Y al final se rompe una malla sana a propósito: un verificador que nunca dice
que no, no está verificando nada.
"""

from __future__ import annotations

import math
import struct
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.services.mesh_inspect import inspect_mesh  # noqa: E402
from app.services.stl_reader import read_stl  # noqa: E402

RADIO = 10.0


def uv_sphere(segments: int, rings: int, radius: float = RADIO) -> np.ndarray:
    """Esfera cerrada por triangulos, con los polos resueltos como abanicos."""
    thetas = np.linspace(0.0, math.pi, rings + 1)
    phis = np.linspace(0.0, 2 * math.pi, segments + 1)[:-1]

    def punto(i_ring: int, i_seg: int) -> tuple[float, float, float]:
        t, p = thetas[i_ring], phis[i_seg % segments]
        return (
            radius * math.sin(t) * math.cos(p),
            radius * math.sin(t) * math.sin(p),
            radius * math.cos(t),
        )

    triangulos: list = []
    for r in range(rings):
        for s in range(segments):
            a, b = punto(r, s), punto(r, s + 1)
            c, d = punto(r + 1, s + 1), punto(r + 1, s)
            if r == 0:
                triangulos.append([a, c, d])
            elif r == rings - 1:
                triangulos.append([a, b, d])
            else:
                triangulos.append([a, b, d])
                triangulos.append([b, c, d])
    return np.asarray(triangulos, dtype=np.float64)


def write_binary_stl(path: Path, triangles: np.ndarray) -> None:
    with path.open("wb") as handle:
        handle.write(b"upflow spike".ljust(80, b"\0"))
        handle.write(struct.pack("<I", len(triangles)))
        for tri in triangles.astype(np.float32):
            handle.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            handle.write(tri.tobytes())
            handle.write(struct.pack("<H", 0))


def main() -> int:
    exacto = 4 / 3 * math.pi * RADIO**3
    print(f"volumen exacto de la esfera (r={RADIO:.0f}): {exacto:.2f} mm3\n")

    print("=== 1. ¿el volumen converge a la fórmula? ===")
    print(f"{'triángulos':>12} {'volumen':>12} {'error':>9}  estanco")
    anterior = None
    for segmentos, anillos in ((16, 8), (32, 16), (64, 32), (128, 64)):
        malla = uv_sphere(segmentos, anillos)
        reporte = inspect_mesh(malla)
        if reporte.volume is None:
            print(f"{reporte.triangle_count:>12} {'ABIERTA':>12}")
            return 1
        error = abs(reporte.volume - exacto) / exacto
        print(
            f"{reporte.triangle_count:>12} {reporte.volume:>12.2f} {error:>8.2%}  "
            f"{'sí' if reporte.printable else 'NO'}"
        )
        if anterior is not None and error >= anterior:
            print("\nNO CONVERGE: refinar la malla no acerca el volumen al real.")
            return 1
        anterior = error

    if anterior > 0.01:
        print(f"\nCONVERGE PERO LEJOS: {anterior:.2%} de error con la malla más fina.")
        return 1
    print(f"\nCONVERGE: el error cae hasta {anterior:.3%}. La medición es real.\n")

    print("=== 2. ¿aguanta el tamaño que producen estos generadores? ===")
    grande = uv_sphere(1024, 512)
    with tempfile.TemporaryDirectory() as tmp:
        destino = Path(tmp) / "grande.stl"
        arranque = time.perf_counter()
        write_binary_stl(destino, grande)
        escritura = time.perf_counter() - arranque

        arranque = time.perf_counter()
        leida = read_stl(destino)
        lectura = time.perf_counter() - arranque

        arranque = time.perf_counter()
        reporte = inspect_mesh(leida)
        analisis = time.perf_counter() - arranque

    mb = destino_size = len(grande) * 50 / 1024 / 1024
    print(f"triángulos           : {reporte.triangle_count:,}")
    print(f"vértices soldados    : {reporte.vertex_count:,}")
    print(f"archivo              : {mb:.1f} MB")
    print(f"escribir             : {escritura:.2f} s")
    print(f"leer                 : {lectura:.2f} s")
    print(f"analizar             : {analisis:.2f} s")
    print(f"estanco y manifold   : {'sí' if reporte.printable else 'NO'}")

    if analisis > 20.0:
        print("\nDEMASIADO LENTO: no se puede correr esto dentro de un job.")
        return 1

    print("\n=== 3. ¿dice que no cuando corresponde? ===")
    rota = np.delete(grande, np.arange(0, 300), axis=0)
    reporte_roto = inspect_mesh(rota)
    print(f"malla con 300 triángulos borrados -> estanca: {reporte_roto.is_watertight}")
    print(f"aristas de borde detectadas       : {reporte_roto.boundary_edges}")
    for problema in reporte_roto.problems:
        print(f"  - {problema}")

    if reporte_roto.is_watertight:
        print("\nNO SIRVE: le abrí un agujero de 300 triángulos y dijo que estaba cerrada.")
        return 1

    print("\n================ VEREDICTO ================")
    print("MIDE DE VERDAD: converge al volumen exacto, aguanta el tamaño real,")
    print("y detecta el agujero que le abrí a propósito.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
