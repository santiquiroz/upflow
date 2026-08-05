"""Spike: ¿RIFE fantasmea en un corte de escena, y el arreglo lo saca?

Uso:
    .venv\\Scripts\\python scripts\\spike_scenecut.py

La sospecha: entre el último cuadro de una escena y el primero de la siguiente,
RIFE inventa un cuadro intermedio que es la mezcla de dos imágenes que no tienen
nada que ver. En anime, donde los cortes son duros y frecuentes, eso se ve como
un fundido de un fotograma en cada corte.

Se mide con dos escenas de colores planos pegadas. Un cuadro limpio queda cerca
de su color; uno mezclado queda lejos de los dos. Después se aplica la
reparación y se vuelve a medir: si el arreglo sirve, no queda ninguno mezclado.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from app.services.scene_cuts import repair_interpolated_cuts  # noqa: E402

FFMPEG = REPO / "vendor" / "ffmpeg" / "bin" / "ffmpeg.exe"
RIFE = REPO / "vendor" / "rife" / "rife-ncnn-vulkan.exe"
MODELO = REPO / "vendor" / "rife" / "models" / "rife-v4.25"

COLORES = ["red", "red", "blue", "blue"]
CUT_INDEX = 2  # el primer cuadro de la escena nueva
SALIDA_CUADROS = 7
# Un cuadro limpio queda a menos de 1,2 de su color; uno mezclado, a decenas.
UMBRAL_MEZCLA = 20.0


def leer_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as imagen:
        return np.asarray(imagen.convert("RGB"), dtype=np.float32)


def distancia(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def mezclados(carpeta: Path, rojo: np.ndarray, azul: np.ndarray) -> list[int]:
    encontrados = []
    for i, path in enumerate(sorted(carpeta.glob("*.png"))):
        cuadro = leer_png(path)
        d_rojo, d_azul = distancia(cuadro, rojo), distancia(cuadro, azul)
        es_mezcla = min(d_rojo, d_azul) > UMBRAL_MEZCLA
        if es_mezcla:
            encontrados.append(i)
        print(f"  {i}: {d_rojo:7.1f} / {d_azul:7.1f} {'  <-- MEZCLA' if es_mezcla else ''}")
    return encontrados


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        carpeta = Path(tmp)
        entrada, salida = carpeta / "in", carpeta / "out"
        entrada.mkdir()
        salida.mkdir()

        for i, color in enumerate(COLORES, start=1):
            subprocess.run(
                [str(FFMPEG), "-y", "-v", "error", "-f", "lavfi", "-i",
                 f"color=c={color}:s=256x256:d=0.1:r=1", "-frames:v", "1",
                 str(entrada / f"{i:08d}.png")],
                check=True,
            )

        proceso = subprocess.run(
            [str(RIFE), "-i", str(entrada), "-o", str(salida), "-m", str(MODELO),
             "-n", str(SALIDA_CUADROS), "-g", "0", "-f", "%08d.png"],
            capture_output=True,
        )
        if proceso.returncode != 0:
            print("RIFE FALLO:", proceso.stderr.decode("utf-8", "replace")[-400:])
            return 1

        producidos = sorted(salida.glob("*.png"))
        print(f"cuadros de entrada: {len(COLORES)}  ->  producidos: {len(producidos)}")

        base = leer_png(producidos[0])
        rojo = np.zeros_like(base)
        rojo[..., 0] = 255.0
        azul = np.zeros_like(base)
        azul[..., 2] = 255.0

        print("\ncuadro : distancia al rojo / al azul")
        antes = mezclados(salida, rojo, azul)
        if not antes:
            print("\nNO HAY FANTASMA: RIFE ya resuelve el corte solo.")
            return 0
        print(f"\nHAY FANTASMA: {len(antes)} cuadro(s) mezclan las dos escenas.")

        reparados = repair_interpolated_cuts(
            salida,
            cut_indices=[CUT_INDEX],
            source_count=len(COLORES),
            output_count=len(producidos),
        )
        print(f"\ncuadros reparados: {reparados}")
        print("\ndespues de reparar:")
        quedan = mezclados(salida, rojo, azul)

        print("\n================ VEREDICTO ================")
        if quedan:
            print(f"EL ARREGLO NO ALCANZO: quedan {len(quedan)} cuadro(s) mezclados.")
            return 1
        print("EL FANTASMA DESAPARECIO: cada cuadro pertenece a una sola escena.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
