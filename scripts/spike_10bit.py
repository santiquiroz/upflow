"""Mide DONDE se pierde la profundidad de 10 bits, antes de tocar nada.

El pipeline decodifica a `rgb24` y encodea a `yuv420p`. Los dos son de 8 bits,
asi que hay dos sospechosos y la pregunta es cual manda. Cambiar el encoder no
sirve de nada si la perdida ya paso al decodificar.

Se mide con un degradado suave, que es donde el banding de 8 bits se ve: en
10 bits hay 1024 niveles por canal y en 8 bits hay 256.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FFMPEG = REPO / "vendor" / "ffmpeg" / "bin" / "ffmpeg.exe"
FFPROBE = REPO / "vendor" / "ffmpeg" / "bin" / "ffprobe.exe"
ANCHO, ALTO = 512, 128


def niveles_distintos(datos: bytes, dtype) -> int:
    return len(np.unique(np.frombuffer(datos, dtype=dtype)))


def formato(video: Path) -> str:
    salida = subprocess.run(
        [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=pix_fmt,bits_per_raw_sample", "-of", "json", str(video)],
        capture_output=True, check=True,
    )
    stream = json.loads(salida.stdout)["streams"][0]
    return f"{stream.get('pix_fmt')} ({stream.get('bits_per_raw_sample', '?')} bits)"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        carpeta = Path(tmp)
        fuente = carpeta / "fuente10.mkv"

        subprocess.run(
            [str(FFMPEG), "-y", "-v", "error", "-f", "lavfi", "-i",
             f"gradients=s={ANCHO}x{ALTO}:d=1:c0=black:c1=white:nb_colors=2:r=5",
             "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-crf", "10",
             "-x265-params", "log-level=none", str(fuente)],
            check=True,
        )
        print(f"fuente               : {formato(fuente)}")

        def leer_crudo(pix_fmt: str, dtype) -> int:
            salida = subprocess.run(
                [str(FFMPEG), "-v", "error", "-i", str(fuente), "-frames:v", "1",
                 "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
                capture_output=True, check=True,
            )
            return niveles_distintos(salida.stdout, dtype)

        niveles_8 = leer_crudo("rgb24", np.uint8)
        niveles_16 = leer_crudo("rgb48le", np.uint16)

        print(f"\nniveles distintos leyendo rgb24  (camino de hoy) : {niveles_8}")
        print(f"niveles distintos leyendo rgb48le (16 bits)      : {niveles_16}")

        if niveles_16 <= niveles_8:
            print("\nEL FUENTE NO TIENE MAS DE 8 BITS DE INFORMACION: la medicion no dice nada.")
            return 1

        print(f"\nleer a 16 bits recupera {niveles_16 / max(niveles_8, 1):.1f}x mas niveles.")
        print("=> la perdida empieza al DECODIFICAR, no al encodear.")

        print("\n=== encoders que aceptan yuv420p10le en este build ===")
        for encoder in ("libx265", "libx264", "hevc_amf", "av1_amf", "libsvtav1"):
            prueba = carpeta / f"p_{encoder}.mkv"
            proceso = subprocess.run(
                [str(FFMPEG), "-y", "-v", "error", "-f", "lavfi", "-i",
                 "color=c=gray:s=320x240:d=0.2:r=5", "-c:v", encoder,
                 "-pix_fmt", "yuv420p10le", str(prueba)],
                capture_output=True,
            )
            if proceso.returncode == 0 and prueba.exists():
                print(f"  {encoder:12s} SI  -> {formato(prueba)}")
            else:
                razon = proceso.stderr.decode("utf-8", "replace").strip().splitlines()
                print(f"  {encoder:12s} NO  ({razon[-1][:80] if razon else 'sin salida'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
