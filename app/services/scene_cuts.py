"""Evita que la interpolacion invente cuadros a caballo de un corte de escena.

Medido el 2026-08-05 (`scripts/spike_scenecut.py`): con dos escenas pegadas —dos
cuadros rojos seguidos de dos azules— RIFE produce un cuadro que queda a 144 del
rojo y a 26 del azul, cuando los limpios quedan a menos de 1,2 de su color. O
sea que inventa una mezcla de dos imagenes que no tienen nada que ver. En anime,
donde los cortes son duros y frecuentes, eso se ve como un fundido de un
fotograma en cada corte.

El arreglo no es interpolar mejor: en un corte NO HAY movimiento que estimar,
porque las dos imagenes no comparten nada. Es duplicar el ultimo cuadro de la
escena que termina, que es lo que hace cualquier interpolador serio.

La deteccion la hace ffmpeg con `scdet`, en una pasada de decodificado sin
encodear nada — mucho mas barato que releer los PNG ya escritos.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

# El default de scdet. Mas bajo detecta transiciones suaves como cortes y
# duplicaria cuadros donde SI habia movimiento que interpolar.
DEFAULT_THRESHOLD = 10.0

_SCORE_LINE = re.compile(r"lavfi\.scd\.score:\s*[\d.]+,\s*lavfi\.scd\.time:\s*([\d.]+)")


def build_scene_detect_command(*, ffmpeg: str, video: Path, threshold: float) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-i",
        str(video),
        "-vf",
        f"scdet=threshold={threshold}",
        # Se decodifica y se tira: lo unico que interesa son las lineas que
        # imprime el filtro.
        "-f",
        "null",
        "-",
    ]


def parse_scene_cut_times(output: str) -> list[float]:
    return [float(m.group(1)) for m in _SCORE_LINE.finditer(output)]


def source_index_at_time(seconds: float, *, fps: float) -> int:
    """El cuadro de origen con el que ARRANCA la escena nueva.

    Se redondea y no se trunca: truncar correria el corte un cuadro y el
    fantasma quedaria igual, solo que en otro lado.
    """
    if fps <= 0:
        return 0
    return int(round(seconds * fps))


def frames_straddling_cut(*, cut_index: int, source_count: int, output_count: int) -> list[int]:
    """Los cuadros de salida que mezclan las dos escenas.

    La interpolacion reparte `output_count` cuadros sobre la linea de tiempo de
    los `source_count` originales, asi que el cuadro de salida j vive en la
    posicion j*(source_count-1)/(output_count-1). Los que caen ESTRICTAMENTE
    entre el ultimo cuadro de la escena vieja y el primero de la nueva son los
    inventados a caballo del corte.
    """
    if cut_index <= 0 or cut_index >= source_count or output_count < 2 or source_count < 2:
        return []
    paso = (source_count - 1) / (output_count - 1)
    anterior = cut_index - 1
    return [
        j
        for j in range(output_count)
        if anterior < j * paso < cut_index
    ]


def repair_interpolated_cuts(
    frames_dir: Path, *, cut_indices: list[int], source_count: int, output_count: int
) -> int:
    """Reemplaza cada cuadro mezclado por el anterior. Devuelve cuantos cambio."""
    reparados = 0
    for cut_index in cut_indices:
        for indice in frames_straddling_cut(
            cut_index=cut_index, source_count=source_count, output_count=output_count
        ):
            # Los archivos van de 1 en adelante; el indice de salida de 0.
            mezclado = frames_dir / f"{indice + 1:08d}.png"
            limpio = frames_dir / f"{indice:08d}.png"
            if not mezclado.exists() or not limpio.exists():
                # Un cuadro que no esta en disco no puede tirar abajo el video
                # entero: el resto de la reparacion sigue.
                continue
            shutil.copyfile(limpio, mezclado)
            reparados += 1
    return reparados
