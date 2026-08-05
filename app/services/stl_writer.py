"""Escribe la malla a STL binario, que es lo que va al laminador.

Binario y no ASCII: un ASCII de un millon de triangulos pesa unas cinco veces
mas y ningun laminador lo agradece.

Las normales se calculan de verdad aunque casi todos los laminadores las
ignoren, porque hay herramientas de reparacion que si las leen y escribir ceros
ahi es escribir una mentira que despues alguien usa.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

HEADER = b"Upflow"


def _unit_normals(triangles: np.ndarray) -> np.ndarray:
    lado_a = triangles[:, 1] - triangles[:, 0]
    lado_b = triangles[:, 2] - triangles[:, 0]
    normales = np.cross(lado_a, lado_b)
    largos = np.linalg.norm(normales, axis=1, keepdims=True)
    # Un triangulo de area cero da largo cero, y dividir ahi mete NaN en el
    # archivo, que lo rompe entero. Esos van con normal cero, que es lo que el
    # formato usa para "no se".
    seguros = np.where(largos > 0, largos, 1.0)
    return np.where(largos > 0, normales / seguros, 0.0)


def write_stl(path: Path, triangles: np.ndarray) -> Path:
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(
            f"Se esperaban triangulos con forma (n, 3, 3) y llegaron {triangles.shape}."
        )
    if triangles.shape[0] == 0:
        raise ValueError("La malla esta vacia: no hay nada que escribir.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normales = _unit_normals(triangles).astype(np.float32)
    vertices = triangles.reshape(-1, 9).astype(np.float32)

    registros = np.zeros(
        len(triangles),
        dtype=np.dtype([("normal", "<3f4"), ("vertices", "<9f4"), ("attr", "<u2")]),
    )
    registros["normal"] = normales
    registros["vertices"] = vertices

    with path.open("wb") as handle:
        handle.write(HEADER.ljust(80, b"\0")[:80])
        handle.write(struct.pack("<I", len(triangles)))
        handle.write(registros.tobytes())
    return path
