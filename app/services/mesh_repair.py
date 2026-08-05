"""Tapa los agujeros de una malla, sin inventar geometria que nunca estuvo.

Reparar aca significa UNA cosa: cerrar los bucles de borde que dejaron
triangulos faltantes, poniendo una tapa de abanico desde el centro del bucle. NO
significa reconstruir la forma que el modelo tendria que haber tenido, ni
adivinar detalle perdido. Un agujero grande queda tapado con una superficie
plana, que es honesto y feo, y es mejor que una invencion que parece correcta.

Y NUNCA se declara reparada: se vuelve a MEDIR y se devuelve la medicion. Un
reparador que afirma haber arreglado sin comprobarlo es peor que no tener
reparador, porque manda a imprimir con confianza prestada.
"""

from __future__ import annotations

import numpy as np

from app.services.mesh_inspect import (
    MIN_TRIANGLE_AREA,
    MeshReport,
    inspect_mesh,
    weld_vertices,
)


def _drop_degenerates(triangles: np.ndarray) -> np.ndarray:
    lado_a = triangles[:, 1] - triangles[:, 0]
    lado_b = triangles[:, 2] - triangles[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(lado_a, lado_b), axis=1)
    return triangles[areas > MIN_TRIANGLE_AREA]


def _directed_boundary_edges(caras: np.ndarray) -> list[tuple[int, int]]:
    """Las aristas que solo tocan un triangulo, EN LA DIRECCION que las recorre.

    La direccion importa para el bobinado de la tapa: una arista de borde que el
    triangulo recorre a->b tiene que ser cerrada por una cara que la recorra
    b->a, o la tapa queda mirando adentro y el laminador la lee como un hueco.
    """
    dirigidas = np.concatenate(
        [caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [2, 0]]], axis=0
    )
    sin_direccion = np.sort(dirigidas, axis=1)
    _unicas, inversa, cuentas = np.unique(
        sin_direccion, axis=0, return_inverse=True, return_counts=True
    )
    solitarias = cuentas[inversa.reshape(-1)] == 1
    return [tuple(par) for par in dirigidas[solitarias]]


def boundary_loops(triangles: np.ndarray) -> list[list[int]]:
    """Los bucles cerrados de aristas de borde, en indices de vertice soldado."""
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.size == 0:
        return []
    _vertices, caras = weld_vertices(_drop_degenerates(triangles))
    if caras.size == 0:
        return []

    siguiente: dict[int, int] = {}
    for origen, destino in _directed_boundary_edges(caras):
        # Un vertice con dos bordes salientes es una malla mas rara de lo que
        # este reparador sabe manejar; se queda con el primero y el resultado se
        # mide igual al final.
        siguiente.setdefault(int(origen), int(destino))

    bucles: list[list[int]] = []
    visitados: set[int] = set()
    for arranque in siguiente:
        if arranque in visitados:
            continue
        bucle: list[int] = []
        actual = arranque
        while actual not in visitados and actual in siguiente:
            visitados.add(actual)
            bucle.append(actual)
            actual = siguiente[actual]
        if len(bucle) >= 3 and actual == arranque:
            bucles.append(bucle)
    return bucles


def _fan_cap(vertices: np.ndarray, bucle: list[int]) -> np.ndarray:
    """Tapa el bucle con un abanico desde su centro.

    El bucle viene en la direccion en que lo recorren los triangulos de al lado,
    asi que la tapa lo recorre al reves para quedar mirando hacia afuera.
    """
    puntos = vertices[bucle]
    centro = puntos.mean(axis=0)
    caras = [
        [puntos[(i + 1) % len(bucle)], puntos[i], centro] for i in range(len(bucle))
    ]
    return np.asarray(caras, dtype=np.float64)


def repair_mesh(triangles: np.ndarray) -> tuple[np.ndarray, MeshReport]:
    """Devuelve la malla tapada y la MEDICION del resultado, no una promesa."""
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.size == 0:
        return triangles, inspect_mesh(triangles)

    limpia = _drop_degenerates(triangles)
    bucles = boundary_loops(limpia)
    if not bucles:
        # Tocar lo que ya estaba bien solo puede empeorarlo.
        return limpia, inspect_mesh(limpia)

    vertices, _caras = weld_vertices(limpia)
    tapas = [_fan_cap(vertices, bucle) for bucle in bucles]
    reparada = np.concatenate([limpia, *tapas])
    return reparada, inspect_mesh(reparada)
