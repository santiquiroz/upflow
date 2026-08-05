"""Cuanta superficie de la pieza queda en voladizo, y en que orientacion menos.

En FDM cada capa se apoya en la anterior. Una cara que mira hacia abajo mas alla
de cierto angulo no tiene sobre que apoyarse: necesita soporte, o sea material
extra, tiempo extra y una superficie fea donde estaba pegado. En una pieza
funcional esa superficie fea suele caer justo en la cara que tiene que asentar.

`limit_deg` se mide DESDE LA VERTICAL y sigue la convencion de los laminadores:
es la inclinacion maxima que todavia imprime sin soporte, asi que un numero MAS
GRANDE es MAS PERMISIVO. Tenerlo al reves seria una trampa: alguien que copia el
45 de su PrusaSlicer esperaria ese comportamiento y obtendria el contrario.

El 45 tipico no es una constante del universo — depende de la maquina, del
material y de la velocidad — asi que entra como parametro y no como verdad.

Se calcula exacto sobre las normales: una comparacion por cara, sin muestreo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_OVERHANG_LIMIT_DEG = 45.0

# Las vueltas de 90 grados que dejan la pieza apoyada en una cara distinta.
# No se prueban angulos libres a proposito: una orientacion torcida necesita una
# balsa y anula la ventaja de tener menos voladizo.
_ROTATIONS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (180, 0, 0),
    (90, 0, 0),
    (270, 0, 0),
    (0, 90, 0),
    (0, 270, 0),
)


@dataclass(frozen=True, slots=True)
class OverhangReport:
    total_area: float
    overhang_area: float

    @property
    def overhang_area_ratio(self) -> float:
        return self.overhang_area / self.total_area if self.total_area > 0 else 0.0


@dataclass(frozen=True, slots=True)
class OrientationChoice:
    rotation_deg: tuple[int, int, int]
    mesh: np.ndarray
    overhang_area_ratio: float


def _face_normals_and_areas(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lado_a = triangles[:, 1] - triangles[:, 0]
    lado_b = triangles[:, 2] - triangles[:, 0]
    cruz = np.cross(lado_a, lado_b)
    largos = np.linalg.norm(cruz, axis=1)
    areas = 0.5 * largos
    seguros = np.where(largos > 0, largos, 1.0)
    return cruz / seguros[:, None], areas


def overhang_report(
    triangles: np.ndarray, *, limit_deg: float = DEFAULT_OVERHANG_LIMIT_DEG
) -> OverhangReport:
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.size == 0:
        return OverhangReport(total_area=0.0, overhang_area=0.0)

    normales, areas = _face_normals_and_areas(triangles)
    # `hacia_abajo` es el coseno del angulo entre la normal y el suelo: 1 para
    # una cara horizontal mirando abajo, 0 para una pared vertical.
    #
    # El umbral es el SENO del limite, no el coseno, porque el limite se mide
    # desde la vertical y este producto desde abajo. Con seno, una pared vertical
    # nunca es voladizo y subir el limite afloja el criterio, que es lo que hace
    # cualquier laminador.
    hacia_abajo = -normales[:, 2]
    umbral = math.sin(math.radians(limit_deg))
    en_voladizo = hacia_abajo > umbral

    return OverhangReport(
        total_area=float(areas.sum()),
        overhang_area=float(areas[en_voladizo].sum()),
    )


def _rotate(triangles: np.ndarray, rotation: tuple[int, int, int]) -> np.ndarray:
    resultado = triangles
    for eje, grados in enumerate(rotation):
        if grados == 0:
            continue
        radianes = math.radians(grados)
        c, s = math.cos(radianes), math.sin(radianes)
        if eje == 0:
            matriz = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
        elif eje == 1:
            matriz = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
        else:
            matriz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        resultado = resultado @ matriz.T
    return resultado


def best_print_orientation(
    triangles: np.ndarray, *, limit_deg: float = DEFAULT_OVERHANG_LIMIT_DEG
) -> OrientationChoice:
    """La vuelta de 90 grados que deja menos superficie en voladizo.

    Empata a favor de no girar: si la orientacion original ya es la mejor, se
    deja como esta en vez de rotar por rotar.
    """
    mejor: OrientationChoice | None = None
    for rotacion in _ROTATIONS:
        girada = _rotate(triangles, rotacion) if rotacion != (0, 0, 0) else triangles
        ratio = overhang_report(girada, limit_deg=limit_deg).overhang_area_ratio
        if mejor is None or ratio < mejor.overhang_area_ratio - 1e-12:
            mejor = OrientationChoice(
                rotation_deg=rotacion, mesh=girada, overhang_area_ratio=ratio
            )
    if mejor is None:
        raise ValueError("No hay orientacion que evaluar.")
    return mejor
