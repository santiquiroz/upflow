"""Dice si una malla se puede imprimir, midiendola en vez de creerle al que la genero.

"Watertight y manifold" es el unico predictor de impresion en el que coinciden
todos los bandos: los que venden generadores 3D y los que los critican. Asi que
se mide aca adentro, con la definicion completa y sin misticismo:

  - Cada arista de una superficie cerrada la comparten EXACTAMENTE 2 triangulos.
  - Una arista con 1 triangulo es un agujero: el laminador no sabe que esta
    adentro y que afuera, y rellena cualquier cosa.
  - Una arista con 3 o mas es no-manifold: la geometria se bifurca y el
    laminador tiene que adivinar.

LA TRAMPA que hay que pagar primero: el STL es sopa de triangulos y repite cada
vertice completo en cada cara. Sin soldar los puntos identicos, TODAS las
aristas parecen de borde y hasta un cubo perfecto da "roto". Por eso `weld_vertices`
va antes que cualquier conteo, y suelda con tolerancia: los generadores escriben
float32 y el mismo punto vuelve con basura en el ultimo bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# La tolerancia de soldado NO es un numero fijo, y ese fue un error medido:
# con 0,001 mm fijos, una esfera sana de 1.046.528 triangulos daba NO ESTANCA
# porque junto a los polos los vertices caen a 0,000376 mm y se fusionaban 3.328
# caras buenas (`scripts/spike_mesh_inspect.py`). Declarar rota una malla sana es
# el peor error que puede cometer este verificador.
#
# Lo unico que el soldado tiene que deshacer es la duplicacion del STL: el MISMO
# punto escrito dos veces con redondeo de float32. float32 guarda ~7 cifras, asi
# que el error de redondeo escala con la magnitud de la coordenada y la
# tolerancia tiene que escalar igual.
FLOAT32_RELATIVE_STEP = 1e-6
# Piso para un modelo pegado al origen, donde no hay magnitud de la que colgarse.
MIN_WELD_TOLERANCE_MM = 1e-9

# Un triangulo con menos area que esto no aporta superficie. Se descarta ANTES
# de contar aristas: si no, sus dos aristas coincidentes ensucian el conteo y
# abren un solido que estaba cerrado.
MIN_TRIANGLE_AREA = 1e-12


@dataclass(frozen=True, slots=True)
class MeshReport:
    triangle_count: int
    vertex_count: int
    boundary_edges: int
    non_manifold_edges: int
    degenerate_triangles: int
    # `None` cuando la malla esta abierta: el volumen de algo sin adentro no
    # significa nada, y devolver un numero igual seria inventar una medida.
    volume: float | None
    size: tuple[float, float, float]
    problems: list[str] = field(default_factory=list)

    @property
    def is_watertight(self) -> bool:
        return self.boundary_edges == 0

    @property
    def is_manifold(self) -> bool:
        return self.non_manifold_edges == 0

    @property
    def printable(self) -> bool:
        return self.is_watertight and self.is_manifold


def weld_tolerance_for(triangles: np.ndarray) -> float:
    """Cuan cerca tienen que estar dos puntos para ser el mismo.

    Sale de la precision de float32 a la escala del modelo, no de un numero
    elegido a dedo: una pieza de 800 mm tiene mas error de redondeo por punto
    que una de 8 mm, y usar la misma tolerancia para las dos rompe una de las dos.
    """
    magnitud = float(np.max(np.abs(triangles))) if triangles.size else 0.0
    return max(MIN_WELD_TOLERANCE_MM, magnitud * FLOAT32_RELATIVE_STEP)


def weld_vertices(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convierte la sopa de triangulos en vertices unicos + caras indexadas.

    Se cuantiza a la tolerancia antes de comparar: soldar de menos parte cada
    esquina en varias y declara rota una malla sana; soldar de mas fusiona
    detalle real. La tolerancia es el unico parametro y esta elegida para que
    ninguno de los dos pase.
    """
    puntos = triangles.reshape(-1, 3)
    llaves = np.round(puntos / weld_tolerance_for(triangles)).astype(np.int64)
    _unicas, primera, inversa = np.unique(
        llaves, axis=0, return_index=True, return_inverse=True
    )
    vertices = puntos[primera]
    caras = inversa.reshape(-1, 3)
    return vertices, caras


def _triangle_areas(triangles: np.ndarray) -> np.ndarray:
    lado_a = triangles[:, 1] - triangles[:, 0]
    lado_b = triangles[:, 2] - triangles[:, 0]
    return 0.5 * np.linalg.norm(np.cross(lado_a, lado_b), axis=1)


def _drop_faces_collapsed_by_welding(caras: np.ndarray) -> tuple[np.ndarray, int]:
    """Saca las caras cuyas esquinas quedaron pegadas al soldar.

    Una cara con dos indices iguales ya no tiene area, y contar sus aristas
    inventa problemas que el modelo no tiene: asi es como una esfera sana
    reportaba 2.629 aristas bifurcadas que no existian.
    """
    if caras.size == 0:
        return caras, 0
    distintas = (
        (caras[:, 0] != caras[:, 1])
        & (caras[:, 1] != caras[:, 2])
        & (caras[:, 0] != caras[:, 2])
    )
    return caras[distintas], int(np.sum(~distintas))


def _edge_counts(caras: np.ndarray) -> tuple[int, int]:
    """Cuantas aristas tienen 1 triangulo (borde) y cuantas 3 o mas (bifurcadas)."""
    aristas = np.concatenate(
        [caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [2, 0]]], axis=0
    )
    # Sin direccion: la arista A->B y la B->A son la misma linea fisica.
    ordenadas = np.sort(aristas, axis=1)
    _unicas, cuentas = np.unique(ordenadas, axis=0, return_counts=True)
    return int(np.sum(cuentas == 1)), int(np.sum(cuentas > 2))


def _signed_volume(triangles: np.ndarray) -> float:
    """Suma de tetraedros contra el origen. El signo es la orientacion."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def _describe_problems(boundary: int, non_manifold: int, degenerate: int) -> list[str]:
    problemas: list[str] = []
    if boundary:
        problemas.append(
            f"{boundary} aristas de borde: la malla no es estanca y el laminador "
            "no puede saber que queda adentro."
        )
    if non_manifold:
        problemas.append(
            f"{non_manifold} aristas con mas de dos caras: la geometria se "
            "bifurca y el laminador tiene que adivinar."
        )
    if degenerate:
        problemas.append(
            f"{degenerate} triangulos de area cero: no aportan superficie y "
            "engordan el archivo."
        )
    return problemas


def inspect_mesh(triangles: np.ndarray) -> MeshReport:
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(
            f"Se esperaban triangulos con forma (n, 3, 3) y llegaron {triangles.shape}."
        )

    areas = _triangle_areas(triangles)
    utiles = triangles[areas > MIN_TRIANGLE_AREA]
    degenerados = int(triangles.shape[0] - utiles.shape[0])

    vertices, caras = weld_vertices(utiles) if utiles.size else (np.empty((0, 3)), np.empty((0, 3), dtype=int))
    caras, colapsadas = _drop_faces_collapsed_by_welding(caras)
    degenerados += colapsadas
    borde, bifurcadas = _edge_counts(caras) if caras.size else (0, 0)

    esquina_min = triangles.reshape(-1, 3).min(axis=0)
    esquina_max = triangles.reshape(-1, 3).max(axis=0)

    return MeshReport(
        triangle_count=int(triangles.shape[0]),
        vertex_count=int(vertices.shape[0]),
        boundary_edges=borde,
        non_manifold_edges=bifurcadas,
        degenerate_triangles=degenerados,
        # El valor absoluto porque una malla con las normales invertidas mide lo
        # mismo: el signo es orientacion, no tamano.
        volume=abs(_signed_volume(utiles)) if borde == 0 and utiles.size else None,
        size=tuple(float(v) for v in (esquina_max - esquina_min)),
        problems=_describe_problems(borde, bifurcadas, degenerados),
    )
