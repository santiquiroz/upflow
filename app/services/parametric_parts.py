"""Piezas con cotas EXACTAS, construidas por formula y no por un modelo.

Esta es la respuesta al hallazgo central de la investigacion de motores 3D: un
generador de malla no sabe que la pieza tiene que medir 80 mm — no hay cota que
salga de un prompt — y una pieza que tiene que encajar necesita la medida exacta.
Aca la geometria se construye analiticamente: la cota sale de la formula.

Sale estanca por construccion, lo cual NO se asume: se verifica con el mismo
banco (`mesh_inspect`) que verifica cualquier STL bajado de internet. Que el
generador y el verificador sean independientes es lo que hace que la
verificacion signifique algo.

El espaciador, el buje, la arandela y el separador son la MISMA pieza con otros
numeros, y son el pan de cada dia de una reparacion mecanica.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_SEGMENTS = 64
MIN_SEGMENTS = 3
# Por debajo del diametro de la boquilla, el laminador descarta la pared en
# silencio y la pieza sale con un agujero donde iba material.
DEFAULT_MIN_WALL_MM = 0.4


class PartError(ValueError):
    pass


def _require_positive(**medidas: float) -> None:
    for nombre, valor in medidas.items():
        if valor <= 0:
            raise PartError(f"{nombre} tiene que ser mayor que cero, no {valor!r}.")


def _quad(a, b, c, d) -> list[list]:
    """Un rectangulo como dos triangulos, con el bobinado consistente."""
    return [[a, b, c], [a, c, d]]


def box(*, x: float, y: float, z: float) -> np.ndarray:
    """Un prisma recto. La pieza mas simple, y la base de casi todo lo demas."""
    _require_positive(x=x, y=y, z=z)
    e = [
        (0, 0, 0), (x, 0, 0), (x, y, 0), (0, y, 0),
        (0, 0, z), (x, 0, z), (x, y, z), (0, y, z),
    ]
    caras: list[list] = []
    caras += _quad(e[0], e[3], e[2], e[1])  # abajo, mirando -Z
    caras += _quad(e[4], e[5], e[6], e[7])  # arriba, mirando +Z
    caras += _quad(e[0], e[1], e[5], e[4])
    caras += _quad(e[1], e[2], e[6], e[5])
    caras += _quad(e[2], e[3], e[7], e[6])
    caras += _quad(e[3], e[0], e[4], e[7])
    return np.asarray(caras, dtype=np.float64)


def _ring(radius: float, height: float, segments: int) -> np.ndarray:
    angulos = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    return np.stack(
        [radius * np.cos(angulos), radius * np.sin(angulos), np.full(segments, height)],
        axis=1,
    )


def _validate_segments(segments: int) -> None:
    if segments < MIN_SEGMENTS:
        raise PartError(
            f"Con {segments} lados no hay circulo. El minimo son {MIN_SEGMENTS}."
        )


def cylinder(
    *, diameter: float, height: float, segments: int = DEFAULT_SEGMENTS
) -> np.ndarray:
    """Un cilindro recto, triangulado como prisma de `segments` lados.

    El diametro exacto lo tienen los VERTICES: entre dos vertices la superficie
    queda por dentro del circulo teorico, que es como funciona cualquier malla y
    por eso el volumen converge desde abajo.
    """
    _require_positive(diameter=diameter, height=height)
    _validate_segments(segments)

    radio = diameter / 2.0
    abajo, arriba = _ring(radio, 0.0, segments), _ring(radio, height, segments)
    centro_abajo = np.array([0.0, 0.0, 0.0])
    centro_arriba = np.array([0.0, 0.0, height])

    caras: list[list] = []
    for i in range(segments):
        j = (i + 1) % segments
        caras.append([centro_abajo, abajo[j], abajo[i]])
        caras.append([centro_arriba, arriba[i], arriba[j]])
        caras += _quad(abajo[i], abajo[j], arriba[j], arriba[i])
    return np.asarray(caras, dtype=np.float64)


def tube(
    *,
    outer_diameter: float,
    inner_diameter: float,
    height: float,
    segments: int = DEFAULT_SEGMENTS,
    min_wall_mm: float = DEFAULT_MIN_WALL_MM,
) -> np.ndarray:
    """Un tubo: espaciador, buje, arandela o separador segun los numeros.

    El agujero es GEOMETRIA, no una marca: el volumen que sale es el del anillo,
    no el del cilindro entero, y eso es lo que se verifica.
    """
    _require_positive(
        outer_diameter=outer_diameter, inner_diameter=inner_diameter, height=height
    )
    _validate_segments(segments)
    if inner_diameter >= outer_diameter:
        raise PartError(
            f"El agujero interior ({inner_diameter} mm) no puede ser igual ni mayor "
            f"que el diametro exterior ({outer_diameter} mm): no quedaria pieza."
        )
    pared = (outer_diameter - inner_diameter) / 2.0
    if pared < min_wall_mm:
        raise PartError(
            f"La pared quedaria de {pared:.2f} mm y la boquilla deposita "
            f"{min_wall_mm} mm: el laminador la descartaria en silencio y la pieza "
            "saldria con un agujero donde iba material."
        )

    r_ext, r_int = outer_diameter / 2.0, inner_diameter / 2.0
    ext_abajo, ext_arriba = _ring(r_ext, 0.0, segments), _ring(r_ext, height, segments)
    int_abajo, int_arriba = _ring(r_int, 0.0, segments), _ring(r_int, height, segments)

    caras: list[list] = []
    for i in range(segments):
        j = (i + 1) % segments
        # Coronas de arriba y abajo: el anillo entre los dos radios.
        caras += _quad(ext_abajo[i], int_abajo[i], int_abajo[j], ext_abajo[j])
        caras += _quad(ext_arriba[i], ext_arriba[j], int_arriba[j], int_arriba[i])
        # Pared exterior, mirando afuera.
        caras += _quad(ext_abajo[i], ext_abajo[j], ext_arriba[j], ext_arriba[i])
        # Pared del agujero, mirando adentro.
        caras += _quad(int_abajo[j], int_abajo[i], int_arriba[i], int_arriba[j])
    return np.asarray(caras, dtype=np.float64)
