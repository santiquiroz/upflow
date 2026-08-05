"""Placa rectangular con agujeros redondos, exacta y cerrada.

Desbloquea la pieza funcional mas pedida: la placa de montaje. Un rectangulo con
agujeros en un patron, estirado un espesor.

POR QUE ASI Y NO CON UN TRIANGULADOR GENERAL. El camino "correcto" es triangular
un poligono con huecos: coser cada agujero al contorno con un puente y recortar
por orejas. Se intento y falla con mas de un agujero — el rayo del segundo cruza
el puente del primero, y ahi hace falta la refinacion por visibilidad de Eberly.
Un triangulador sutilmente roto produce tapas con agujeros que PARECEN piezas
bien hechas, que es exactamente la clase de fallo que este modulo existe para
evitar.

Asi que se resuelve el caso real con una construccion que NO PUEDE fallar: la
placa se parte en una cuadricula donde cada agujero queda solo en su celda, y
cada celda con agujero se cose como una TIRA DE CUADRILATEROS entre el circulo y
el borde de la celda, con la misma cantidad de puntos de los dos lados,
emparejados por angulo. Sin heuristicas y sin casos raros.

Las celdas vecinas comparten los MISMOS puntos del borde, que es lo que evita
las grietas: una junta con puntos distintos de cada lado deja aristas de borde y
la placa entera sale no estanca.

El precio: los agujeros tienen que poder separarse con lineas rectas, que es
justo como esta cualquier patron de montaje.
"""

from __future__ import annotations

import math

import numpy as np

MIN_SEGMENTS = 8


class PolygonError(ValueError):
    pass


def _cell_bounds(centros: list[float], minimo: float, maximo: float) -> list[tuple[float, float]]:
    """Cortes a mitad de camino entre centros consecutivos."""
    ordenados = sorted(set(centros))
    limites = [minimo]
    for a, b in zip(ordenados, ordenados[1:]):
        limites.append((a + b) / 2.0)
    limites.append(maximo)
    return list(zip(limites, limites[1:]))


def _rect_point_at_angle(
    cx: float, cy: float, x0: float, y0: float, x1: float, y1: float, angulo: float
) -> tuple[float, float]:
    """Donde el rayo desde (cx, cy) con ese angulo toca el borde del rectangulo."""
    dx, dy = math.cos(angulo), math.sin(angulo)
    distancias = []
    if abs(dx) > 1e-12:
        distancias.append((x1 - cx) / dx if dx > 0 else (x0 - cx) / dx)
    if abs(dy) > 1e-12:
        distancias.append((y1 - cy) / dy if dy > 0 else (y0 - cy) / dy)
    t = min(d for d in distancias if d > 0)
    return (cx + t * dx, cy + t * dy)


def _cell_ring(
    x0: float, y0: float, x1: float, y1: float, cx: float, cy: float, radius: float, segments: int
) -> tuple[list[list], list[tuple[float, float]], list[tuple[float, float]]]:
    """Tira de cuadrilateros entre el circulo y el borde de la celda.

    Los angulos son la union de los del circulo y los de las cuatro esquinas: asi
    las esquinas quedan como puntos del anillo y el borde de la celda se recorre
    entero, sin cortar por dentro de ninguna.
    """
    if not (x0 < cx - radius and cx + radius < x1 and y0 < cy - radius and cy + radius < y1):
        raise PolygonError(
            f"El agujero en ({cx:.1f}, {cy:.1f}) de {radius * 2:.1f} mm de diametro no cabe "
            "en su celda: "
            "esta muy cerca del borde de la placa o de otro agujero."
        )

    angulos = [2 * math.pi * i / segments for i in range(segments)]
    angulos += [
        math.atan2(ey - cy, ex - cx) % (2 * math.pi)
        for ex, ey in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    ]
    angulos = sorted(set(angulos))

    circulo = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angulos]
    marco = [_rect_point_at_angle(cx, cy, x0, y0, x1, y1, a) for a in angulos]

    caras: list[list] = []
    for i in range(len(angulos)):
        j = (i + 1) % len(angulos)
        caras.append([circulo[i], marco[i], marco[j]])
        caras.append([circulo[i], marco[j], circulo[j]])
    return caras, circulo, marco


def _fan(boundary: list[tuple[float, float]]) -> list[list]:
    """Abanico desde el centro. Vale siempre porque el borde es convexo."""
    cx = sum(p[0] for p in boundary) / len(boundary)
    cy = sum(p[1] for p in boundary) / len(boundary)
    centro = (cx, cy)
    return [
        [boundary[i], boundary[(i + 1) % len(boundary)], centro] for i in range(len(boundary))
    ]


def _cell_boundary(
    x0: float, y0: float, x1: float, y1: float, extras: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Las cuatro esquinas mas los puntos que dejaron las celdas vecinas."""
    tol = 1e-9
    abajo = sorted([p for p in extras if abs(p[1] - y0) < tol], key=lambda p: p[0])
    derecha = sorted([p for p in extras if abs(p[0] - x1) < tol], key=lambda p: p[1])
    arriba = sorted([p for p in extras if abs(p[1] - y1) < tol], key=lambda p: -p[0])
    izquierda = sorted([p for p in extras if abs(p[0] - x0) < tol], key=lambda p: -p[1])

    borde = [(x0, y0)] + abajo + [(x1, y0)] + derecha + [(x1, y1)] + arriba + [(x0, y1)] + izquierda
    salida: list[tuple[float, float]] = []
    for p in borde:
        if not salida or abs(p[0] - salida[-1][0]) > tol or abs(p[1] - salida[-1][1]) > tol:
            salida.append(p)
    return salida


def _quad(a, b, c, d) -> list[list]:
    return [[a, b, c], [a, c, d]]


def plate_with_holes(
    *,
    width: float,
    depth: float,
    thickness: float,
    holes: list[tuple[float, float, float]],
    segments: int = 32,
) -> np.ndarray:
    """Placa de `width` x `depth` x `thickness` con agujeros `(cx, cy, diametro)`."""
    if width <= 0 or depth <= 0 or thickness <= 0:
        raise PolygonError("La placa necesita las tres medidas mayores que cero.")
    if segments < MIN_SEGMENTS:
        raise PolygonError(f"Con {segments} lados el agujero no seria redondo.")

    if not holes:
        from app.services.parametric_parts import box

        return box(x=width, y=depth, z=thickness)

    columnas = _cell_bounds([h[0] for h in holes], 0.0, width)
    filas = _cell_bounds([h[1] for h in holes], 0.0, depth)

    planas: list[list] = []
    circulos: list[list[tuple[float, float]]] = []
    # Los puntos que cada anillo deja sobre el borde de su celda: las celdas
    # vecinas tienen que usarlos o queda una grieta en la junta.
    en_borde: dict[tuple, list[tuple[float, float]]] = {}
    con_agujero: dict[tuple, tuple[float, float, float]] = {}

    for x0, x1 in columnas:
        for y0, y1 in filas:
            dentro = [h for h in holes if x0 <= h[0] <= x1 and y0 <= h[1] <= y1]
            if len(dentro) > 1:
                raise PolygonError(
                    "Dos agujeros caen en la misma celda: el patron tiene que poder "
                    "separarse con lineas rectas."
                )
            if dentro:
                con_agujero[(x0, y0, x1, y1)] = dentro[0]

    for (x0, y0, x1, y1), (cx, cy, diametro) in con_agujero.items():
        caras, circulo, marco = _cell_ring(x0, y0, x1, y1, cx, cy, diametro / 2.0, segments)
        planas += caras
        circulos.append(circulo)
        en_borde[(x0, y0, x1, y1)] = marco

    todos_los_puntos = [p for marco in en_borde.values() for p in marco]
    for x0, x1 in columnas:
        for y0, y1 in filas:
            if (x0, y0, x1, y1) in con_agujero:
                continue
            tol = 1e-9
            extras = [
                p
                for p in todos_los_puntos
                if x0 - tol <= p[0] <= x1 + tol and y0 - tol <= p[1] <= y1 + tol
            ]
            planas += _fan(_cell_boundary(x0, y0, x1, y1, extras))

    caras: list[list] = []
    for tri in planas:
        abajo = [(p[0], p[1], 0.0) for p in tri]
        arriba = [(p[0], p[1], thickness) for p in tri]
        caras.append([abajo[0], abajo[2], abajo[1]])
        caras.append([arriba[0], arriba[1], arriba[2]])

    # Pared exterior: los puntos del perimetro incluyen los que dejaron los
    # anillos sobre el borde de la placa.
    perimetro = _cell_boundary(
        0.0, 0.0, width, depth,
        [
            p
            for p in todos_los_puntos
            if abs(p[0]) < 1e-9 or abs(p[0] - width) < 1e-9 or abs(p[1]) < 1e-9 or abs(p[1] - depth) < 1e-9
        ],
    )
    for i in range(len(perimetro)):
        a, b = perimetro[i], perimetro[(i + 1) % len(perimetro)]
        caras += _quad(
            (a[0], a[1], 0.0), (b[0], b[1], 0.0), (b[0], b[1], thickness), (a[0], a[1], thickness)
        )

    for circulo in circulos:
        for i in range(len(circulo)):
            a, b = circulo[i], circulo[(i + 1) % len(circulo)]
            caras += _quad(
                (a[0], a[1], 0.0), (a[0], a[1], thickness), (b[0], b[1], thickness), (b[0], b[1], 0.0)
            )

    return np.asarray(caras, dtype=np.float64)
