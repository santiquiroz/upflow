"""Placa de contorno RECTILINEO con agujeros: la L, la T, el marco.

Generaliza `polygon_mesh.plate_with_holes` de un rectangulo a cualquier figura
formada por rectangulos que no se solapan. Con eso aparece la escuadra, que es
—junto con la placa y el espaciador— la otra pieza basica de una reparacion
mecanica.

LA REGLA QUE LO HACE UN SOLO SOLIDO: las paredes se emiten SOLO donde una celda
llena da al vacio o al exterior. Emitir tambien la pared de una junta interna
dejaria esa arista con cuatro caras, y el banco lo rechaza — con razon. Medido
antes de escribir esto: dos cajas apoyadas una en otra dan estanca=True pero
manifold=False, o sea no imprimible.

La cuadricula se arma sobre TODOS los rectangulos a la vez, asi que las celdas
vecinas comparten exactamente los mismos puntos y no queda ninguna junta abierta.
"""

from __future__ import annotations

import numpy as np

from app.services.polygon_mesh import (
    MIN_SEGMENTS,
    PolygonError,
    _cell_boundary,
    _cell_ring,
    _fan,
    _quad,
)

TOL = 1e-9


def _midpoints(valores: list[float]) -> list[float]:
    ordenados = sorted(set(valores))
    return [(a + b) / 2.0 for a, b in zip(ordenados, ordenados[1:])]


def _grid_lines(cortes: list[float], minimo: float, maximo: float) -> list[float]:
    return sorted({minimo, maximo, *(v for v in cortes if minimo < v < maximo)})


def _is_filled(x: float, y: float, llenas: set) -> bool:
    return any(x0 < x < x1 and y0 < y < y1 for x0, y0, x1, y1 in llenas)


def _points_on_edge(a, b, extras: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """El borde de `a` a `b` con los puntos que dejaron los anillos, en orden."""
    horizontal = abs(a[1] - b[1]) < TOL
    if horizontal:
        sobre = [
            p for p in extras
            if abs(p[1] - a[1]) < TOL and min(a[0], b[0]) < p[0] < max(a[0], b[0])
        ]
        orden = sorted(sobre, key=lambda p: p[0], reverse=a[0] > b[0])
    else:
        sobre = [
            p for p in extras
            if abs(p[0] - a[0]) < TOL and min(a[1], b[1]) < p[1] < max(a[1], b[1])
        ]
        orden = sorted(sobre, key=lambda p: p[1], reverse=a[1] > b[1])

    salida = [a]
    for p in orden:
        if abs(p[0] - salida[-1][0]) > TOL or abs(p[1] - salida[-1][1]) > TOL:
            salida.append(p)
    salida.append(b)
    return salida


def rectilinear_plate(
    *,
    solid: list[tuple[float, float, float, float]],
    thickness: float,
    holes: list[tuple[float, float, float]],
    segments: int = 32,
) -> np.ndarray:
    """`solid` son los rectangulos que forman la pieza, sin solaparse."""
    if thickness <= 0:
        raise PolygonError("El espesor tiene que ser mayor que cero.")
    if not solid:
        raise PolygonError("No hay ningun rectangulo que formar.")
    if segments < MIN_SEGMENTS:
        raise PolygonError(f"Con {segments} lados el agujero no seria redondo.")

    # Los cortes salen de los bordes de los rectangulos MAS los puntos medios
    # entre centros de agujeros: cortar EN el centro dejaria el agujero partido
    # por la junta.
    xs = _grid_lines(
        [r[0] for r in solid] + [r[2] for r in solid] + _midpoints([h[0] for h in holes]),
        min(r[0] for r in solid), max(r[2] for r in solid),
    )
    ys = _grid_lines(
        [r[1] for r in solid] + [r[3] for r in solid] + _midpoints([h[1] for h in holes]),
        min(r[1] for r in solid), max(r[3] for r in solid),
    )

    celdas = [
        (x0, y0, x1, y1)
        for x0, x1 in zip(xs, xs[1:])
        for y0, y1 in zip(ys, ys[1:])
    ]
    llenas = {
        c for c in celdas
        if any(
            rx0 - TOL <= c[0] and c[2] <= rx1 + TOL and ry0 - TOL <= c[1] and c[3] <= ry1 + TOL
            for rx0, ry0, rx1, ry1 in solid
        )
    }
    if not llenas:
        raise PolygonError("Los rectangulos no forman ninguna celda entera.")

    con_agujero: dict[tuple, tuple[float, float, float]] = {}
    for celda in llenas:
        dentro = [
            h for h in holes
            if celda[0] <= h[0] <= celda[2] and celda[1] <= h[1] <= celda[3]
        ]
        if len(dentro) > 1:
            raise PolygonError(
                "Dos agujeros caen en la misma celda: el patron tiene que poder "
                "separarse con lineas rectas."
            )
        if dentro:
            con_agujero[celda] = dentro[0]

    sin_celda = [
        h for h in holes
        if not any(c[0] <= h[0] <= c[2] and c[1] <= h[1] <= c[3] for c in llenas)
    ]
    if sin_celda:
        raise PolygonError(
            f"Hay {len(sin_celda)} agujero(s) fuera de la pieza: no tienen material que atravesar."
        )

    planas: list[list] = []
    circulos: list[list[tuple[float, float]]] = []
    marcos: list[tuple[float, float]] = []
    for celda, (cx, cy, diametro) in con_agujero.items():
        caras, circulo, marco = _cell_ring(*celda, cx, cy, diametro / 2.0, segments)
        planas += caras
        circulos.append(circulo)
        marcos += marco

    for celda in llenas:
        if celda in con_agujero:
            continue
        x0, y0, x1, y1 = celda
        extras = [
            p for p in marcos
            if x0 - TOL <= p[0] <= x1 + TOL and y0 - TOL <= p[1] <= y1 + TOL
        ]
        planas += _fan(_cell_boundary(x0, y0, x1, y1, extras))

    caras: list[list] = []
    for tri in planas:
        abajo = [(p[0], p[1], 0.0) for p in tri]
        arriba = [(p[0], p[1], thickness) for p in tri]
        caras.append([abajo[0], abajo[2], abajo[1]])
        caras.append([arriba[0], arriba[1], arriba[2]])

    # El vecino se sondea JUSTO del otro lado del borde, no a media celda: las
    # celdas no miden todas lo mismo, y medio ancho de una celda chica cae fuera
    # del vecino grande. Ese error dejaba paredes en juntas internas — cuatro
    # caras en una arista — y solo se veia en la pieza SIN agujeros, donde la
    # cuadricula queda despareja.
    salto = min(min(c[2] - c[0], c[3] - c[1]) for c in llenas) / 100.0
    for x0, y0, x1, y1 in llenas:
        bordes = (
            (((x0, y0), (x1, y0)), ((x0 + x1) / 2, y0 - salto)),
            (((x1, y0), (x1, y1)), (x1 + salto, (y0 + y1) / 2)),
            (((x1, y1), (x0, y1)), ((x0 + x1) / 2, y1 + salto)),
            (((x0, y1), (x0, y0)), (x0 - salto, (y0 + y1) / 2)),
        )
        for (a, b), (vx, vy) in bordes:
            if _is_filled(vx, vy, llenas):
                continue
            puntos = _points_on_edge(a, b, marcos)
            for p, q in zip(puntos, puntos[1:]):
                caras += _quad(
                    (p[0], p[1], 0.0), (q[0], q[1], 0.0),
                    (q[0], q[1], thickness), (p[0], p[1], thickness),
                )

    for circulo in circulos:
        for i in range(len(circulo)):
            a, b = circulo[i], circulo[(i + 1) % len(circulo)]
            caras += _quad(
                (a[0], a[1], 0.0), (a[0], a[1], thickness),
                (b[0], b[1], thickness), (b[0], b[1], 0.0),
            )

    return np.asarray(caras, dtype=np.float64)
