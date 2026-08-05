"""Convierte una malla sin unidades en una pieza que entra en una impresora.

El STL no tiene unidades. Ni una. Todo laminador asume milimetros, y un modelo
generado por IA no sabe que tiene que medir 80 mm: sale en la escala que se le
ocurrio al modelo. Fijar esa medida es el paso que convierte una malla en una
PIEZA, y es lo unico que ninguna generacion resuelve sola.

Despues viene la pregunta que ninguna IA contesta: ¿entra en la cama? Una Ender 3
son 220x220x250 mm. Una pieza de carro de 300 mm no entra derecha, entra en
diagonal, y enterarse recien en el laminador es tarde.
"""

from __future__ import annotations

import math

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}

# Volumen util de las camas mas comunes, en mm. Son las medidas que publica cada
# fabricante; conviene confirmarlas contra la maquina real antes de confiar en
# el ultimo milimetro.
PRINTER_BEDS: dict[str, tuple[float, float, float]] = {
    "ender-3": (220.0, 220.0, 250.0),
    "ender-3-v3": (220.0, 220.0, 250.0),
    "ender-5-plus": (350.0, 350.0, 400.0),
    "k1": (220.0, 220.0, 250.0),
    "k1-max": (300.0, 300.0, 300.0),
    "k2-plus": (350.0, 350.0, 350.0),
    "cr-10-smart-pro": (300.0, 300.0, 400.0),
}

# Margen contra el borde: una pieza que toca el limite exacto choca con el
# cabezal o con los clips de la cama.
BED_MARGIN_MM = 5.0


def scale_to_dimension(triangles: np.ndarray, *, axis: str, millimetres: float) -> np.ndarray:
    """Escala UNIFORME para que el eje elegido mida lo pedido.

    Uniforme y no por eje: estirar un solo eje deforma la pieza y un agujero
    redondo sale ovalado, que en una pieza que tiene que encajar es el final.
    """
    if axis not in AXES:
        raise ValueError(f"Eje desconocido: {axis!r}. Son 'x', 'y' o 'z'.")
    if millimetres <= 0:
        raise ValueError(f"La medida pedida tiene que ser mayor que cero, no {millimetres!r}.")

    indice = AXES[axis]
    coordenadas = triangles[:, :, indice]
    actual = float(coordenadas.max() - coordenadas.min())
    if actual <= 0:
        raise ValueError(
            f"El modelo es plano en el eje {axis}: no hay medida que escalar."
        )
    return triangles * (millimetres / actual)


def fits_on_bed(
    size: tuple[float, float, float], *, bed: tuple[float, float, float]
) -> tuple[bool, str]:
    """Dice si la pieza entra, y cuando no, cual medida sobra."""
    util = (bed[0] - BED_MARGIN_MM, bed[1] - BED_MARGIN_MM, bed[2])
    alto = size[2]
    if alto > util[2]:
        return False, (
            f"La pieza mide {alto:.0f} mm de alto y la impresora llega a {bed[2]:.0f} mm."
        )

    largo, ancho = sorted((size[0], size[1]), reverse=True)
    cama_larga, cama_corta = sorted((util[0], util[1]), reverse=True)
    if largo <= cama_larga and ancho <= cama_corta:
        return True, "Entra derecha."

    # La diagonal de la cama es mas larga que cualquiera de sus lados: una pieza
    # larga y angosta que no entra derecha suele entrar girada.
    diagonal = math.hypot(util[0], util[1])
    if largo <= diagonal and ancho <= cama_corta:
        return True, f"Entra girada en diagonal (la diagonal de la cama son {diagonal:.0f} mm)."

    return False, (
        f"La pieza mide {largo:.0f} x {ancho:.0f} mm y la cama util son "
        f"{cama_larga:.0f} x {cama_corta:.0f} mm, ni siquiera en diagonal."
    )


def smallest_bed_orientation(
    size: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Las mismas medidas, con el lado mas largo acostado.

    Acostar la pieza baja el alto — que es lo que mas suele sobrar — y de paso
    reduce el tiempo de impresion y el soporte que hace falta.
    """
    largo, medio, corto = sorted(size, reverse=True)
    return (largo, medio, corto)
