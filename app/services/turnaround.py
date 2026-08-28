"""Parte una hoja de turnaround en sus vistas y mide donde esta el dibujo.

Una hoja de personaje trae varias vistas en fila sobre fondo blanco, y a veces
una barra de altura a un lado o una paleta de color al otro. Se separan por las
COLUMNAS VACIAS, que es lo unico que todas las hojas tienen en comun: no hay
formato estandar, ni cantidad fija de vistas, ni posiciones.

`ink_bounds` existe por un error medido: si se escala una vista por el tamano de
su ARCHIVO en vez de por el dibujo que tiene adentro, un recorte con margen deja
al personaje mas chico que la altura pedida, y toda la referencia queda mintiendo
por ese margen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

# Por debajo de esto es tinta. El fondo de una hoja exportada nunca es 255 puro
# en todos los pixeles: el antialias y el JPEG dejan grises muy claros que
# cuentan como fondo.
WHITE_THRESHOLD = 244

# Un panel mas angosto que esto no es una vista ni por asomo.
MIN_PANEL_WIDTH_RATIO = 0.06

# Cuanta de su propia caja tiene que llenar un panel para ser un personaje.
# El ancho solo no alcanza, y esta medido: la barra de altura de una hoja real
# media 129 px de ancho —mas que el 6% del pliego— y se colaba como si fuera la
# vista frontal, corriendo el nombre de todas las demas una posicion. Una barra
# es una linea con dos topes: casi todo su rectangulo es blanco.
MIN_PANEL_INK_DENSITY = 0.08

# Cuanto puede diferir la altura de un panel respecto de la mediana de sus
# companeros y seguir siendo la misma figura. Las cuatro vistas de un turnaround
# miden practicamente lo mismo de alto (medido en una hoja real: 646, 656, 652 y
# 655 px); una paleta de color o una nota al margen no.
HEIGHT_TOLERANCE = 0.12

# Cuantas columnas vacias seguidas separan dos vistas. Muy chico parte un
# personaje en dos por el hueco entre las piernas; muy grande junta dos vistas.
DEFAULT_MIN_GAP = 12


@dataclass(frozen=True, slots=True)
class Box:
    """Caja en pixeles, origen arriba a la izquierda, como PIL."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


class EmptySheetError(ValueError):
    """La hoja no tiene tinta: no hay nada que partir ni que medir."""


class UnreadableSheetError(ValueError):
    """El archivo no es una imagen que se pueda abrir.

    Se traduce ACA y no en la capa HTTP: PIL tira `UnidentifiedImageError`,
    `OSError` o `ValueError` segun con que se tropiece, y hacer que la ruta
    conozca esos tres nombres la obliga a importar PIL para atraparlos.
    """


def open_sheet(sheet_path: Path) -> Image.Image:
    """Abre la hoja en RGB, o dice que no se puede en vez de reventar en 500."""
    try:
        with Image.open(sheet_path) as hoja:
            return hoja.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise UnreadableSheetError("El archivo no es una imagen legible.") from exc


def ink_mask(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L")) < WHITE_THRESHOLD


def ink_bounds(image: Image.Image) -> Box:
    """La caja del dibujo dentro de la imagen, sin el margen blanco."""
    mascara = ink_mask(image)
    columnas = np.where(mascara.any(axis=0))[0]
    filas = np.where(mascara.any(axis=1))[0]
    if not columnas.size or not filas.size:
        raise EmptySheetError("la imagen no tiene tinta")
    return Box(int(columnas[0]), int(filas[0]), int(columnas[-1]) + 1, int(filas[-1]) + 1)


def column_runs(has_ink: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Tramos de columnas con tinta, cortando donde hay `min_gap` vacias seguidas."""
    tramos: list[tuple[int, int]] = []
    inicio: int | None = None
    vacias = 0
    for x, con_tinta in enumerate(has_ink):
        if con_tinta:
            if inicio is None:
                inicio = x
            vacias = 0
            continue
        if inicio is None:
            continue
        vacias += 1
        if vacias >= min_gap:
            tramos.append((inicio, x - vacias + 1))
            inicio = None
    if inicio is not None:
        tramos.append((inicio, len(has_ink)))
    return tramos


def ink_density(mask: np.ndarray, box: Box) -> float:
    recorte = mask[box.y0 : box.y1, box.x0 : box.x1]
    area = recorte.size
    return float(recorte.sum()) / area if area else 0.0


def dominant_height_group(boxes: list[Box]) -> list[Box]:
    """Los paneles que miden lo mismo de alto que la mayoria.

    Se queda con el grupo MAS NUMEROSO, no con el mas alto: en una hoja de
    turnaround las figuras son mayoria por definicion, y lo que sobra —paleta,
    leyenda, nota— es lo raro. Con empate gana el grupo mas alto, que es la
    figura y no una fila de muestras de color.
    """
    if len(boxes) < 2:
        return boxes

    grupos: list[list[Box]] = []
    for caja in sorted(boxes, key=lambda b: b.height, reverse=True):
        for grupo in grupos:
            referencia = grupo[0].height
            if abs(caja.height - referencia) / referencia <= HEIGHT_TOLERANCE:
                grupo.append(caja)
                break
        else:
            grupos.append([caja])

    mejor = max(grupos, key=lambda grupo: (len(grupo), grupo[0].height))
    return sorted(mejor, key=lambda caja: caja.x0)


def panel_boxes(image: Image.Image) -> list[Box]:
    """Todo bloque de dibujo separado por columnas blancas, de izquierda a derecha.

    Generico a proposito: descarta lo que no puede ser un dibujo —demasiado
    angosto, o casi todo blanco como una barra de altura— y nada mas. Quedarse
    solo con las figuras de un turnaround es otra pregunta, y la contesta
    `character_view_boxes`.
    """
    mascara = ink_mask(image)
    if not mascara.any():
        raise EmptySheetError("la hoja no tiene tinta")

    ancho_minimo = image.width * MIN_PANEL_WIDTH_RATIO
    cajas: list[Box] = []
    for x0, x1 in column_runs(mascara.any(axis=0), DEFAULT_MIN_GAP):
        if (x1 - x0) < ancho_minimo:
            continue
        filas = np.where(mascara[:, x0:x1].any(axis=1))[0]
        caja = Box(x0, int(filas[0]), x1, int(filas[-1]) + 1)
        if ink_density(mascara, caja) < MIN_PANEL_INK_DENSITY:
            continue
        cajas.append(caja)
    return cajas


def character_view_boxes(image: Image.Image) -> list[Box]:
    """Solo las vistas del personaje: se cae la paleta, la leyenda y la nota.

    Las cuatro vistas de un turnaround miden lo mismo de alto porque son la
    misma figura girando. Cualquier otra cosa en la hoja mide distinto, y esa
    es toda la regla.
    """
    return dominant_height_group(panel_boxes(image))
