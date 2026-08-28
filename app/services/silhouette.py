"""Mide un personaje dibujado y dice de que medida se puede fiar uno.

Lo que un modelador quiere saber antes de empezar es cuantas cabezas mide, a
que altura esta la cadera y cuanto mide de ancho a cada altura. Eso se lee del
dibujo; no hace falta adivinarlo.

Lo que NO se puede leer tambien esta dicho. Medido el 2026-08-28 sobre una hoja
real: un personaje con los brazos colgando **no tiene cintura en su silueta**
—los brazos rellenan el estrechamiento— y buscarla igual devuelve el borde de
la ventana de busqueda, que es un numero inventado con cara de medicion. Por
eso cada altura viaja con el acuerdo entre las dos vistas: lo que las dos ven
en el mismo lugar es una articulacion; lo que cada una ve en otro lado es ruido.

Todo se mide sobre la SILUETA (el contorno externo por fila) y nunca sobre las
lineas interiores: un cinturon dibujado parte la fila en tres tramos y no dice
nada del ancho del cuerpo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.turnaround import ink_mask, open_sheet

# Dos vistas que ubican la misma altura mas cerca que esto estan de acuerdo.
# Es 1 cm sobre un personaje de 1,70 m: mas fino que eso es ruido de trazo.
TOLERANCIA_ACUERDO_M = 0.02

# Donde buscar el cuello: entre la coronilla y el 40% de la altura. Amplio a
# proposito — un personaje de tres cabezas y uno de ocho caen los dos adentro.
VENTANA_CUELLO = (0.10, 0.45)


@dataclass(frozen=True, slots=True)
class Banda:
    """El ancho de la silueta a una altura, en metros."""

    z: float
    ancho: float


@dataclass(frozen=True, slots=True)
class Altura:
    """Una altura leida del dibujo, con lo que opinan las dos vistas."""

    nombre: str
    z: float
    frente: float
    lado: float

    @property
    def acuerdo(self) -> bool:
        return abs(self.frente - self.lado) <= TOLERANCIA_ACUERDO_M

    @property
    def desacuerdo_cm(self) -> float:
        return round(abs(self.frente - self.lado) * 100, 1)


@dataclass(frozen=True, slots=True)
class Proporciones:
    altura_m: float
    cabeza_m: float
    alturas: tuple[Altura, ...]
    bandas_frente: tuple[Banda, ...]
    bandas_lado: tuple[Banda, ...]

    @property
    def cabezas_de_alto(self) -> float:
        return round(self.altura_m / self.cabeza_m, 2) if self.cabeza_m else 0.0

    @property
    def dudosas(self) -> tuple[str, ...]:
        return tuple(a.nombre for a in self.alturas if not a.acuerdo)


def perfil(image: Image.Image, altura_m: float) -> list[Banda]:
    """El ancho de la silueta fila por fila, de la coronilla a los pies."""
    mascara = ink_mask(image)
    alto = mascara.shape[0]
    metros_por_px = altura_m / alto

    bandas: list[Banda] = []
    for y in range(alto):
        columnas = np.where(mascara[y])[0]
        if not columnas.size:
            continue
        ancho = (int(columnas[-1] - columnas[0]) + 1) * metros_por_px
        bandas.append(Banda(z=(alto - y) * metros_por_px, ancho=ancho))
    return bandas


def _mas_angosto(bandas: list[Banda], desde: float, hasta: float) -> float:
    """La altura donde la silueta se estrecha entre dos partes mas anchas."""
    candidatas = [b for b in bandas if desde <= b.z <= hasta]
    if not candidatas:
        return (desde + hasta) / 2
    return min(candidatas, key=lambda b: b.ancho).z


def _mas_ancho(bandas: list[Banda], desde: float, hasta: float) -> float:
    candidatas = [b for b in bandas if desde <= b.z <= hasta]
    if not candidatas:
        return (desde + hasta) / 2
    return max(candidatas, key=lambda b: b.ancho).z


def _cuello(bandas: list[Banda], altura_m: float) -> float:
    desde = altura_m * (1 - VENTANA_CUELLO[1])
    hasta = altura_m * (1 - VENTANA_CUELLO[0])
    return _mas_angosto(bandas, desde, hasta)


def _cadera(bandas: list[Banda], altura_m: float) -> float:
    """El punto mas ancho de la mitad de abajo.

    En una figura de pie es la cadera o el borde del short; en las dos vistas
    cae en el mismo lugar, que es lo que lo hace confiable.
    """
    return _mas_ancho(bandas, altura_m * 0.35, altura_m * 0.60)


def ancho_en(bandas: tuple[Banda, ...] | list[Banda], z: float, tolerancia: float = 0.01) -> float:
    cercanas = [b.ancho for b in bandas if abs(b.z - z) <= tolerancia]
    return max(cercanas) if cercanas else 0.0


def medir(front_path: Path, side_path: Path, altura_m: float) -> Proporciones:
    """Las proporciones del personaje, con el acuerdo entre vistas incluido."""
    frente = perfil(open_sheet(front_path), altura_m)
    lado = perfil(open_sheet(side_path), altura_m)
    if not frente or not lado:
        raise ValueError("alguna de las vistas no tiene silueta")

    medidas = {
        "cuello": (_cuello(frente, altura_m), _cuello(lado, altura_m)),
        "cadera": (_cadera(frente, altura_m), _cadera(lado, altura_m)),
    }
    alturas = tuple(
        # La altura reportada es el promedio de las dos vistas: cuando estan de
        # acuerdo da lo mismo, y cuando no, ninguna de las dos merece ganar.
        Altura(nombre=nombre, z=round((f + l) / 2, 4), frente=round(f, 4), lado=round(l, 4))
        for nombre, (f, l) in medidas.items()
    )
    cuello = next(a for a in alturas if a.nombre == "cuello")
    return Proporciones(
        altura_m=altura_m,
        cabeza_m=round(altura_m - cuello.z, 4),
        alturas=alturas,
        bandas_frente=tuple(frente),
        bandas_lado=tuple(lado),
    )
