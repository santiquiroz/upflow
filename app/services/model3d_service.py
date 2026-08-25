"""Las operaciones de modelado 3D, encadenables de a una.

Cada funcion hace UNA cosa y devuelve el veredicto junto al archivo — la misma
regla que ya rige el carril de impresion: una malla que no cierra no es una
pieza, y decir "listo" sobre eso es el peor falso positivo, el que da confianza.

Son atomicas a proposito. La persona que usa la pantalla aprieta un boton por
vez; un agente que llega por MCP encadena varias y necesita medir entre medio.
Las dos cosas se sirven con las mismas piezas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.services import blender_service
from app.services.turnaround import Box, character_view_boxes, ink_bounds

# Como se llama cada vista en la escena. El orden es el de una hoja de
# turnaround estandar, de izquierda a derecha.
VIEW_ORDER = ("front", "side", "back", "side_left")

AUDIT_SCRIPT = "audit_mesh.py"
REFERENCE_SCRIPT = "build_reference_scene.py"


@dataclass(frozen=True, slots=True)
class DetectedView:
    name: str
    image: Path
    ink: Box

    def as_payload(self) -> dict[str, Any]:
        return {"image": str(self.image), "inkBox": list(self.ink.as_tuple())}


def audit_mesh(settings: Settings, mesh_path: Path) -> dict[str, Any]:
    """Mide una malla sin tocarla."""
    return blender_service.run_script(settings, AUDIT_SCRIPT, {"mesh": str(mesh_path)})


def detect_views(sheet_path: Path, *, names: tuple[str, ...] = VIEW_ORDER) -> list[DetectedView]:
    """Las vistas de una hoja, nombradas por posicion y ya medidas.

    Nombrar por posicion es una convencion, no una deduccion: no hay forma de
    saber mirando los pixeles si el tercer panel es la espalda o un tres
    cuartos. Quien lo sepa —la persona o el agente— renombra despues.
    """
    with Image.open(sheet_path) as hoja:
        rgb = hoja.convert("RGB")
        cajas = character_view_boxes(rgb)
        return [
            DetectedView(name=nombre, image=sheet_path, ink=caja)
            for nombre, caja in zip(names, cajas)
        ]


def split_views(
    sheet_path: Path,
    out_dir: Path,
    *,
    names: tuple[str, ...] = VIEW_ORDER,
) -> list[DetectedView]:
    """Escribe una imagen por vista y devuelve cada una con su caja de tinta.

    La caja se recalcula sobre el RECORTE y no se hereda de la hoja: son
    sistemas de coordenadas distintos, y pasar la de la hoja al script de
    Blender lo haria escalar por un margen que ya no existe.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(sheet_path) as hoja:
        rgb = hoja.convert("RGB")
        cajas = character_view_boxes(rgb)
        detectadas: list[DetectedView] = []
        for nombre, caja in zip(names, cajas):
            recorte = rgb.crop(caja.as_tuple())
            destino = out_dir / f"{nombre}.png"
            recorte.save(destino)
            detectadas.append(DetectedView(name=nombre, image=destino, ink=ink_bounds(recorte)))
    return detectadas


def sheet_warnings(sheet_path: Path, *, expected_views: int = len(VIEW_ORDER)) -> list[str]:
    """Lo que la hoja tiene de raro, dicho antes de que arruine la escena.

    Contar es la unica senal confiable. Medido sobre una hoja real: dos vistas
    dibujadas tan juntas que se SUPERPONEN no dejan ninguna columna con poca
    tinta entre medio, asi que no hay corte vertical que las separe ni ancho
    sospechoso que las delate — pero el conteo baja de cuatro a tres y eso se
    ve siempre.

    Se pregunta aparte de partir: partir devuelve vistas, esto devuelve dudas,
    y mezclarlas obligaria a revisar el resultado para saber si hubo problema.
    """
    with Image.open(sheet_path) as hoja:
        cajas = character_view_boxes(hoja.convert("RGB"))

    if len(cajas) == expected_views:
        return []
    return [
        f"se detectaron {len(cajas)} vistas y se esperaban {expected_views}. "
        "Si hay dos dibujadas superpuestas no se pueden separar solas: "
        "recortalas a mano y pasalas como imagenes sueltas."
    ]


def build_reference_scene(
    settings: Settings,
    views: list[DetectedView],
    output: Path,
    *,
    height_meters: float = 1.70,
) -> dict[str, Any]:
    """La escena de Blender lista para modelar encima."""
    return blender_service.run_script(
        settings,
        REFERENCE_SCRIPT,
        {
            "views": {vista.name: vista.as_payload() for vista in views},
            "heightMeters": height_meters,
            "output": str(output),
        },
    )
