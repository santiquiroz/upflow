"""Que piezas parametricas existen y que numeros pide cada una.

El catalogo vive aparte del generador para que la pantalla pueda armar el
formulario sin conocer la geometria: pide la lista, dibuja los campos, manda los
numeros. Agregar una pieza nueva es una entrada aca y una funcion alla.

Las etiquetas son CLAVES de traduccion, no copia: la oracion la arma el frontend
en el idioma activo.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.parametric_parts import PartError, box, cylinder, tube


@dataclass(frozen=True, slots=True)
class PartParam:
    name: str
    label_key: str
    default: float
    minimum: float


@dataclass(frozen=True, slots=True)
class PartKind:
    id: str
    label_key: str
    description_key: str
    params: tuple[PartParam, ...]


PART_KINDS: tuple[PartKind, ...] = (
    PartKind(
        id="tube",
        label_key="part.tube",
        description_key="part.tube.description",
        params=(
            PartParam("outer_diameter", "part.param.outerDiameter", 20.0, 0.1),
            PartParam("inner_diameter", "part.param.innerDiameter", 8.4, 0.1),
            PartParam("height", "part.param.height", 12.0, 0.1),
        ),
    ),
    PartKind(
        id="box",
        label_key="part.box",
        description_key="part.box.description",
        params=(
            PartParam("x", "part.param.x", 30.0, 0.1),
            PartParam("y", "part.param.y", 20.0, 0.1),
            PartParam("z", "part.param.z", 10.0, 0.1),
        ),
    ),
    PartKind(
        id="cylinder",
        label_key="part.cylinder",
        description_key="part.cylinder.description",
        params=(
            PartParam("diameter", "part.param.diameter", 12.0, 0.1),
            PartParam("height", "part.param.height", 40.0, 0.1),
        ),
    ),
)

_BUILDERS = {"tube": tube, "box": box, "cylinder": cylinder}


def build_part(kind: str, params: dict[str, float]):
    constructor = _BUILDERS.get(kind)
    if constructor is None:
        raise PartError(
            f"No conozco la pieza {kind!r}. Las que si: {', '.join(sorted(_BUILDERS))}."
        )
    esperados = {p.name for k in PART_KINDS if k.id == kind for p in k.params}
    faltantes = esperados - set(params)
    if faltantes:
        raise PartError(f"Faltan medidas: {', '.join(sorted(faltantes))}.")
    sobrantes = set(params) - esperados
    if sobrantes:
        raise PartError(f"Medidas que esta pieza no usa: {', '.join(sorted(sobrantes))}.")
    return constructor(**params)
