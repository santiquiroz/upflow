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
from app.services.polygon_mesh import PolygonError, plate_with_holes
from app.services.rectilinear_plate import rectilinear_plate


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
        id="plate",
        label_key="part.plate",
        description_key="part.plate.description",
        params=(
            PartParam("width", "part.param.width", 60.0, 1.0),
            PartParam("depth", "part.param.depth", 40.0, 1.0),
            PartParam("thickness", "part.param.thickness", 4.0, 0.4),
            PartParam("hole_diameter", "part.param.holeDiameter", 6.4, 0.5),
            PartParam("margin", "part.param.margin", 10.0, 1.0),
        ),
    ),
    PartKind(
        id="bracket",
        label_key="part.bracket",
        description_key="part.bracket.description",
        params=(
            PartParam("arm_a", "part.param.armA", 60.0, 5.0),
            PartParam("arm_b", "part.param.armB", 40.0, 5.0),
            PartParam("leg_width", "part.param.legWidth", 15.0, 3.0),
            PartParam("thickness", "part.param.thickness", 5.0, 0.8),
            PartParam("hole_diameter", "part.param.holeDiameter", 6.4, 0.5),
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

def _mounting_plate(
    *, width: float, depth: float, thickness: float, hole_diameter: float, margin: float
):
    """Placa con un agujero en cada esquina, a `margin` del borde.

    Cuatro agujeros en las esquinas es el patron de montaje de casi todo. Un
    patron cualquiera vendra despues; empezar por el general habria sido
    empezar por el caso que nadie pide.
    """
    esquinas = [
        (margin, margin, hole_diameter),
        (width - margin, margin, hole_diameter),
        (margin, depth - margin, hole_diameter),
        (width - margin, depth - margin, hole_diameter),
    ]
    return plate_with_holes(
        width=width, depth=depth, thickness=thickness, holes=esquinas
    )


def _corner_bracket(
    *, arm_a: float, arm_b: float, leg_width: float, thickness: float, hole_diameter: float
):
    """Escuadra plana en L, con un agujero cerca del extremo de cada brazo.

    Plana y no en angulo recto a proposito: una escuadra plana se imprime
    acostada, sin un solo voladizo, y es mas fuerte porque las capas corren a lo
    largo del esfuerzo. La version en angulo necesita soporte y se parte justo
    por la esquina, que es donde trabaja.
    """
    if leg_width >= min(arm_a, arm_b):
        raise PartError(
            f"El ancho del brazo ({leg_width} mm) tiene que ser menor que los dos "
            f"brazos ({arm_a} y {arm_b} mm), o no queda una L."
        )
    solido = [
        (0.0, 0.0, arm_a, leg_width),
        (0.0, leg_width, leg_width, arm_b),
    ]
    agujeros = [
        (arm_a - leg_width / 2.0, leg_width / 2.0, hole_diameter),
        (leg_width / 2.0, arm_b - leg_width / 2.0, hole_diameter),
    ]
    return rectilinear_plate(solid=solido, thickness=thickness, holes=agujeros)


_BUILDERS = {
    "tube": tube,
    "box": box,
    "cylinder": cylinder,
    "plate": _mounting_plate,
    "bracket": _corner_bracket,
}


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
    try:
        return constructor(**params)
    except PolygonError as exc:
        # La placa habla su propio dialecto de error; afuera todo es PartError.
        raise PartError(str(exc)) from exc
