"""Un solo veredicto sobre si un STL se puede imprimir, y que arreglarle.

Compone lo demas: leer el archivo, verificar que la malla sea estanca y sin
bifurcaciones, llevarla a una medida real, ver si entra en la cama de la maquina
y cuanta superficie queda en voladizo.

Sirve SOLO, sin ningun modelo generativo de por medio: cualquier STL —bajado de
internet, exportado de Fusion, generado por IA— pasa por el mismo tamiz. Esa es
la gracia, y es lo que permite medir a un generador en vez de creerle: se le
pide una pieza y se la pasa por aca.

Distingue dos cosas que no son lo mismo:
  - BLOQUEOS: la pieza no se imprime asi. Hay que arreglarla.
  - CONSEJOS: se imprime, pero saldria mejor de otra manera.
Mezclarlos entrena a ignorar las dos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.services.mesh_fit import PRINTER_BEDS, fits_on_bed, scale_to_dimension
from app.services.mesh_inspect import MeshReport, inspect_mesh
from app.services.mesh_overhang import (
    DEFAULT_OVERHANG_LIMIT_DEG,
    best_print_orientation,
    overhang_report,
)
from app.services.stl_reader import StlUnreadable, read_stl

# Debajo de esta diferencia, girar la pieza no paga el trabajo de reorientarla.
WORTH_ROTATING_DELTA = 0.05


class PrintCheckUnavailable(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PrintCheckReport:
    size: tuple[float, float, float]
    mesh: MeshReport
    overhang_ratio: float
    # Lo que impide imprimir. Vacio = se puede.
    blockers: list[str] = field(default_factory=list)
    # Lo que se imprime igual pero saldria mejor de otra forma.
    advice: list[str] = field(default_factory=list)

    @property
    def can_print(self) -> bool:
        return not self.blockers


def _resolve_bed(
    printer: str | None, bed: tuple[float, float, float] | None
) -> tuple[float, float, float]:
    if bed is not None:
        return bed
    if printer is None:
        raise PrintCheckUnavailable("Hace falta una impresora o una cama.")
    conocida = PRINTER_BEDS.get(printer.strip().lower())
    if conocida is None:
        raise PrintCheckUnavailable(
            f"No conozco la impresora {printer!r}. Las que si: "
            f"{', '.join(sorted(PRINTER_BEDS))}. "
            "Tambien se puede pasar la cama a mano."
        )
    return conocida


def _rotation_advice(triangles: np.ndarray, actual: float, limit_deg: float) -> list[str]:
    mejor = best_print_orientation(triangles, limit_deg=limit_deg)
    if actual - mejor.overhang_area_ratio <= WORTH_ROTATING_DELTA:
        # Un consejo que no cambia nada entrena a ignorar los consejos.
        return []
    ejes = ("X", "Y", "Z")
    giros = ", ".join(
        f"{grados}° en {ejes[i]}" for i, grados in enumerate(mejor.rotation_deg) if grados
    )
    return [
        f"Conviene girar la pieza {giros}: el voladizo baja de "
        f"{actual:.0%} a {mejor.overhang_area_ratio:.0%} y necesita menos soporte."
    ]


def check_stl_for_printing(
    path: Path,
    *,
    printer: str | None = None,
    bed: tuple[float, float, float] | None = None,
    target_axis: str | None = None,
    target_mm: float | None = None,
    overhang_limit_deg: float = DEFAULT_OVERHANG_LIMIT_DEG,
) -> PrintCheckReport:
    cama = _resolve_bed(printer, bed)

    try:
        triangulos = read_stl(Path(path))
    except StlUnreadable as exc:
        raise PrintCheckUnavailable(str(exc)) from exc

    if target_axis is not None and target_mm is not None:
        try:
            triangulos = scale_to_dimension(
                triangulos, axis=target_axis, millimetres=target_mm
            )
        except ValueError as exc:
            raise PrintCheckUnavailable(str(exc)) from exc

    malla = inspect_mesh(triangulos)
    voladizo = overhang_report(triangulos, limit_deg=overhang_limit_deg)

    bloqueos = list(malla.problems) if not malla.printable else []
    entra, motivo = fits_on_bed(malla.size, bed=cama)
    if not entra:
        bloqueos.append(motivo)

    consejos: list[str] = []
    if entra and "diagonal" in motivo:
        consejos.append(motivo)
    if malla.printable:
        consejos.extend(
            _rotation_advice(triangulos, voladizo.overhang_area_ratio, overhang_limit_deg)
        )
    if malla.degenerate_triangles and malla.printable:
        consejos.append(
            f"{malla.degenerate_triangles} triangulos no aportan superficie: "
            "el archivo pesa mas de lo que hace falta."
        )

    return PrintCheckReport(
        size=malla.size,
        mesh=malla,
        overhang_ratio=voladizo.overhang_area_ratio,
        blockers=bloqueos,
        advice=consejos,
    )
