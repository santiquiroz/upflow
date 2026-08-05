from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.mesh_inspect import inspect_mesh
from app.services.parametric_parts import box
from app.services.polygon_mesh import PolygonError
from app.services.rectilinear_plate import rectilinear_plate

# ---------------------------------------------------------------------------
# La escuadra es, junto con la placa y el espaciador, la otra pieza basica de una
# reparacion mecanica. Y no se puede armar pegando dos solidos: eso deja la
# arista de la junta con CUATRO caras, y el banco lo rechaza — con razon.
#
# Medido antes de escribir el modulo: dos cajas apoyadas una en otra dan
# estanca=True pero manifold=False. El test de abajo deja esa medicion clavada
# para que nadie "simplifique" volviendo a pegar cajas.
# ---------------------------------------------------------------------------

ELE = [(0.0, 0.0, 60.0, 15.0), (0.0, 15.0, 15.0, 40.0)]
TE = [(0.0, 0.0, 60.0, 15.0), (22.5, 15.0, 37.5, 50.0)]


def area(solido, agujeros) -> float:
    return sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in solido) - sum(
        math.pi * (d / 2) ** 2 for _, _, d in agujeros
    )


class TestGluingBoxesDoesNotWork:
    def test_two_touching_boxes_are_not_manifold(self) -> None:
        """La medicion que justifica todo este modulo."""
        horizontal = box(x=50.0, y=30.0, z=4.0)
        vertical = box(x=4.0, y=30.0, z=40.0) + np.array([0.0, 0.0, 4.0])

        reporte = inspect_mesh(np.concatenate([horizontal, vertical]))

        assert reporte.is_watertight
        assert not reporte.is_manifold
        assert not reporte.printable


class TestShapes:
    @pytest.mark.parametrize(
        "solido,agujeros",
        [
            (ELE, []),
            (ELE, [(45.0, 7.5, 6.4), (7.5, 30.0, 6.4)]),
            (ELE, [(30.0, 7.5, 5.0), (50.0, 7.5, 5.0), (7.5, 32.0, 5.0)]),
            (TE, []),
            (TE, [(8.0, 7.5, 5.0), (52.0, 7.5, 5.0)]),
        ],
    )
    def test_every_shape_comes_out_printable(self, solido, agujeros) -> None:
        malla = rectilinear_plate(solid=solido, thickness=5.0, holes=agujeros, segments=32)

        reporte = inspect_mesh(malla)

        assert reporte.is_watertight, reporte.problems
        assert reporte.is_manifold, reporte.problems
        assert reporte.printable

    def test_a_shape_without_holes_is_exact(self) -> None:
        # Sin agujeros no hay curvas que discretizar: el volumen tiene que dar
        # EXACTO. Este es el caso que destapo el sondeo de vecinos roto.
        reporte = inspect_mesh(rectilinear_plate(solid=ELE, thickness=5.0, holes=[]))

        assert reporte.volume == pytest.approx(area(ELE, []) * 5.0)

    def test_the_holes_remove_the_material_they_should(self) -> None:
        agujeros = [(45.0, 7.5, 6.4), (7.5, 30.0, 6.4)]

        medido = inspect_mesh(
            rectilinear_plate(solid=ELE, thickness=5.0, holes=agujeros, segments=64)
        ).volume

        assert medido == pytest.approx(area(ELE, agujeros) * 5.0, rel=2e-3)

    def test_the_outside_measures_what_was_asked(self) -> None:
        reporte = inspect_mesh(rectilinear_plate(solid=ELE, thickness=5.0, holes=[]))

        assert reporte.size == pytest.approx((60.0, 40.0, 5.0))


class TestGuards:
    def test_a_hole_outside_the_shape_is_refused(self) -> None:
        # En la L, la esquina superior derecha no tiene material.
        with pytest.raises(PolygonError, match="fuera de la pieza"):
            rectilinear_plate(solid=ELE, thickness=5.0, holes=[(50.0, 35.0, 5.0)])

    def test_an_empty_shape_is_refused(self) -> None:
        with pytest.raises(PolygonError):
            rectilinear_plate(solid=[], thickness=5.0, holes=[])

    def test_a_zero_thickness_is_refused(self) -> None:
        with pytest.raises(PolygonError):
            rectilinear_plate(solid=ELE, thickness=0.0, holes=[])
