from __future__ import annotations

import math

import pytest

from app.services.mesh_inspect import inspect_mesh
from app.services.polygon_mesh import PolygonError, plate_with_holes

# ---------------------------------------------------------------------------
# La placa de montaje es la pieza funcional mas pedida, y la unica de esta
# familia que necesita geometria de verdad: un rectangulo con huecos.
#
# El triangulador general (puentes + orejas) se intento y falla con mas de un
# agujero. Un triangulador sutilmente roto produce tapas con agujeros que PARECEN
# piezas bien hechas — la peor clase de fallo. Asi que la construccion es una
# tira de cuadrilateros con la misma cantidad de puntos de los dos lados, que no
# puede fallar, y las celdas vecinas comparten los MISMOS puntos del borde.
# ---------------------------------------------------------------------------


def area_esperada(w: float, d: float, agujeros: list[tuple[float, float, float]]) -> float:
    return w * d - sum(math.pi * (diam / 2) ** 2 for _, _, diam in agujeros)


class TestGeometry:
    @pytest.mark.parametrize(
        "agujeros",
        [
            [(30.0, 20.0, 12.0)],
            [(15.0, 20.0, 8.0), (45.0, 20.0, 8.0)],
            [(x, y, 6.4) for x in (12.0, 48.0) for y in (10.0, 30.0)],
            [(x, y, 5.0) for x in (10.0, 30.0, 50.0) for y in (10.0, 30.0)],
        ],
    )
    def test_every_pattern_comes_out_printable(self, agujeros) -> None:
        malla = plate_with_holes(
            width=60.0, depth=40.0, thickness=5.0, holes=agujeros, segments=48
        )

        reporte = inspect_mesh(malla)

        assert reporte.is_watertight, reporte.problems
        assert reporte.is_manifold, reporte.problems
        assert reporte.printable

    def test_the_holes_remove_the_material_they_should(self) -> None:
        # Si los agujeros fueran una marca y no geometria, el volumen seria el de
        # la placa entera. Esta es la comprobacion de que existen.
        agujeros = [(x, y, 6.4) for x in (12.0, 48.0) for y in (10.0, 30.0)]
        esperado = area_esperada(60.0, 40.0, agujeros) * 5.0

        medido = inspect_mesh(
            plate_with_holes(width=60.0, depth=40.0, thickness=5.0, holes=agujeros, segments=64)
        ).volume

        assert medido == pytest.approx(esperado, rel=2e-3)

    def test_the_outside_measures_what_was_asked(self) -> None:
        reporte = inspect_mesh(
            plate_with_holes(width=60.0, depth=40.0, thickness=5.0, holes=[(30.0, 20.0, 10.0)])
        )

        assert reporte.size == pytest.approx((60.0, 40.0, 5.0))

    def test_a_plate_without_holes_is_just_a_block(self) -> None:
        reporte = inspect_mesh(
            plate_with_holes(width=60.0, depth=40.0, thickness=5.0, holes=[])
        )

        assert reporte.volume == pytest.approx(12000.0)
        assert reporte.printable


class TestGuards:
    def test_a_hole_hanging_off_the_edge_is_refused(self) -> None:
        with pytest.raises(PolygonError, match="cabe|borde"):
            plate_with_holes(
                width=60.0, depth=40.0, thickness=5.0, holes=[(2.0, 20.0, 10.0)]
            )

    def test_two_holes_too_close_together_are_refused(self) -> None:
        # Se rechaza en vez de producir una placa con la junta rota.
        with pytest.raises(PolygonError):
            plate_with_holes(
                width=60.0,
                depth=40.0,
                thickness=5.0,
                holes=[(28.0, 20.0, 10.0), (32.0, 20.0, 10.0)],
            )

    def test_too_few_segments_is_refused(self) -> None:
        with pytest.raises(PolygonError, match="redondo"):
            plate_with_holes(
                width=60.0, depth=40.0, thickness=5.0, holes=[(30.0, 20.0, 10.0)], segments=4
            )

    def test_a_zero_dimension_is_refused(self) -> None:
        with pytest.raises(PolygonError):
            plate_with_holes(width=60.0, depth=0.0, thickness=5.0, holes=[])
