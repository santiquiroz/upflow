from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.mesh_inspect import inspect_mesh
from app.services.parametric_parts import PartError, box, cylinder, tube

# ---------------------------------------------------------------------------
# Esto es la respuesta al hallazgo central de la investigacion: un generador de
# malla NO sabe que la pieza tiene que medir 80 mm, y una pieza que tiene que
# encajar necesita la cota exacta.
#
# Aca la geometria se construye ANALITICAMENTE. La cota no se aproxima: sale de
# la formula. Y sale estanca por construccion, lo cual no se asume — se verifica
# con el mismo banco que verifica cualquier STL.
#
# Un espaciador, un buje, una arandela y un separador son la misma pieza con
# otros numeros, y son el pan de cada dia de una reparacion mecanica.
# ---------------------------------------------------------------------------


class TestBox:
    def test_it_measures_exactly_what_was_asked(self) -> None:
        reporte = inspect_mesh(box(x=40.0, y=25.0, z=10.0))

        assert reporte.size == pytest.approx((40.0, 25.0, 10.0))

    def test_it_is_a_printable_solid(self) -> None:
        assert inspect_mesh(box(x=40.0, y=25.0, z=10.0)).printable

    def test_the_volume_is_the_formula(self) -> None:
        assert inspect_mesh(box(x=40.0, y=25.0, z=10.0)).volume == pytest.approx(10000.0)

    def test_a_zero_dimension_is_refused(self) -> None:
        with pytest.raises(PartError):
            box(x=40.0, y=0.0, z=10.0)


class TestCylinder:
    def test_the_height_is_exact(self) -> None:
        reporte = inspect_mesh(cylinder(diameter=20.0, height=15.0))

        assert reporte.size[2] == pytest.approx(15.0)

    def test_the_diameter_is_exact_across_the_flats(self) -> None:
        # Un cilindro triangulado es un prisma: el diametro exacto lo tienen los
        # vertices, y la caja envolvente los toca.
        reporte = inspect_mesh(cylinder(diameter=20.0, height=15.0, segments=128))

        assert reporte.size[0] == pytest.approx(20.0, rel=1e-9)

    def test_it_is_a_printable_solid(self) -> None:
        assert inspect_mesh(cylinder(diameter=20.0, height=15.0)).printable

    def test_the_volume_converges_to_pi_r_squared_h(self) -> None:
        exacto = math.pi * 10.0**2 * 15.0
        grueso = inspect_mesh(cylinder(diameter=20.0, height=15.0, segments=16)).volume
        fino = inspect_mesh(cylinder(diameter=20.0, height=15.0, segments=256)).volume

        assert abs(fino - exacto) < abs(grueso - exacto)
        assert fino == pytest.approx(exacto, rel=1e-3)

    def test_too_few_segments_is_refused(self) -> None:
        # Con menos de tres no hay circulo, hay una lamina.
        with pytest.raises(PartError):
            cylinder(diameter=20.0, height=15.0, segments=2)


class TestTube:
    """El espaciador, el buje, la arandela y el separador son esta pieza."""

    def test_it_is_a_printable_solid(self) -> None:
        assert inspect_mesh(tube(outer_diameter=20.0, inner_diameter=8.0, height=12.0)).printable

    def test_the_outside_measures_what_was_asked(self) -> None:
        reporte = inspect_mesh(
            tube(outer_diameter=20.0, inner_diameter=8.0, height=12.0, segments=128)
        )

        assert reporte.size[0] == pytest.approx(20.0, rel=1e-9)
        assert reporte.size[2] == pytest.approx(12.0)

    def test_the_volume_is_the_ring_not_the_full_cylinder(self) -> None:
        # Si el agujero no estuviera de verdad, el volumen seria el del cilindro
        # entero. Esta es la comprobacion de que el agujero existe.
        exacto = math.pi * (10.0**2 - 4.0**2) * 12.0
        medido = inspect_mesh(
            tube(outer_diameter=20.0, inner_diameter=8.0, height=12.0, segments=256)
        ).volume

        assert medido == pytest.approx(exacto, rel=1e-3)

    def test_a_hole_bigger_than_the_part_is_refused(self) -> None:
        with pytest.raises(PartError, match="agujero|interior"):
            tube(outer_diameter=8.0, inner_diameter=20.0, height=12.0)

    def test_a_hole_equal_to_the_outside_is_refused(self) -> None:
        # Pared de espesor cero: no queda pieza.
        with pytest.raises(PartError):
            tube(outer_diameter=20.0, inner_diameter=20.0, height=12.0)

    def test_a_washer_is_the_same_part_with_less_height(self) -> None:
        arandela = tube(outer_diameter=20.0, inner_diameter=10.4, height=1.6)

        assert inspect_mesh(arandela).printable

    def test_the_wall_thickness_is_reported_when_it_is_too_thin(self) -> None:
        # Una pared mas fina que la boquilla no se imprime: el laminador la
        # descarta en silencio y la pieza sale con un agujero donde iba pared.
        with pytest.raises(PartError, match="pared"):
            tube(outer_diameter=20.0, inner_diameter=19.9, height=12.0, min_wall_mm=0.4)


class TestExactnessAgainstTheBench:
    """La comprobacion que cierra el circulo: lo generado pasa el mismo banco que
    verifica cualquier STL bajado de internet."""

    @pytest.mark.parametrize(
        "pieza",
        [
            box(x=30.0, y=30.0, z=5.0),
            cylinder(diameter=12.0, height=40.0),
            tube(outer_diameter=25.0, inner_diameter=16.0, height=8.0),
        ],
    )
    def test_every_generated_part_passes_the_print_check(self, pieza: np.ndarray) -> None:
        reporte = inspect_mesh(pieza)

        assert reporte.printable, reporte.problems
        assert reporte.degenerate_triangles == 0
