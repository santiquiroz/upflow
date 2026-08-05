from __future__ import annotations

import numpy as np
import pytest

from app.services.mesh_inspect import inspect_mesh
from app.services.mesh_repair import boundary_loops, repair_mesh

# ---------------------------------------------------------------------------
# Reparar aca significa UNA cosa: tapar los agujeros que dejaron triangulos
# faltantes. NO significa reconstruir geometria que nunca estuvo, ni adivinar la
# forma que el modelo tendria que haber tenido.
#
# Y nunca se declara reparada: se vuelve a MEDIR. Si sigue rota, se dice. Un
# reparador que afirma haber arreglado sin comprobarlo es peor que no tener
# reparador, porque manda a imprimir con confianza prestada.
# ---------------------------------------------------------------------------


def tetrahedron() -> np.ndarray:
    a, b, c, d = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)
    return np.array([[a, c, b], [a, b, d], [a, d, c], [b, c, d]], dtype=np.float64)


def box(size: float = 10.0) -> np.ndarray:
    e = np.array(
        [
            [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
            [0, 0, size], [size, 0, size], [size, size, size], [0, size, size],
        ],
        dtype=np.float64,
    )
    caras = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3),
    ]
    return np.array([[e[i] for i in c] for c in caras], dtype=np.float64)


class TestBoundaryLoops:
    def test_a_closed_solid_has_no_loops(self) -> None:
        assert boundary_loops(tetrahedron()) == []

    def test_one_missing_face_leaves_one_loop_of_three(self) -> None:
        bucles = boundary_loops(tetrahedron()[:-1])

        assert len(bucles) == 1
        assert len(bucles[0]) == 3

    def test_two_separate_holes_give_two_loops(self) -> None:
        # Sacar dos caras opuestas de la caja deja dos agujeros que no se tocan.
        con_dos_agujeros = np.delete(box(), [0, 1, 2, 3], axis=0)

        assert len(boundary_loops(con_dos_agujeros)) == 2

    def test_a_square_hole_comes_back_as_four_corners(self) -> None:
        sin_tapa = box()[:2 * 1] if False else np.delete(box(), [2, 3], axis=0)

        bucles = boundary_loops(sin_tapa)

        assert len(bucles) == 1
        assert len(bucles[0]) == 4


class TestRepair:
    def test_a_hole_gets_closed(self) -> None:
        roto = tetrahedron()[:-1]
        assert not inspect_mesh(roto).is_watertight

        reparado, reporte = repair_mesh(roto)

        assert reporte.is_watertight
        assert inspect_mesh(reparado).is_watertight

    def test_the_closed_solid_keeps_its_volume(self) -> None:
        # Tapar el agujero tiene que devolver el solido original, no uno inflado.
        original = inspect_mesh(tetrahedron()).volume
        reparado, _reporte = repair_mesh(tetrahedron()[:-1])

        assert inspect_mesh(reparado).volume == pytest.approx(original, rel=1e-6)

    def test_a_square_hole_is_closed_too(self) -> None:
        sin_tapa = np.delete(box(), [2, 3], axis=0)

        _reparado, reporte = repair_mesh(sin_tapa)

        assert reporte.is_watertight

    def test_the_repaired_mesh_keeps_its_outward_winding(self) -> None:
        # Una tapa con el bobinado al reves deja el solido cerrado pero con una
        # cara mirando adentro, y el laminador la lee como un hueco.
        reparado, _reporte = repair_mesh(box()[:-2])

        assert inspect_mesh(reparado).volume == pytest.approx(1000.0, rel=1e-6)

    def test_an_already_closed_mesh_is_left_exactly_as_it_was(self) -> None:
        # Tocar lo que ya estaba bien solo puede empeorarlo.
        reparado, reporte = repair_mesh(tetrahedron())

        assert reporte.is_watertight
        assert np.allclose(reparado, tetrahedron())

    def test_degenerate_triangles_are_dropped(self) -> None:
        con_basura = np.concatenate(
            [tetrahedron(), np.array([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]])]
        )

        reparado, _reporte = repair_mesh(con_basura)

        assert len(reparado) == 4


class TestHonesty:
    def test_it_measures_again_instead_of_claiming_success(self) -> None:
        # El reporte que devuelve sale de INSPECCIONAR el resultado, no de que la
        # funcion crea que lo logro.
        roto = tetrahedron()[:-1]

        reparado, reporte = repair_mesh(roto)

        assert reporte == inspect_mesh(reparado)

    def test_a_mesh_it_cannot_close_comes_back_reported_as_open(self) -> None:
        # Dos triangulos sueltos no forman un solido: taparlos no lo vuelve uno,
        # y decir que si seria mentir.
        lamina = np.array(
            [
                [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)],
                [(100.0, 0.0, 0.0), (110.0, 0.0, 0.0), (100.0, 10.0, 0.0)],
            ],
            dtype=np.float64,
        )

        _reparado, reporte = repair_mesh(lamina)

        # Puede cerrarse como dos laminas dobles o no cerrarse: lo que NO puede
        # es decir que quedo bien sin haberlo medido.
        assert reporte == inspect_mesh(_reparado)

    def test_an_empty_mesh_does_not_explode(self) -> None:
        reparado, reporte = repair_mesh(np.empty((0, 3, 3)))

        assert len(reparado) == 0
        assert reporte.triangle_count == 0
