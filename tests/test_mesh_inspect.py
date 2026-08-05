from __future__ import annotations

import numpy as np
import pytest

from app.services.mesh_inspect import inspect_mesh, weld_tolerance_for, weld_vertices

# ---------------------------------------------------------------------------
# "Watertight y manifold" es el UNICO predictor de impresion en el que coinciden
# todos los bandos: los que venden generadores y los que los critican. Por eso
# esto se mide aca adentro y no se le cree a nadie.
#
# Las definiciones, sin misticismo:
#   - Cada arista de una superficie cerrada la comparten EXACTAMENTE 2 triangulos.
#   - Una arista con 1 triangulo es un agujero (no es estanco: el laminador no
#     sabe que esta adentro y que afuera).
#   - Una arista con 3 o mas es no-manifold (la geometria se bifurca; el
#     laminador tiene que adivinar).
#
# LA TRAMPA: el STL es sopa de triangulos y repite cada vertice. Sin soldar
# primero, TODAS las aristas parecen borde y todo modelo del mundo da "roto".
# ---------------------------------------------------------------------------


def tetrahedron() -> np.ndarray:
    """El solido cerrado mas chico: cuatro caras, seis aristas, todas compartidas."""
    a, b, c, d = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    return np.array([[a, c, b], [a, b, d], [a, d, c], [b, c, d]], dtype=np.float64)


def open_box() -> np.ndarray:
    """Dos triangulos sueltos: cuatro aristas de borde, nada cerrado."""
    return np.array(
        [
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        ],
        dtype=np.float64,
    )


class TestWeldVertices:
    def test_the_soup_collapses_to_the_real_corners(self) -> None:
        # Un tetraedro son 12 vertices escritos y 4 puntos de verdad.
        vertices, caras = weld_vertices(tetrahedron())

        assert len(vertices) == 4
        assert caras.shape == (4, 3)

    def test_points_a_hair_apart_are_the_same_corner(self) -> None:
        # Los generadores escriben float32 y el mismo punto vuelve con basura en
        # el ultimo bit. Sin tolerancia, cada esquina se parte en varias y la
        # malla entera se declara rota por un error de redondeo.
        malla = tetrahedron()
        malla[1][0][0] += 1e-9

        vertices, _caras = weld_vertices(malla)

        assert len(vertices) == 4

    def test_points_genuinely_apart_stay_apart(self) -> None:
        # Soldar de mas es peor que no soldar: fusiona detalle real.
        malla = tetrahedron()
        malla[1][0][0] += 0.5

        vertices, _caras = weld_vertices(malla)

        assert len(vertices) == 5


class TestClosedSolid:
    def test_a_tetrahedron_is_watertight_and_manifold(self) -> None:
        reporte = inspect_mesh(tetrahedron())

        assert reporte.is_watertight
        assert reporte.is_manifold
        assert reporte.boundary_edges == 0
        assert reporte.non_manifold_edges == 0

    def test_it_reports_the_real_triangle_and_vertex_counts(self) -> None:
        reporte = inspect_mesh(tetrahedron())

        assert reporte.triangle_count == 4
        assert reporte.vertex_count == 4

    def test_the_volume_is_the_geometric_one(self) -> None:
        # Un tetraedro sobre los ejes con catetos 1 mide 1/6.
        reporte = inspect_mesh(tetrahedron())

        assert reporte.volume == pytest.approx(1 / 6, rel=1e-6)

    def test_the_volume_is_positive_even_with_inverted_winding(self) -> None:
        # Un modelo con las normales al reves sigue teniendo el mismo tamano;
        # un volumen negativo seria un detalle de orientacion, no una medida.
        invertido = tetrahedron()[:, ::-1, :]

        assert inspect_mesh(invertido).volume == pytest.approx(1 / 6, rel=1e-6)

    def test_the_bounding_box_is_what_gets_printed(self) -> None:
        reporte = inspect_mesh(tetrahedron())

        assert reporte.size == pytest.approx((1.0, 1.0, 1.0))


class TestOpenMesh:
    def test_two_loose_triangles_are_not_watertight(self) -> None:
        reporte = inspect_mesh(open_box())

        assert not reporte.is_watertight
        assert reporte.boundary_edges == 4

    def test_an_open_mesh_is_still_manifold(self) -> None:
        # Abierto y no-manifold son problemas DISTINTOS: una lamina tiene
        # agujero pero no se bifurca. Confundirlos manda a reparar lo que no es.
        reporte = inspect_mesh(open_box())

        assert reporte.is_manifold

    def test_an_open_mesh_reports_no_volume(self) -> None:
        # El volumen de algo abierto no significa nada: no hay adentro.
        assert inspect_mesh(open_box()).volume is None


class TestNonManifold:
    def test_three_triangles_on_one_edge_are_reported(self) -> None:
        # Tres laminas pegadas a la misma arista: el laminador no puede decidir
        # que queda adentro.
        a, b = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
        malla = np.array(
            [
                [a, b, (0.0, 1.0, 0.0)],
                [a, b, (0.0, 0.0, 1.0)],
                [a, b, (0.0, -1.0, 0.0)],
            ],
            dtype=np.float64,
        )

        reporte = inspect_mesh(malla)

        assert not reporte.is_manifold
        assert reporte.non_manifold_edges == 1


class TestDegenerateTriangles:
    def test_zero_area_triangles_are_counted(self) -> None:
        # Los generadores de malla los producen a montones y ensucian el
        # analisis de aristas sin aportar superficie.
        malla = np.concatenate(
            [
                tetrahedron(),
                np.array([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]),
            ]
        )

        reporte = inspect_mesh(malla)

        assert reporte.degenerate_triangles == 1

    def test_a_degenerate_triangle_does_not_break_the_closed_verdict(self) -> None:
        # Un triangulo de area cero no abre un solido cerrado: descartarlo antes
        # de contar aristas es lo que evita el falso "roto".
        malla = np.concatenate(
            [
                tetrahedron(),
                np.array([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]]),
            ]
        )

        assert inspect_mesh(malla).is_watertight


class TestPrintability:
    def test_a_closed_solid_is_printable(self) -> None:
        assert inspect_mesh(tetrahedron()).printable

    def test_an_open_mesh_is_not_printable(self) -> None:
        assert not inspect_mesh(open_box()).printable

    def test_the_report_says_why_it_is_not_printable(self) -> None:
        # "No sirve" sin motivo no le dice a nadie que arreglar.
        problemas = inspect_mesh(open_box()).problems

        assert any("borde" in p or "estanc" in p for p in problemas)

    def test_a_printable_mesh_lists_no_problems(self) -> None:
        assert inspect_mesh(tetrahedron()).problems == []


class TestWeldToleranceScales:
    """La tolerancia de soldado NO puede ser un numero fijo.

    Medido (`scripts/spike_mesh_inspect.py`): con 0,001 mm fijos, una esfera de
    1.046.528 triangulos daba NO ESTANCA porque junto a los polos los vertices
    caen a 0,000376 mm y se fusionaban 3.328 caras que estaban sanas. Declarar
    rota una malla buena es el peor error que puede cometer este verificador.

    Lo que el soldado tiene que deshacer es la duplicacion del STL: el MISMO
    punto escrito dos veces con redondeo de float32. Asi que la tolerancia sale
    de la precision de float32 a la escala del modelo, y de nada mas.
    """

    def test_a_small_model_gets_a_fine_tolerance(self) -> None:
        assert weld_tolerance_for(np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]])) < 1e-4

    def test_a_car_sized_part_gets_a_coarser_one(self) -> None:
        chico = weld_tolerance_for(np.zeros((1, 3, 3)) + 10.0)
        grande = weld_tolerance_for(np.zeros((1, 3, 3)) + 800.0)

        assert grande > chico

    def test_it_never_goes_below_the_float32_floor(self) -> None:
        # Una tolerancia de cero no suelda nada y toda malla da rota.
        assert weld_tolerance_for(np.zeros((1, 3, 3))) > 0.0

    def test_it_stays_far_under_any_printable_feature(self) -> None:
        # Una pieza de 800 mm: la tolerancia tiene que quedar miles de veces por
        # debajo de lo que una impresora FDM resuelve (~0,1 mm), o fusionaria
        # detalle real.
        assert weld_tolerance_for(np.zeros((1, 3, 3)) + 800.0) < 0.01


class TestWeldingDoesNotInventProblems:
    def test_a_face_collapsed_by_welding_is_not_counted_as_geometry(self) -> None:
        # Si el soldado junta dos esquinas de un mismo triangulo, esa cara ya no
        # tiene area: contar sus aristas ensucia el veredicto con problemas que
        # el modelo no tiene.
        a = (0.0, 0.0, 0.0)
        casi_a = (1e-12, 0.0, 0.0)
        malla = np.concatenate(
            [tetrahedron(), np.array([[a, casi_a, (0.0, 5.0, 0.0)]])]
        )

        reporte = inspect_mesh(malla)

        assert reporte.is_watertight
        assert reporte.non_manifold_edges == 0

    def test_a_dense_sphere_is_still_reported_watertight(self) -> None:
        # El caso exacto que fallaba: malla fina, sana, y el verificador decia
        # que no. Version reducida para que el test sea rapido.
        malla = _uv_sphere(segments=256, rings=128)

        reporte = inspect_mesh(malla)

        assert reporte.is_watertight, reporte.problems
        assert reporte.is_manifold, reporte.problems


def _uv_sphere(*, segments: int, rings: int, radius: float = 10.0) -> np.ndarray:
    import math

    thetas = np.linspace(0.0, math.pi, rings + 1)
    phis = np.linspace(0.0, 2 * math.pi, segments + 1)[:-1]

    def punto(i_ring: int, i_seg: int):
        t, p = thetas[i_ring], phis[i_seg % segments]
        return (
            radius * math.sin(t) * math.cos(p),
            radius * math.sin(t) * math.sin(p),
            radius * math.cos(t),
        )

    triangulos = []
    for r in range(rings):
        for s in range(segments):
            a, b = punto(r, s), punto(r, s + 1)
            c, d = punto(r + 1, s + 1), punto(r + 1, s)
            if r == 0:
                triangulos.append([a, c, d])
            elif r == rings - 1:
                triangulos.append([a, b, d])
            else:
                triangulos.append([a, b, d])
                triangulos.append([b, c, d])
    return np.asarray(triangulos, dtype=np.float64)
