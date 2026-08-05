from __future__ import annotations

import numpy as np
import pytest

from app.services.mesh_overhang import (
    DEFAULT_OVERHANG_LIMIT_DEG,
    best_print_orientation,
    overhang_report,
)

# ---------------------------------------------------------------------------
# En FDM cada capa se apoya en la anterior. Una cara que mira hacia abajo mas
# alla de cierto angulo no tiene sobre que apoyarse y necesita soporte: material
# extra, tiempo extra, y una superficie fea donde estaba pegado.
#
# El limite tipico son 45 grados desde la vertical. No es una constante del
# universo — depende de la maquina y del material — asi que entra como parametro.
#
# Esto se calcula EXACTO sobre las normales, sin muestreo ni aproximacion: es
# una comparacion por cara y cuesta lo mismo que recorrer la malla una vez.
# ---------------------------------------------------------------------------


def flat_plate_facing_up() -> np.ndarray:
    """Una placa horizontal: la cara de arriba no necesita nada."""
    return np.array(
        [
            [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)],
            [(10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)],
        ],
        dtype=np.float64,
    )


def flat_plate_facing_down() -> np.ndarray:
    """La misma placa con el bobinado al reves: mira hacia abajo, voladizo puro."""
    return flat_plate_facing_up()[:, ::-1, :]


def vertical_wall() -> np.ndarray:
    """Una pared vertical: se imprime sola, capa sobre capa."""
    return np.array(
        [
            [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 10.0)],
            [(10.0, 0.0, 0.0), (10.0, 0.0, 10.0), (0.0, 0.0, 10.0)],
        ],
        dtype=np.float64,
    )


class TestOverhangReport:
    def test_an_upward_face_needs_no_support(self) -> None:
        assert overhang_report(flat_plate_facing_up()).overhang_area_ratio == pytest.approx(0.0)

    def test_a_downward_face_is_all_overhang(self) -> None:
        assert overhang_report(flat_plate_facing_down()).overhang_area_ratio == pytest.approx(1.0)

    def test_a_vertical_wall_needs_no_support(self) -> None:
        # Justo en el limite: una pared vertical se imprime sola. Contarla como
        # voladizo llenaria de soporte cualquier caja.
        assert overhang_report(vertical_wall()).overhang_area_ratio == pytest.approx(0.0)

    def test_the_ratio_is_by_area_not_by_triangle_count(self) -> None:
        # Mil triangulitos mirando abajo pesan menos que una cara grande: contar
        # caras en vez de area miente sobre cuanto soporte hace falta.
        chico_abajo = np.array(
            [[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]], dtype=np.float64
        )
        grande_arriba = np.array(
            [[(0.0, 0.0, 5.0), (100.0, 0.0, 5.0), (0.0, 100.0, 5.0)]], dtype=np.float64
        )

        reporte = overhang_report(np.concatenate([chico_abajo, grande_arriba]))

        assert reporte.overhang_area_ratio < 0.01

    def test_a_stricter_limit_never_finds_less_overhang(self) -> None:
        inclinada = np.array(
            [[(0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (10.0, 0.0, -6.0)]], dtype=np.float64
        )

        estricto = overhang_report(inclinada, limit_deg=30.0).overhang_area_ratio
        permisivo = overhang_report(inclinada, limit_deg=70.0).overhang_area_ratio

        assert estricto >= permisivo

    def test_the_default_limit_is_the_usual_forty_five(self) -> None:
        assert DEFAULT_OVERHANG_LIMIT_DEG == 45.0

    def test_it_reports_the_area_in_square_millimetres(self) -> None:
        # Un porcentaje sin superficie no dice si son dos centimetros o media pieza.
        reporte = overhang_report(flat_plate_facing_down())

        assert reporte.overhang_area == pytest.approx(100.0)
        assert reporte.total_area == pytest.approx(100.0)

    def test_an_empty_mesh_does_not_divide_by_zero(self) -> None:
        assert overhang_report(np.empty((0, 3, 3))).overhang_area_ratio == 0.0


class TestBestPrintOrientation:
    def test_it_finds_the_rotation_with_the_least_overhang(self) -> None:
        # Una placa mirando abajo es 100% voladizo; darla vuelta la deja en 0%.
        eleccion = best_print_orientation(flat_plate_facing_down())

        assert eleccion.overhang_area_ratio == pytest.approx(0.0)

    def test_it_names_the_rotation_it_chose(self) -> None:
        # "Giralo" sin decir cuanto no le sirve a nadie.
        eleccion = best_print_orientation(flat_plate_facing_down())

        assert eleccion.rotation_deg != (0, 0, 0)

    def test_it_leaves_an_already_good_part_alone(self) -> None:
        # Si la orientacion original ya es la mejor, no se gira por girar.
        assert best_print_orientation(flat_plate_facing_up()).rotation_deg == (0, 0, 0)

    def test_the_returned_mesh_is_the_rotated_one(self) -> None:
        eleccion = best_print_orientation(flat_plate_facing_down())

        assert overhang_report(eleccion.mesh).overhang_area_ratio == pytest.approx(
            eleccion.overhang_area_ratio
        )

    def test_rotating_never_changes_the_size_of_the_part(self) -> None:
        # Una rotacion de 90 grados permuta las medidas; no puede inventar ni
        # perder material.
        original = np.array(
            [[(0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, 10.0, -5.0)]], dtype=np.float64
        )

        eleccion = best_print_orientation(original)

        def lados(m: np.ndarray) -> list[float]:
            puntos = m.reshape(-1, 3)
            return sorted(float(v) for v in (puntos.max(axis=0) - puntos.min(axis=0)))

        assert lados(eleccion.mesh) == pytest.approx(lados(original), abs=1e-9)
