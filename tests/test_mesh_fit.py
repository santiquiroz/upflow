from __future__ import annotations

import numpy as np
import pytest

from app.services.mesh_fit import (
    PRINTER_BEDS,
    fits_on_bed,
    scale_to_dimension,
    smallest_bed_orientation,
)

# ---------------------------------------------------------------------------
# El STL no tiene unidades. Ni una. Todo laminador asume milimetros, y un modelo
# generado por IA no sabe que tiene que medir 80 mm: sale en la escala que se le
# ocurrio al modelo. Ese es el paso que convierte una malla en una PIEZA.
#
# Y despues hay una pregunta que ninguna IA contesta: ¿entra en la cama?
# Una Ender 3 son 220x220x250 mm. Una pieza de carro de 300 mm no entra, y
# enterarse en el laminador es tarde.
# ---------------------------------------------------------------------------


def box(x: float, y: float, z: float) -> np.ndarray:
    """Una caja de x por y por z, cerrada, para medir escalas sin ruido."""
    esquinas = np.array(
        [
            [0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0],
            [0, 0, z], [x, 0, z], [x, y, z], [0, y, z],
        ],
        dtype=np.float64,
    )
    caras = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3),
    ]
    return np.array([[esquinas[i] for i in cara] for cara in caras], dtype=np.float64)


class TestScaleToDimension:
    def test_the_chosen_axis_lands_on_the_requested_size(self) -> None:
        escalada = scale_to_dimension(box(2.0, 1.0, 1.0), axis="x", millimetres=80.0)

        assert escalada[:, :, 0].max() - escalada[:, :, 0].min() == pytest.approx(80.0)

    def test_the_other_axes_keep_their_proportion(self) -> None:
        # Escalar sin uniformidad deforma la pieza: un agujero redondo sale oval.
        escalada = scale_to_dimension(box(2.0, 1.0, 4.0), axis="x", millimetres=80.0)

        ancho = escalada[:, :, 1].max() - escalada[:, :, 1].min()
        alto = escalada[:, :, 2].max() - escalada[:, :, 2].min()
        assert ancho == pytest.approx(40.0)
        assert alto == pytest.approx(160.0)

    def test_scaling_by_height_works_too(self) -> None:
        escalada = scale_to_dimension(box(1.0, 1.0, 2.0), axis="z", millimetres=50.0)

        assert escalada[:, :, 2].max() - escalada[:, :, 2].min() == pytest.approx(50.0)

    def test_a_flat_axis_cannot_be_scaled(self) -> None:
        # Dividir por cero daria infinitos silenciosos en cada vertice.
        plana = box(10.0, 10.0, 0.0)

        with pytest.raises(ValueError, match="plano|cero"):
            scale_to_dimension(plana, axis="z", millimetres=50.0)

    def test_a_non_positive_target_is_refused(self) -> None:
        with pytest.raises(ValueError):
            scale_to_dimension(box(1.0, 1.0, 1.0), axis="x", millimetres=0.0)

    def test_an_unknown_axis_is_refused(self) -> None:
        with pytest.raises(ValueError, match="(?i)eje"):
            scale_to_dimension(box(1.0, 1.0, 1.0), axis="w", millimetres=10.0)


class TestFitsOnBed:
    def test_a_part_that_fits_fits(self) -> None:
        cabe, _motivo = fits_on_bed((100.0, 100.0, 100.0), bed=(220.0, 220.0, 250.0))

        assert cabe

    def test_a_part_taller_than_the_printer_does_not(self) -> None:
        cabe, motivo = fits_on_bed((100.0, 100.0, 300.0), bed=(220.0, 220.0, 250.0))

        assert not cabe
        assert "alto" in motivo or "250" in motivo

    def test_a_part_that_only_fits_rotated_is_reported_as_fitting(self) -> None:
        # 300 de largo no entra en 220, pero girado en la diagonal si: no decirlo
        # manda a partir una pieza que no hacia falta partir.
        cabe, motivo = fits_on_bed((300.0, 20.0, 20.0), bed=(220.0, 220.0, 250.0))

        assert cabe
        assert "girada" in motivo or "diagonal" in motivo

    def test_a_part_that_does_not_fit_even_rotated(self) -> None:
        cabe, _motivo = fits_on_bed((400.0, 400.0, 20.0), bed=(220.0, 220.0, 250.0))

        assert not cabe

    def test_the_reason_names_the_measurement_that_failed(self) -> None:
        _cabe, motivo = fits_on_bed((100.0, 100.0, 999.0), bed=(220.0, 220.0, 250.0))

        assert "999" in motivo


class TestPrinterBeds:
    def test_the_common_creality_beds_are_known(self) -> None:
        assert "ender-3" in PRINTER_BEDS
        assert PRINTER_BEDS["ender-3"] == (220.0, 220.0, 250.0)

    def test_every_bed_has_three_positive_dimensions(self) -> None:
        for nombre, cama in PRINTER_BEDS.items():
            assert len(cama) == 3, nombre
            assert all(v > 0 for v in cama), nombre


class TestSmallestBedOrientation:
    def test_it_lays_the_longest_side_flat(self) -> None:
        # Acostar la pieza baja el alto, que es lo que mas suele sobrar, y de
        # paso reduce el tiempo y el soporte.
        rotada = smallest_bed_orientation((20.0, 30.0, 200.0))

        assert rotada[2] == pytest.approx(20.0)
        assert sorted(rotada) == pytest.approx([20.0, 30.0, 200.0])

    def test_an_already_flat_part_is_left_alone(self) -> None:
        assert smallest_bed_orientation((200.0, 30.0, 20.0)) == pytest.approx((200.0, 30.0, 20.0))
