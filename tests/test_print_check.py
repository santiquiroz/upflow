from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from app.services.print_check import PrintCheckUnavailable, check_stl_for_printing

# ---------------------------------------------------------------------------
# Esto compone todo lo demas en UN veredicto: leer, verificar estanqueidad,
# escalar a una medida real, ver si entra en la cama y cuanto voladizo queda.
#
# Sirve solo, sin ningun modelo generativo de por medio: cualquier STL —bajado
# de internet, exportado de Fusion, generado por IA— pasa por el mismo tamiz.
# Esa es la gracia: el veredicto no depende de quien hizo la malla.
# ---------------------------------------------------------------------------


def write_stl(path: Path, triangles: np.ndarray) -> Path:
    from app.services.stl_writer import write_stl as escribir

    return escribir(path, triangles)


def closed_box(x: float = 20.0, y: float = 20.0, z: float = 20.0) -> np.ndarray:
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


class TestVerdict:
    def test_a_closed_box_prints(self, tmp_path: Path) -> None:
        archivo = write_stl(tmp_path / "caja.stl", closed_box())

        reporte = check_stl_for_printing(archivo, printer="ender-3")

        assert reporte.can_print
        assert reporte.blockers == []

    def test_an_open_mesh_does_not_print_and_says_why(self, tmp_path: Path) -> None:
        abierta = closed_box()[:-2]
        archivo = write_stl(tmp_path / "abierta.stl", abierta)

        reporte = check_stl_for_printing(archivo, printer="ender-3")

        assert not reporte.can_print
        assert any("estanc" in b or "borde" in b for b in reporte.blockers)

    def test_a_part_bigger_than_the_bed_does_not_print(self, tmp_path: Path) -> None:
        archivo = write_stl(tmp_path / "enorme.stl", closed_box(500.0, 500.0, 500.0))

        reporte = check_stl_for_printing(archivo, printer="ender-3")

        assert not reporte.can_print
        assert any("cama" in b or "impresora" in b or "mm" in b for b in reporte.blockers)

    def test_the_same_huge_part_prints_on_a_bigger_printer(self, tmp_path: Path) -> None:
        # El veredicto es de la pieza CON la maquina, no de la pieza sola.
        archivo = write_stl(tmp_path / "grande.stl", closed_box(320.0, 320.0, 320.0))

        assert not check_stl_for_printing(archivo, printer="ender-3").can_print
        assert check_stl_for_printing(archivo, printer="ender-5-plus").can_print


class TestScaling:
    def test_it_can_resize_the_part_before_judging(self, tmp_path: Path) -> None:
        # Una malla generada sale en la escala que se le ocurrio al modelo: sin
        # este paso el veredicto es sobre un tamano que nadie pidio.
        archivo = write_stl(tmp_path / "chica.stl", closed_box(1.0, 1.0, 1.0))

        reporte = check_stl_for_printing(
            archivo, printer="ender-3", target_axis="x", target_mm=150.0
        )

        assert reporte.size[0] == pytest.approx(150.0, abs=0.01)

    def test_resizing_can_turn_a_fitting_part_into_one_that_does_not(self, tmp_path: Path) -> None:
        archivo = write_stl(tmp_path / "caja.stl", closed_box())

        assert check_stl_for_printing(archivo, printer="ender-3").can_print
        assert not check_stl_for_printing(
            archivo, printer="ender-3", target_axis="x", target_mm=400.0
        ).can_print

    def test_without_a_target_the_file_is_judged_as_it_comes(self, tmp_path: Path) -> None:
        archivo = write_stl(tmp_path / "caja.stl", closed_box(20.0, 20.0, 20.0))

        assert check_stl_for_printing(archivo, printer="ender-3").size == pytest.approx(
            (20.0, 20.0, 20.0), abs=0.01
        )


class TestAdvice:
    def test_it_suggests_rotating_when_that_removes_overhang(self, tmp_path: Path) -> None:
        # Un tetraedro chato: la cara grande queda mirando abajo y girarlo 180
        # la pone arriba. Cerrado a proposito — un consejo sobre una malla que ni
        # siquiera imprime seria ruido encima de un problema mas grave.
        a, b, c, d = (0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 100.0, 0.0), (0.0, 0.0, 10.0)
        chato = np.array([[a, c, b], [a, b, d], [a, d, c], [b, c, d]], dtype=np.float64)
        archivo = write_stl(tmp_path / "chato.stl", chato)

        reporte = check_stl_for_printing(archivo, printer="ender-3")

        assert any("girar" in c or "orientac" in c for c in reporte.advice)

    def test_a_part_in_its_best_orientation_gets_no_rotation_advice(self, tmp_path: Path) -> None:
        # Un consejo que no cambia nada entrena a ignorar los consejos.
        archivo = write_stl(tmp_path / "caja.stl", closed_box())

        reporte = check_stl_for_printing(archivo, printer="ender-3")

        assert not any("girar" in c for c in reporte.advice)

    def test_it_reports_the_overhang_share(self, tmp_path: Path) -> None:
        archivo = write_stl(tmp_path / "caja.stl", closed_box())

        assert 0.0 <= check_stl_for_printing(archivo, printer="ender-3").overhang_ratio <= 1.0


class TestGuards:
    def test_an_unknown_printer_lists_the_known_ones(self, tmp_path: Path) -> None:
        archivo = write_stl(tmp_path / "caja.stl", closed_box())

        with pytest.raises(PrintCheckUnavailable, match="ender-3"):
            check_stl_for_printing(archivo, printer="impresora-que-no-existe")

    def test_a_file_that_is_not_an_stl_fails_clearly(self, tmp_path: Path) -> None:
        roto = tmp_path / "no-es.stl"
        roto.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 120)

        with pytest.raises(PrintCheckUnavailable):
            check_stl_for_printing(roto, printer="ender-3")

    def test_a_custom_bed_can_be_passed_instead_of_a_name(self, tmp_path: Path) -> None:
        # Nadie va a mantener una tabla con todas las impresoras del mundo.
        archivo = write_stl(tmp_path / "caja.stl", closed_box(300.0, 300.0, 300.0))

        reporte = check_stl_for_printing(archivo, bed=(400.0, 400.0, 400.0))

        assert reporte.can_print
