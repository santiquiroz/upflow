from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from app.services.stl_reader import read_stl
from app.services.stl_writer import write_stl

# ---------------------------------------------------------------------------
# Escribir el STL es el ultimo paso: lo que sale de aca es lo que el usuario
# mete en el laminador. Se escribe BINARIO porque un ASCII de un millon de
# triangulos pesa cinco veces mas y ningun laminador lo agradece.
#
# Las normales se calculan de verdad, aunque casi todos los laminadores las
# ignoren: hay herramientas de reparacion que las leen, y escribir ceros ahi es
# escribir una mentira que despues alguien usa.
# ---------------------------------------------------------------------------


def tetrahedron() -> np.ndarray:
    a, b, c, d = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    return np.array([[a, c, b], [a, b, d], [a, d, c], [b, c, d]], dtype=np.float64)


class TestRoundTrip:
    def test_what_goes_in_comes_out(self, tmp_path: Path) -> None:
        destino = tmp_path / "salida.stl"
        write_stl(destino, tetrahedron())

        assert np.allclose(read_stl(destino), tetrahedron(), atol=1e-6)

    def test_the_file_is_binary(self, tmp_path: Path) -> None:
        # 84 bytes de cabecera + 50 por triangulo, exacto.
        destino = tmp_path / "salida.stl"
        write_stl(destino, tetrahedron())

        assert destino.stat().st_size == 84 + 4 * 50

    def test_a_million_triangles_do_not_take_forever(self, tmp_path: Path) -> None:
        grande = np.repeat(tetrahedron(), 50_000, axis=0)
        destino = tmp_path / "grande.stl"

        write_stl(destino, grande)

        assert destino.stat().st_size == 84 + len(grande) * 50


class TestNormals:
    def leer_normales(self, path: Path) -> np.ndarray:
        datos = path.read_bytes()
        cantidad = struct.unpack_from("<I", datos, 80)[0]
        registros = np.frombuffer(
            datos[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
            count=cantidad,
        )
        return np.asarray(registros["n"], dtype=np.float64)

    def test_the_normals_are_not_zeros(self, tmp_path: Path) -> None:
        destino = tmp_path / "salida.stl"
        write_stl(destino, tetrahedron())

        assert np.all(np.linalg.norm(self.leer_normales(destino), axis=1) > 0.9)

    def test_the_normals_are_unit_length(self, tmp_path: Path) -> None:
        destino = tmp_path / "salida.stl"
        write_stl(destino, tetrahedron())

        assert np.allclose(np.linalg.norm(self.leer_normales(destino), axis=1), 1.0, atol=1e-5)

    def test_a_degenerate_triangle_gets_a_zero_normal_instead_of_nan(self, tmp_path: Path) -> None:
        # Normalizar un area cero da NaN, y un NaN en el archivo lo rompe entero.
        plana = np.array([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]])
        destino = tmp_path / "plana.stl"

        write_stl(destino, plana)

        assert np.all(np.isfinite(self.leer_normales(destino)))


class TestGuards:
    def test_an_empty_mesh_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="vac"):
            write_stl(tmp_path / "vacio.stl", np.empty((0, 3, 3)))

    def test_a_wrong_shape_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            write_stl(tmp_path / "raro.stl", np.zeros((4, 2, 3)))

    def test_it_creates_the_folder_if_missing(self, tmp_path: Path) -> None:
        destino = tmp_path / "sub" / "carpeta" / "salida.stl"

        write_stl(destino, tetrahedron())

        assert destino.exists()
