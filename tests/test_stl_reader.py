from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from app.services.stl_reader import StlUnreadable, read_stl

# ---------------------------------------------------------------------------
# El STL es una SOPA DE TRIANGULOS: no tiene indice de vertices, cada triangulo
# repite sus tres puntos completos. Esa es la trampa central de todo lo que
# venga despues — contar aristas sobre los vertices crudos da que TODAS son
# borde, porque ningun triangulo comparte memoria con su vecino.
#
# Aca solo se lee. El analisis vive aparte.
# ---------------------------------------------------------------------------

# Un tetraedro: cuatro triangulos, cerrado, el solido mas chico que existe.
TETRA_VERTICES = [
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
]


def write_binary_stl(path: Path, triangles: list, header: bytes = b"upflow test") -> None:
    with path.open("wb") as handle:
        handle.write(header.ljust(80, b"\0")[:80])
        handle.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            handle.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for vertex in tri:
                handle.write(struct.pack("<3f", *vertex))
            handle.write(struct.pack("<H", 0))


def write_ascii_stl(path: Path, triangles: list) -> None:
    lineas = ["solid upflow"]
    for tri in triangles:
        lineas.append("  facet normal 0 0 0")
        lineas.append("    outer loop")
        for vertex in tri:
            lineas.append("      vertex {:.6e} {:.6e} {:.6e}".format(*vertex))
        lineas.append("    endloop")
        lineas.append("  endfacet")
    lineas.append("endsolid upflow")
    path.write_text("\n".join(lineas), encoding="ascii")


class TestBinaryStl:
    def test_reads_every_triangle(self, tmp_path: Path) -> None:
        destino = tmp_path / "tetra.stl"
        write_binary_stl(destino, TETRA_VERTICES)

        malla = read_stl(destino)

        assert malla.shape == (4, 3, 3)

    def test_the_coordinates_survive_the_round_trip(self, tmp_path: Path) -> None:
        destino = tmp_path / "tetra.stl"
        write_binary_stl(destino, TETRA_VERTICES)

        malla = read_stl(destino)

        assert malla[0][1] == pytest.approx([1.0, 0.0, 0.0])
        assert malla[3][2] == pytest.approx([0.0, 1.0, 0.0])

    def test_a_header_that_starts_with_solid_is_still_binary(self, tmp_path: Path) -> None:
        # La trampa clasica: muchos exportadores escriben "solid" en el header
        # de un STL BINARIO. Decidir el formato por esa palabra rompe el archivo.
        destino = tmp_path / "mentiroso.stl"
        write_binary_stl(destino, TETRA_VERTICES, header=b"solid cosa exportada")

        malla = read_stl(destino)

        assert malla.shape == (4, 3, 3)

    def test_a_truncated_file_says_so_instead_of_returning_garbage(self, tmp_path: Path) -> None:
        destino = tmp_path / "cortado.stl"
        write_binary_stl(destino, TETRA_VERTICES)
        datos = destino.read_bytes()
        destino.write_bytes(datos[: len(datos) - 30])

        with pytest.raises(StlUnreadable, match="incompleto"):
            read_stl(destino)

    def test_an_empty_mesh_is_refused(self, tmp_path: Path) -> None:
        destino = tmp_path / "vacio.stl"
        write_binary_stl(destino, [])

        with pytest.raises(StlUnreadable, match="sin triangulos"):
            read_stl(destino)


class TestAsciiStl:
    def test_reads_every_triangle(self, tmp_path: Path) -> None:
        destino = tmp_path / "tetra-ascii.stl"
        write_ascii_stl(destino, TETRA_VERTICES)

        malla = read_stl(destino)

        assert malla.shape == (4, 3, 3)

    def test_the_two_formats_produce_the_same_mesh(self, tmp_path: Path) -> None:
        binario, ascii_path = tmp_path / "b.stl", tmp_path / "a.stl"
        write_binary_stl(binario, TETRA_VERTICES)
        write_ascii_stl(ascii_path, TETRA_VERTICES)

        assert np.allclose(read_stl(binario), read_stl(ascii_path))

    def test_a_facet_missing_a_vertex_is_refused(self, tmp_path: Path) -> None:
        destino = tmp_path / "roto.stl"
        destino.write_text(
            "solid x\n facet normal 0 0 0\n  outer loop\n"
            "   vertex 0 0 0\n   vertex 1 0 0\n"
            "  endloop\n endfacet\nendsolid x\n",
            encoding="ascii",
        )

        with pytest.raises(StlUnreadable):
            read_stl(destino)


class TestGuards:
    def test_a_missing_file_says_which_one(self, tmp_path: Path) -> None:
        with pytest.raises(StlUnreadable, match="no existe"):
            read_stl(tmp_path / "no-esta.stl")

    def test_something_that_is_not_an_stl_is_refused(self, tmp_path: Path) -> None:
        destino = tmp_path / "foto.stl"
        destino.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 200)

        with pytest.raises(StlUnreadable):
            read_stl(destino)
