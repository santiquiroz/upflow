from __future__ import annotations

from pathlib import Path

import pytest

from app.services.missing_pack import PACK_LABELS
from app.services.openscad_render import (
    OpenScadError,
    assert_safe,
    extract_code,
    render_to_stl,
)

# ---------------------------------------------------------------------------
# La mitad DETERMINISTA del carril de cotas por descripcion: el modelo escribe el
# codigo, esto lo convierte en geometria. Si el codigo esta bien, la pieza mide
# exactamente lo que dice el codigo — de ahi vienen las cotas que ningun
# generador de malla puede dar.
#
# Y el codigo viene de un MODELO, no de una persona de confianza. OpenSCAD sabe
# leer archivos del disco; sin el guard, una descripcion de pieza se convierte en
# una lectura arbitraria.
# ---------------------------------------------------------------------------


class TestExtractCode:
    def test_it_unwraps_a_markdown_block(self) -> None:
        respuesta = "Aca va la pieza:\n```openscad\ncube([10, 10, 10]);\n```\nListo."

        assert extract_code(respuesta) == "cube([10, 10, 10]);"

    def test_it_unwraps_a_block_without_a_language_tag(self) -> None:
        assert extract_code("```\ncube([1,1,1]);\n```") == "cube([1,1,1]);"

    def test_bare_code_passes_through(self) -> None:
        assert extract_code("cube([10, 10, 10]);") == "cube([10, 10, 10]);"

    def test_the_explanation_around_the_block_is_dropped(self) -> None:
        # Los modelos explican aunque se les pida que no, y esa explicacion no
        # compila.
        respuesta = "Voy a usar difference().\n```scad\nsphere(5);\n```\nEspero que sirva."

        assert extract_code(respuesta) == "sphere(5);"


class TestSafety:
    @pytest.mark.parametrize(
        "codigo",
        [
            'include <BOSL2/std.scad>',
            'use <lib.scad>',
            'import("/etc/passwd");',
            'surface(file = "C:/Windows/win.ini");',
        ],
    )
    def test_code_that_reads_the_disk_is_refused(self, codigo: str) -> None:
        with pytest.raises(OpenScadError):
            assert_safe(codigo)

    def test_a_comment_mentioning_import_is_not_a_call(self) -> None:
        # Rechazar por la palabra sin mirar si es una llamada bloquearia codigo
        # perfectamente valido.
        assert_safe("// no hace falta import() aca\ncube([5,5,5]);")

    def test_normal_geometry_passes(self) -> None:
        assert_safe(
            "difference() {\n  cylinder(h=12, d=20, $fn=64);\n"
            "  translate([0,0,-1]) cylinder(h=14, d=8.4, $fn=64);\n}"
        )


class TestGuards:
    def test_empty_code_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(OpenScadError, match="compilar"):
            render_to_stl("", openscad=tmp_path / "x.exe", destination=tmp_path / "o.stl")

    def test_binario_faltante_dice_que_paquete_falta_sin_dictar_comandos(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(OpenScadError) as exc_info:
            render_to_stl(
                "cube([1,1,1]);",
                openscad=tmp_path / "no-esta.exe",
                destination=tmp_path / "o.stl",
            )

        mensaje = str(exc_info.value)
        # Que IDENTIFIQUE lo que falta, no la redaccion: si se afirma la frase
        # entera, cada retoque de estilo rompe la prueba sin que nada este mal.
        assert PACK_LABELS["openscad"] in mensaje
        # La regresion de verdad: el mensaje viejo mandaba a correr un script a
        # mano. La app baja el paquete sola; sin esta linea la prueba no cuida nada.
        assert ".ps1" not in mensaje


# ---------------------------------------------------------------------------
# Con el binario de verdad. Se saltan si no esta instalado: el resto de la suite
# tiene que correr en una maquina limpia.
# ---------------------------------------------------------------------------

OPENSCAD = Path(r"C:/Program Files/OpenSCAD/openscad.exe")
necesita_openscad = pytest.mark.skipif(
    not OPENSCAD.exists(), reason="OpenSCAD no esta instalado en esta maquina"
)


@necesita_openscad
class TestWithRealOpenScad:
    def test_a_spacer_measures_exactly_what_the_code_says(self, tmp_path: Path) -> None:
        import math

        from app.services.mesh_inspect import inspect_mesh
        from app.services.stl_reader import read_stl

        codigo = (
            "difference() {\n"
            "  cylinder(h = 12, d = 20, $fn = 64);\n"
            "  translate([0, 0, -1]) cylinder(h = 14, d = 8.4, $fn = 64);\n"
            "}"
        )

        resultado = render_to_stl(
            codigo, openscad=OPENSCAD, destination=tmp_path / "pieza.stl"
        )
        reporte = inspect_mesh(read_stl(resultado.stl_path))

        assert reporte.size == pytest.approx((20.0, 20.0, 12.0), abs=1e-3)
        assert reporte.printable
        # El volumen del anillo, no el del cilindro entero: la comprobacion de
        # que el agujero es geometria y no una marca.
        assert reporte.volume == pytest.approx(math.pi * (100 - 4.2**2) * 12, rel=3e-3)

    def test_code_that_does_not_compile_returns_the_compiler_message(
        self, tmp_path: Path
    ) -> None:
        # El error va TAL CUAL: es lo que le permite al modelo corregirse.
        with pytest.raises(OpenScadError) as exc_info:
            render_to_stl(
                "cube([10, 10, 10]", openscad=OPENSCAD, destination=tmp_path / "roto.stl"
            )

        assert str(exc_info.value)

    def test_the_source_is_kept_next_to_the_stl(self, tmp_path: Path) -> None:
        # El .scad es la pieza EDITABLE: guardarlo es la diferencia entre entregar
        # una pieza y entregar algo que se puede ajustar.
        resultado = render_to_stl(
            "cube([10, 10, 10]);", openscad=OPENSCAD, destination=tmp_path / "caja.stl"
        )

        assert resultado.stl_path.with_suffix(".scad").exists()


class TestElGuardNoSeSaltaConSaltosDeLinea:
    """El lexer de OpenSCAD no ve lineas, ve tokens.

    Medido el 2026-08-05: `include\n<archivo>` pasaba el guard Y OpenSCAD lo
    ejecutaba, porque el guard escaneaba linea por linea y la palabra clave
    nunca caia en la misma linea que el `<`. El codigo lo escribe un modelo que
    puede estar en un servidor ajeno, asi que el guard es lo unico que separa
    una descripcion de una pieza de una lectura de archivos arbitraria.
    """

    def test_include_partido_en_dos_lineas_se_bloquea(self) -> None:
        with pytest.raises(OpenScadError):
            assert_safe("include\n<C:/secreto.scad>\ncube([1,1,1]);")

    def test_use_partido_en_dos_lineas_se_bloquea(self) -> None:
        with pytest.raises(OpenScadError):
            assert_safe("use\n <lib.scad>\ncube([1,1,1]);")

    def test_import_partido_en_dos_lineas_se_bloquea(self) -> None:
        with pytest.raises(OpenScadError):
            assert_safe('import\n("pieza.stl")')

    def test_surface_partido_en_dos_lineas_se_bloquea(self) -> None:
        with pytest.raises(OpenScadError):
            assert_safe('surface\n(file="mapa.dat")')

    def test_un_comentario_de_bloque_en_el_medio_no_lo_esconde(self) -> None:
        with pytest.raises(OpenScadError):
            assert_safe('include /* nada que ver */ <lib.scad>')

    def test_una_pieza_normal_sigue_pasando(self) -> None:
        # El guard no sirve de nada si bloquea codigo legitimo: cada falso
        # positivo se come un intento del bucle.
        assert_safe(
            "$fn=64;\n"
            "// un espaciador\n"
            "difference() {\n"
            "  cylinder(d=20, h=12);\n"
            "  translate([0,0,-1]) cylinder(d=8.4, h=14);\n"
            "}\n"
        )

    def test_una_variable_que_empieza_con_use_no_es_un_use(self) -> None:
        assert_safe("use_width = 5;\ncube([use_width, 2, 2]);")
