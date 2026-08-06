from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.stl_writer import write_stl

from app.services.openscad_llm import (
    LlmUnavailable,
    describe_to_stl,
)
from app.services.openscad_render import OpenScadError, RenderResult

# ---------------------------------------------------------------------------
# EL BUCLE ES LA PIEZA CENTRAL. Un modelo que falla una vez suele acertar al
# segundo intento con el error del compilador a la vista; sin el bucle, ese
# intento se pierde y el usuario ve un fallo que era recuperable.
#
# Medido (2026-08-05, `qwen3-coder:30b` local): de tres piezas pedidas en
# castellano, dos salieron con las cotas exactas al primer intento y una no
# compilo. Ese tercer caso es exactamente el que este modulo resuelve.
# ---------------------------------------------------------------------------


class ClienteFalso:
    def __init__(self, respuestas: list[str]) -> None:
        self.respuestas = list(respuestas)
        self.pedidos: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.pedidos.append(user)
        return self.respuestas.pop(0) if self.respuestas else "cube([1,1,1]);"


def cubo(lado: float = 10.0) -> np.ndarray:
    e = np.array(
        [
            [0, 0, 0], [lado, 0, 0], [lado, lado, 0], [0, lado, 0],
            [0, 0, lado], [lado, 0, lado], [lado, lado, lado], [0, lado, lado],
        ],
        dtype=np.float64,
    )
    caras = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3),
    ]
    return np.array([[e[i] for i in c] for c in caras], dtype=np.float64)


def render_de(lado: float = 10.0):
    """Un render falso que escribe un STL DE VERDAD, del tamano que se le pida.

    Escribir bytes cualquiera no serviria: el bucle mide la pieza, asi que el
    doble tiene que producir geometria medible.
    """

    def render(code, *, openscad, destination, **_kwargs) -> RenderResult:
        destination = Path(destination)
        write_stl(destination, cubo(lado))
        destination.with_suffix(".scad").write_text(code, encoding="utf-8")
        return RenderResult(stl_path=destination, log="")

    return render


render_que_anda = render_de(10.0)


def render_que_falla(_code, *, openscad, destination, **_kwargs) -> RenderResult:
    raise OpenScadError("syntax error at line 3")


def render_que_falla_una_vez():
    estado = {"intentos": 0}

    def render(code, *, openscad, destination, **_kwargs) -> RenderResult:
        estado["intentos"] += 1
        if estado["intentos"] == 1:
            raise OpenScadError("Can't parse file: unexpected end of input")
        return render_de(10.0)(code, openscad=openscad, destination=destination)

    return render


class TestHappyPath:
    def test_code_that_compiles_comes_back_with_its_stl(self, tmp_path: Path) -> None:
        cliente = ClienteFalso(["```openscad\ncube([10,10,10]);\n```"])

        resultado = describe_to_stl(
            "un taco de 10",
            client=cliente,
            openscad=tmp_path / "openscad.exe",
            destination=tmp_path / "pieza.stl",
            render=render_que_anda,
        )

        assert resultado.stl_path.exists()
        assert resultado.retries == 0

    def test_the_editable_source_comes_back_too(self, tmp_path: Path) -> None:
        # El .scad es la pieza EDITABLE: entregar solo el STL seria entregar algo
        # que no se puede ajustar, que es justo lo contrario de tener cotas.
        cliente = ClienteFalso(["```openscad\ncube([10,10,10]);\n```"])

        resultado = describe_to_stl(
            "un taco",
            client=cliente,
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            render=render_que_anda,
        )

        assert "cube" in resultado.code

    def test_the_description_reaches_the_model(self, tmp_path: Path) -> None:
        cliente = ClienteFalso(["cube([1,1,1]);"])

        describe_to_stl(
            "un espaciador de 20 mm",
            client=cliente,
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            render=render_que_anda,
        )

        assert "un espaciador de 20 mm" in cliente.pedidos[0]


class TestTheRetryLoop:
    def test_a_compile_error_goes_back_to_the_model(self, tmp_path: Path) -> None:
        cliente = ClienteFalso(["cube([1,1,1]", "cube([1,1,1]);"])

        resultado = describe_to_stl(
            "un taco",
            client=cliente,
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            render=render_que_falla_una_vez(),
        )

        assert resultado.retries == 1
        # El segundo pedido tiene que llevar el error TEXTUAL: sin eso el modelo
        # reintenta a ciegas y suele repetir el mismo fallo.
        assert "unexpected end of input" in cliente.pedidos[1]

    def test_the_second_prompt_also_carries_the_broken_code(self, tmp_path: Path) -> None:
        cliente = ClienteFalso(["cube([1,1,1]", "cube([1,1,1]);"])

        describe_to_stl(
            "un taco",
            client=cliente,
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            render=render_que_falla_una_vez(),
        )

        assert "cube([1,1,1]" in cliente.pedidos[1]

    def test_it_gives_up_after_the_configured_attempts(self, tmp_path: Path) -> None:
        cliente = ClienteFalso(["malo", "malo", "malo", "malo"])

        with pytest.raises(LlmUnavailable, match="3 intentos"):
            describe_to_stl(
                "un taco",
                client=cliente,
                openscad=tmp_path / "o.exe",
                destination=tmp_path / "p.stl",
                attempts=3,
                render=render_que_falla,
            )

        assert len(cliente.pedidos) == 3

    def test_giving_up_says_what_the_compiler_complained_about(self, tmp_path: Path) -> None:
        # "No se pudo" sin el motivo no le sirve a nadie para corregir el pedido.
        cliente = ClienteFalso(["malo", "malo"])

        with pytest.raises(LlmUnavailable, match="syntax error"):
            describe_to_stl(
                "un taco",
                client=cliente,
                openscad=tmp_path / "o.exe",
                destination=tmp_path / "p.stl",
                attempts=2,
                render=render_que_falla,
            )

    def test_every_attempt_is_kept(self, tmp_path: Path) -> None:
        cliente = ClienteFalso(["roto", "bueno"])

        resultado = describe_to_stl(
            "un taco",
            client=cliente,
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            render=render_que_falla_una_vez(),
        )

        assert len(resultado.attempts) == 2
        assert resultado.attempts[0].error
        assert resultado.attempts[1].error is None


class TestGuards:
    def test_an_empty_description_is_refused_before_calling_the_model(
        self, tmp_path: Path
    ) -> None:
        cliente = ClienteFalso([])

        with pytest.raises(LlmUnavailable, match="descripcion"):
            describe_to_stl(
                "   ",
                client=cliente,
                openscad=tmp_path / "o.exe",
                destination=tmp_path / "p.stl",
                render=render_que_anda,
            )

        assert cliente.pedidos == []

    def test_zero_attempts_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            describe_to_stl(
                "algo",
                client=ClienteFalso(["x"]),
                openscad=tmp_path / "o.exe",
                destination=tmp_path / "p.stl",
                attempts=0,
                render=render_que_anda,
            )


class TestTheDimensionalLoop:
    """La realimentacion que importa: compila, se imprime, y NO ENTRA.

    Medido con `qwen3-coder:30b`: las cuatro piezas pedidas compilaron a la
    primera y las cuatro salieron imprimibles, pero DOS midieron mal — un
    espaciador de 12 mm de alto salio de 24. El fallo tipico de este carril no es
    de sintaxis, es de COTAS, que es lo unico que este carril existe para dar.
    """

    def test_a_part_with_the_wrong_size_is_sent_back(self, tmp_path: Path) -> None:
        cliente = ClienteFalso(["primera", "segunda"])

        describe_to_stl(
            "un taco de 20",
            client=cliente,
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            expected_size=(20.0, 20.0, 20.0),
            attempts=2,
            render=render_de(10.0),
        )

        assert len(cliente.pedidos) == 2
        assert "20.0" in cliente.pedidos[1] and "10.00" in cliente.pedidos[1]

    def test_a_part_with_the_right_size_is_accepted_at_once(self, tmp_path: Path) -> None:
        cliente = ClienteFalso(["buena"])

        resultado = describe_to_stl(
            "un taco de 10",
            client=cliente,
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            expected_size=(10.0, 10.0, 10.0),
            render=render_de(10.0),
        )

        assert resultado.retries == 0
        assert len(cliente.pedidos) == 1

    def test_the_orientation_does_not_count_as_a_mistake(self, tmp_path: Path) -> None:
        # Se pidio una pieza, no una orientacion: rechazarla por estar acostada
        # seria rechazar algo correcto.
        cliente = ClienteFalso(["buena"])

        resultado = describe_to_stl(
            "un taco",
            client=cliente,
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            expected_size=(10.0, 10.0, 10.0),
            render=render_de(10.0),
        )

        assert resultado.retries == 0

    def test_without_an_expected_size_nothing_is_checked(self, tmp_path: Path) -> None:
        # Sin cota pedida no hay contra que comparar, y reintentar seria inventar
        # un criterio.
        cliente = ClienteFalso(["cualquiera"])

        resultado = describe_to_stl(
            "algo",
            client=cliente,
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            render=render_de(37.0),
        )

        assert resultado.retries == 0
        assert resultado.measured[0] == pytest.approx(37.0)

    def test_the_measurement_travels_with_the_result(self, tmp_path: Path) -> None:
        resultado = describe_to_stl(
            "algo",
            client=ClienteFalso(["x"]),
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            render=render_de(25.0),
        )

        assert resultado.measured == pytest.approx((25.0, 25.0, 25.0))

    def test_it_gives_up_after_the_attempts_and_returns_the_best_it_had(
        self, tmp_path: Path
    ) -> None:
        # Devolver la pieza mal medida es mejor que no devolver nada: el usuario
        # ve LO QUE MIDE y decide. Esconderla no le da mas informacion.
        cliente = ClienteFalso(["a", "b"])

        resultado = describe_to_stl(
            "un taco de 20",
            client=cliente,
            openscad=tmp_path / "o.exe",
            destination=tmp_path / "p.stl",
            expected_size=(20.0, 20.0, 20.0),
            attempts=2,
            render=render_de(10.0),
        )

        assert resultado.measured[0] == pytest.approx(10.0)
        assert resultado.attempts[-1].error
