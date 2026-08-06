"""Convierte codigo OpenSCAD en un STL, ejecutando el binario como proceso aparte.

Esta es la mitad DETERMINISTA del carril de cotas por descripcion: el modelo
escribe el codigo y esto lo convierte en geometria. Si el codigo esta bien, la
pieza mide exactamente lo que dice el codigo — de ahi vienen las cotas que ningun
generador de malla puede dar.

Medido el 2026-08-05: un espaciador escrito a mano (20 exterior, 8,4 interior, 12
de alto) sale de aca midiendo 20,000 x 20,000 x 12,000 mm, estanco, manifold y
solido, con 0,161% de diferencia de volumen contra la formula — pura
discretizacion del circulo a $fn=64.

OpenSCAD es GPL-2.0, asi que corre como PROCESO APARTE y nunca se enlaza. Mismo
trato que Magpie, que ya viaja asi en este repo por la misma razon.

El error de compilacion se devuelve TAL CUAL: es lo que despues le permite al
modelo corregirse. Tragarlo dejaria el bucle de reintento ciego.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.services.missing_pack import missing_pack_message

# Un modelo que se va por las ramas puede escribir un bucle que no termina nunca.
# Noventa segundos alcanzan de sobra para cualquier pieza de esta escala.
RENDER_TIMEOUT_S = 90

# Lo que NO puede aparecer en el codigo. OpenSCAD puede leer y escribir archivos,
# y el codigo viene de un modelo: sin esto, una descripcion maliciosa (o un modelo
# alucinando) podria leer cualquier cosa del disco.
FORBIDDEN = (
    "import",
    "include",
    "use",
    "surface",
    "dxf_",
    "textmetrics",
)


class OpenScadError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RenderResult:
    stl_path: Path
    # La salida cruda del compilador. Va entera a proposito: es lo que le permite
    # al modelo corregirse en el siguiente intento.
    log: str


def extract_code(text: str) -> str:
    """Saca el codigo del bloque markdown, si vino envuelto.

    Los modelos responden con explicacion aunque se les pida que no. Quedarse con
    la respuesta cruda haria fallar la compilacion por texto que no es codigo.
    """
    bloque = re.search(r"```(?:openscad|scad|c)?\s*\n(.*?)```", text, re.S)
    return (bloque.group(1) if bloque else text).strip()


def _sin_comentarios(code: str) -> str:
    """Saca los comentarios para que no escondan una palabra clave.

    Se hace ANTES de buscar, no despues: `include /* x */ <lib>` es codigo valido
    para OpenSCAD, y dejar el comentario en el medio separaria la palabra del `<`.
    """
    sin_bloque = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", sin_bloque)


def assert_safe(code: str) -> None:
    """Corta el codigo que lee archivos antes de que OpenSCAD lo vea.

    Se busca sobre el TEXTO ENTERO, no linea por linea. El lexer de OpenSCAD no
    ve lineas, ve tokens: medido el 2026-08-05, un `include` con el `<archivo>`
    en la linea siguiente pasaba un guard por lineas y OpenSCAD lo ejecutaba
    igual, porque la palabra clave y el `<` nunca caian juntos en una linea.

    Esto importa porque el codigo lo escribe un modelo que puede estar corriendo
    en un servidor que no es de quien usa la app.
    """
    limpio = _sin_comentarios(code).lower()
    for prohibido in FORBIDDEN:
        if re.search(rf"(^|[^a-z_]){re.escape(prohibido)}\s*[(<\"']", limpio):
            raise OpenScadError(
                f"El codigo generado usa `{prohibido}`, que lee archivos del disco. "
                "Una pieza se describe con geometria, no leyendo archivos."
            )


def render_to_stl(
    code: str, *, openscad: Path, destination: Path, timeout_s: int = RENDER_TIMEOUT_S
) -> RenderResult:
    limpio = extract_code(code)
    if not limpio:
        raise OpenScadError("No hay codigo que compilar.")
    assert_safe(limpio)

    openscad = Path(openscad)
    if not openscad.exists():
        raise OpenScadError(
            missing_pack_message("openscad", detail=f"Se esperaba en {openscad}.")
        )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fuente = destination.with_suffix(".scad")
    fuente.write_text(limpio, encoding="utf-8")

    try:
        proceso = subprocess.run(
            [str(openscad), "-o", str(destination), str(fuente)],
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise OpenScadError(
            f"El codigo tardo mas de {timeout_s} s en compilar. Suele ser un bucle "
            "que no termina o una pieza con demasiado detalle."
        ) from exc

    log = proceso.stderr.decode("utf-8", "replace").strip()
    if proceso.returncode != 0 or not destination.exists():
        raise OpenScadError(log or "OpenSCAD no produjo ningun archivo.")
    return RenderResult(stl_path=destination, log=log)
