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


def assert_safe(code: str) -> None:
    """Rechaza el codigo que toca el disco.

    El codigo viene de un modelo, no de una persona de confianza. OpenSCAD sabe
    leer archivos, y `include` o `import` con una ruta absoluta convierten una
    descripcion de pieza en una lectura arbitraria del disco.
    """
    # Se mira linea por linea sin comentarios: `// import` no es una llamada.
    for linea in code.splitlines():
        limpia = linea.split("//", 1)[0].strip().lower()
        for prohibido in FORBIDDEN:
            if re.search(rf"(^|[^a-z_]){re.escape(prohibido)}\s*[(<\"']", limpia):
                raise OpenScadError(
                    f"El codigo usa `{prohibido}`, que puede leer archivos del disco. "
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
            f"Falta OpenSCAD en {openscad}. Instalalo con scripts/download-openscad.ps1"
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
