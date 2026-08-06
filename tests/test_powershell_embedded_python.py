"""Todo Python incrustado en un .ps1 tiene que seguir siendo Python DESPUES de que PowerShell lo expanda.

Esto no es teorico. El 2026-08-05 `download-shap-e.ps1` fallo en la instalacion
real: un comentario que decia `renderer/` entre backticks vivia dentro de un
here-string INTERPOLANTE (@"..."@), donde el backtick es el caracter de escape de
PowerShell. `r se convirtio en un retorno de carro, se comio la "r", partio la
linea en dos y Python murio con `SyntaxError: invalid syntax. Perhaps you forgot
a comma?` sin bajar un solo byte.

Leer el .ps1 no alcanza para verlo: el texto se ve bien en el archivo. Hay que
preguntarle a PowerShell que produce, que es exactamente lo que hace esta prueba.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# Un here-string que arranca con @" y cierra con "@ al principio de una linea.
HERE_STRING = re.compile(r'@"\r?\n(.*?)\r?\n"@', re.DOTALL)

# Delata que el bloque es Python y no, por ejemplo, un aviso de licencia.
PISTAS_DE_PYTHON = ("import ", "print(", "def ", "from ")


def bloques_de_python(texto: str) -> list[str]:
    return [
        bloque
        for bloque in HERE_STRING.findall(texto)
        if any(pista in bloque for pista in PISTAS_DE_PYTHON)
    ]


def expandido_por_powershell(bloque: str) -> str:
    """Lo que PowerShell 5.1 le entrega DE VERDAD a python.exe.

    Se usa Windows PowerShell y no pwsh a proposito: es el que corre en la
    instalacion del usuario, y el que proceso el backtick como escape.
    """
    envoltorio = f'$b = @"\n{bloque}\n"@\n[Console]::Out.Write($b)'
    proceso = subprocess.run(
        ["powershell", "-NoProfile", "-Command", envoltorio],
        capture_output=True,
        timeout=120,
    )
    return proceso.stdout.decode("utf-8", "replace")


def scripts_con_python() -> list[tuple[Path, str]]:
    encontrados: list[tuple[Path, str]] = []
    for script in sorted(SCRIPTS.glob("*.ps1")):
        for bloque in bloques_de_python(script.read_text(encoding="utf-8", errors="replace")):
            encontrados.append((script, bloque))
    return encontrados


CASOS = scripts_con_python()

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="Hace falta Windows PowerShell, que es el que corre en la instalacion",
)


@pytest.mark.parametrize(
    "script,bloque",
    CASOS,
    ids=[s.name for s, _ in CASOS] or ["sin-bloques"],
)
def test_el_python_incrustado_sobrevive_a_powershell(script: Path, bloque: str) -> None:
    expandido = expandido_por_powershell(bloque)
    # Las variables de PowerShell se expanden a vacio fuera de su script; se
    # rellenan con un literal para que no rompan la sintaxis por si solas.
    parseable = re.sub(r"r'\s*'", "r'x'", expandido)
    try:
        ast.parse(parseable)
    except SyntaxError as exc:
        pytest.fail(
            f"{script.name}: PowerShell rompio el Python incrustado.\n"
            f"  {exc.msg} en la linea {exc.lineno}\n"
            f"  quedo: {(exc.text or '').strip()!r}\n"
            "  Causa tipica: backtick dentro de un here-string interpolante. "
            "Usa @'...'@ y pasa los valores por variable de entorno."
        )


def test_hay_al_menos_un_bloque_que_revisar() -> None:
    # Sin esto la parametrizacion vacia dejaria la prueba en verde para siempre
    # si alguien renombra los scripts o cambia el formato.
    assert CASOS, "No se encontro Python incrustado en scripts/*.ps1"


# El bloque EXACTO que fallo en la instalacion del usuario el 2026-08-05. Sin
# esto, la prueba de arriba nunca vio el bug que dice cazar y quedaria en verde
# aunque el chequeo estuviera roto.
BLOQUE_QUE_FALLO = """from huggingface_hub import snapshot_download
ruta = snapshot_download(
    'openai/shap-e',
    local_dir=r'C:/x',
    #   - `prior` y `text_encoder` tienen fp16 en safetensors: se baja esa.
    #   - `renderer/` se excluye entero: la tuberia no lo usa y son 1,3 GB.
    allow_patterns=['model_index.json'],
)
print(ruta)"""


def test_el_chequeo_caza_el_bug_que_paso_de_verdad() -> None:
    expandido = expandido_por_powershell(BLOQUE_QUE_FALLO)

    # PowerShell tradujo el backtick-r a un retorno de carro y se comio la "r":
    # donde el archivo decia `renderer/` quedo CR + "enderer/".
    assert "\x0denderer/" in expandido
    assert "`renderer/`" not in expandido
    with pytest.raises(SyntaxError):
        ast.parse(expandido)
