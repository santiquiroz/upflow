"""Los logs normales del servidor no pueden verse como un error.

Reportado el 2026-08-06 desde la maquina de otro usuario: al arrancar Upflow
aparecia un bloque rojo con `NativeCommandError` y el texto del arranque de
uvicorn adentro. La app funcionaba perfecto —las peticiones siguientes daban
200 OK— pero el arranque parecia roto.

Causa: uvicorn escribe sus logs normales en stderr, y Windows PowerShell envuelve
CADA linea de stderr de un programa nativo en un objeto ErrorRecord. El host
pinta de rojo cualquier ErrorRecord, aunque el texto diga "INFO".

`2>&1` no alcanza: fusiona los flujos pero deja los objetos como ErrorRecord.
Hace falta convertirlos a texto.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LANZADOR = Path(__file__).resolve().parent.parent / "scripts" / "upflow-launcher.ps1"

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="Hace falta Windows PowerShell, que es el que corre en la instalacion",
)


def tipos_de_salida(tuberia: str) -> list[str]:
    """Que tipo de objeto entrega PowerShell para cada linea de stderr."""
    guion = (
        "$ErrorActionPreference = 'Continue'\n"
        "$py = 'python'\n"
        f"$salida = {tuberia}\n"
        "foreach ($item in $salida) {{ Write-Output $item.GetType().Name }}"
    ).replace("{{", "{").replace("}}", "}")
    proceso = subprocess.run(
        ["powershell", "-NoProfile", "-Command", guion],
        capture_output=True,
        timeout=180,
    )
    return proceso.stdout.decode("utf-8", "replace").split()


ESCRIBE_EN_STDERR = (
    "& $py -c \"import sys; sys.stderr.write('INFO: arrancando' + chr(10))\""
)


def test_sin_convertir_powershell_lo_marca_como_error() -> None:
    # El bug, demostrado. Sin esto la prueba de abajo podria quedar en verde sin
    # haber visto nunca el problema que dice prevenir.
    assert "ErrorRecord" in tipos_de_salida(f"{ESCRIBE_EN_STDERR} 2>&1")


def test_convertido_a_texto_sale_como_salida_normal() -> None:
    tipos = tipos_de_salida(f'{ESCRIBE_EN_STDERR} 2>&1 | ForEach-Object {{ "$_" }}')

    assert tipos, "no salio nada"
    assert "ErrorRecord" not in tipos
    assert set(tipos) == {"String"}


def test_el_lanzador_convierte_la_salida_del_servidor() -> None:
    # La linea que arranca uvicorn tiene que pasar por la conversion; si alguien
    # la saca, el arranque vuelve a verse roto en la maquina del usuario.
    texto = LANZADOR.read_text(encoding="utf-8", errors="replace")
    arranque = next(
        linea for linea in texto.splitlines() if "uvicorn app.main:app" in linea
    )
    posterior = texto[texto.index(arranque) : texto.index(arranque) + 400]

    assert 'ForEach-Object { "$_" }' in posterior, arranque
