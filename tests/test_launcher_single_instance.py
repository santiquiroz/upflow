"""La segunda ventana del lanzador no instala nada.

Reportado el 2026-10-28: el instalador abria Upflow.bat al terminar
(postinstall) y el usuario abria otra ventana desde el acceso directo 38 s
despues. Las dos ventanas corrian `pip install --force-reinstall` a la vez
sobre el mismo site-packages: una renombraba yt-dlp a `~t_dlp-...dist-info`
mientras la otra lo leia y fallaba con "ERROR: Could not install packages
due to an OSError: [WinError 2]". La misma orden corrida sola funciona.

El lanzador toma un mutex con nombre: la ventana que llega
segunda no instala nada. Si el servidor ya responde, abre el navegador; si
todavia se esta instalando, solo avisa y sale.
"""

from __future__ import annotations

import ctypes
import os
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


def preparar_copia(tmp_path: Path) -> Path:
    """Copia el lanzador a un tmp para que su $root no toque el repo."""
    destino = tmp_path / "scripts" / "upflow-launcher.ps1"
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LANZADOR, destino)
    # Puerto donde nada escucha: la segunda ventana no abre el navegador.
    (tmp_path / ".env").write_text("APP_PORT=1\n", encoding="utf-8")
    return destino


def test_segunda_ventana_no_instala(tmp_path: Path) -> None:
    # Con el mutex ya tomado, el lanzador es la "segunda ventana": avisa y
    # sale sin llegar a 'Verificando Python' (no instala nada).
    script = preparar_copia(tmp_path)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    h = kernel32.CreateMutexW(None, True, "Local\\UpflowLauncher")
    assert h, "no se pudo tomar el mutex del lanzador"
    try:
        env = dict(os.environ, UPFLOW_NO_PAUSE="1")
        proceso = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(script)],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=60,
        )
    finally:
        kernel32.ReleaseMutex(h)
        kernel32.CloseHandle(h)
    texto = proceso.stdout.decode("utf-8", "replace")
    assert proceso.returncode == 0
    assert "otra ventana" in texto
    assert "Verificando Python" not in texto

