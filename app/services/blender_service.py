"""Ejecuta Blender como proceso aparte y le habla por JSON.

Blender es GPL, asi que corre SIEMPRE como proceso separado y nunca se enlaza
— mismo trato que OpenSCAD y Magpie, que ya viajan asi en este repo por la
misma razon. Los scripts que corren adentro viven en `blender_scripts/` y
llevan su propia licencia GPL; el resto del arbol sigue MIT.

El contrato entre los dos procesos es UNA linea de stdout con centinela y JSON.
Blender escribe mucho ruido propio (version, addons, warnings de OpenGL) y
parsear texto libre en el medio de eso es como leer la salida de un compilador
con expresiones regulares: anda hasta que cambia una version.

Se devuelve la salida cruda cuando algo falla, entera y sin recortar el motivo.
Es lo mismo que hace `openscad_render`: tragarse el error deja al que reintenta
—persona o agente— sin nada con que corregirse.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.missing_pack import MissingPack

logger = logging.getLogger(__name__)

RESULT_SENTINEL = "UPFLOW_RESULT "

# Blender 4.x movio los importadores a `wm.*_import` y cambio nombres de
# operadores que los scripts usan. Debajo de eso los scripts no corren, y
# fallar aca con un mensaje claro es mejor que fallar adentro con un
# AttributeError sobre un operador que no existe.
MINIMUM_VERSION = (4, 2)

# Auditar una malla grande recorre cada cara y cada arista en Python dentro de
# Blender, y eso escala con el tamano del archivo; un script colgado no debe
# quedarse para siempre. El numero no esta medido contra un caso limite: es un
# techo, no un presupuesto.
DEFAULT_TIMEOUT_S = 900

VERSION_PATTERN = re.compile(r"Blender\s+(\d+)\.(\d+)(?:\.(\d+))?")


class BlenderError(RuntimeError):
    """Blender corrio y fallo. Lleva la salida cruda a proposito."""

    def __init__(self, message: str, *, output: str = "") -> None:
        self.output = output
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class BlenderBuild:
    path: Path
    version: tuple[int, int, int]

    @property
    def version_string(self) -> str:
        return ".".join(str(parte) for parte in self.version)

    @property
    def meets_minimum(self) -> bool:
        return self.version[:2] >= MINIMUM_VERSION


def parse_version(texto: str) -> tuple[int, int, int] | None:
    encontrado = VERSION_PATTERN.search(texto)
    if not encontrado:
        return None
    mayor, menor, parche = encontrado.groups()
    return int(mayor), int(menor), int(parche or 0)


def probe(settings: Settings, *, timeout: float = 30) -> BlenderBuild | None:
    """Que Blender hay, o None si no hay ninguno usable.

    Devuelve None en vez de tirar: preguntar si una capacidad esta disponible
    no es un error. `require_build` convierte ese None en un MissingPack con el
    mensaje que ve el usuario, y la ruta de capacidades del carril lo usa para
    responder "apagado" sin fallar.
    """
    binario = settings.blender_binary_path
    if not binario.exists():
        return None
    try:
        salida = subprocess.run(
            [str(binario), "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Blender no respondio en %s: %s", binario, exc)
        return None

    version = parse_version(salida.stdout or "")
    if version is None:
        return None
    return BlenderBuild(path=binario, version=version)


def require_build(settings: Settings) -> BlenderBuild:
    build = probe(settings)
    if build is None:
        raise MissingPack("blender")
    if not build.meets_minimum:
        minimo = ".".join(str(parte) for parte in MINIMUM_VERSION)
        raise BlenderError(
            f"Blender {build.version_string} es anterior a {minimo}, "
            "que es donde los importadores pasaron a wm.*_import."
        )
    return build


def script_path(settings: Settings, name: str) -> Path:
    """La ruta de un script vendorizado, validando que no se salga de su carpeta.

    El nombre puede venir de la capa HTTP; sin esta comprobacion un `..` deja
    ejecutar cualquier archivo del disco dentro de Blender.
    """
    carpeta = settings.blender_scripts_dir.resolve()
    candidato = (carpeta / name).resolve()
    if candidato.parent != carpeta or candidato.suffix != ".py":
        raise BlenderError(f"script fuera de la carpeta vendorizada: {name}")
    if not candidato.exists():
        raise BlenderError(f"script inexistente: {name}")
    return candidato


def extract_result(stdout: str) -> dict[str, Any] | None:
    """El JSON de la ULTIMA linea centinela.

    La ultima y no la primera: un script que reporta progreso emite varias, y
    la que cierra es la que trae el resultado.
    """
    encontrado: dict[str, Any] | None = None
    for linea in stdout.splitlines():
        if not linea.startswith(RESULT_SENTINEL):
            continue
        try:
            encontrado = json.loads(linea[len(RESULT_SENTINEL) :])
        except json.JSONDecodeError:
            continue
    return encontrado


def run_script(
    settings: Settings,
    script: str,
    payload: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Corre un script dentro de Blender y devuelve lo que reporto.

    `--factory-startup` va siempre: sin eso los addons y preferencias del
    usuario entran a la escena y el resultado deja de ser reproducible entre
    dos maquinas.
    """
    build = require_build(settings)
    ruta = script_path(settings, script)
    comando = [
        str(build.path),
        "--background",
        "--factory-startup",
        "--python",
        str(ruta),
        "--",
        json.dumps(payload),
    ]

    try:
        proceso = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BlenderError(f"{script} paso los {timeout:.0f} s") from exc

    salida = (proceso.stdout or "") + (proceso.stderr or "")
    resultado = extract_result(proceso.stdout or "")

    if proceso.returncode != 0 or resultado is None or resultado.get("error"):
        # La salida entera va al LOG del servidor y no al cliente: es donde
        # esta el traceback de Blender, y ademas trae rutas absolutas de la
        # maquina que no le sirven a nadie del otro lado.
        logger.error("blender %s fallo (codigo %s): %s", script, proceso.returncode, salida)

    if proceso.returncode != 0 and resultado is None:
        raise BlenderError(f"{script} termino en {proceso.returncode}", output=salida)
    if resultado is None:
        raise BlenderError(f"{script} no reporto resultado", output=salida)
    if resultado.get("error"):
        raise BlenderError(str(resultado["error"]), output=salida)

    logger.info("blender %s corrio %s", build.version_string, script)
    return resultado
