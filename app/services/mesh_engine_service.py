"""Corre un motor generativo de malla como proceso aparte y le habla por JSON.

Mismo trato que Blender, por razones distintas pero igual de duras. Blender va
aparte porque es GPL; estos van aparte porque **sus dependencias son
incompatibles con las de la app**: TripoSG fija `numpy==1.22.3`, que rompe la
mitad del resto del arbol. Instalarlos en el entorno de Upflow para "simplificar"
cambia un problema de orquestacion por uno de dependencias irreparable.

Cada motor vive en `{mesh_engines_dir}/{nombre}-env` con su propio interprete, y
su script de entrada esta vendorizado en `engine_scripts/`. El contrato entre
los dos procesos es UNA linea de stdout con centinela y JSON, igual que con
Blender: estas tuberias escupen barras de progreso y warnings de torch, y
parsear texto libre en el medio de eso anda hasta que cambia una version.

NINGUN motor se descarga solo. Son varios GB de pesos con licencias distintas
entre si —MIT en TripoSG, comunitaria con exclusion territorial en Hunyuan3D— y
elegir cual bajar es una decision del usuario, no un efecto secundario de
apretar un boton.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.blender_service import extract_result

logger = logging.getLogger(__name__)

# Generar una malla en CPU son minutos, no segundos: medido sobre TripoSG en
# esta maquina. El numero es un techo para que un proceso colgado no se quede
# para siempre, no un presupuesto.
DEFAULT_TIMEOUT_S = 3600

ENGINE_SCRIPTS_DIR = Path(__file__).resolve().parent / "engine_scripts"

# Que le hace falta a cada motor, ademas de su entorno. `source` es el repo del
# motor, que trae el codigo del modelo y no viaja en este arbol.
ENGINES = {
    "triposg": {
        "script": "triposg_generate.py",
        "source": "TripoSG",
        "license": "MIT",
        "device": "cpu",
    },
}


class MeshEngineError(RuntimeError):
    """El motor corrio y fallo. Lleva la salida cruda a proposito."""

    def __init__(self, message: str, *, output: str = "") -> None:
        self.output = output
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class EngineBuild:
    name: str
    python: Path
    source: Path
    license: str

    @property
    def ready(self) -> bool:
        return self.python.exists() and self.source.exists()

    @property
    def missing(self) -> str | None:
        """Que falta exactamente, en terminos accionables.

        Decir "no disponible" manda a adivinar entre bajar 8 GB de pesos y
        crear un entorno; decir cual de los dos falta no.
        """
        if not self.python.exists():
            return f"falta el entorno del motor en {self.python.parent.parent}"
        if not self.source.exists():
            return f"falta el codigo del motor en {self.source}"
        return None


def _interpreter_name() -> tuple[str, str]:
    return ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")


def build_for(settings: Settings, name: str) -> EngineBuild:
    """Donde estaria ese motor, exista o no."""
    if name not in ENGINES:
        raise MeshEngineError(f"motor desconocido: {name}")
    ficha = ENGINES[name]
    raiz = settings.mesh_engines_dir
    carpeta, ejecutable = _interpreter_name()
    return EngineBuild(
        name=name,
        python=raiz / f"{name}-env" / carpeta / ejecutable,
        source=raiz / str(ficha["source"]),
        license=str(ficha["license"]),
    )


def available(settings: Settings) -> dict[str, dict[str, Any]]:
    """Que motores hay HOY en esta maquina, y por que falta cada uno que falta.

    Nunca tira: preguntar por una capacidad ausente no es un error, es la
    respuesta. Es la misma regla que ya rige el carril de Blender.
    """
    estado: dict[str, dict[str, Any]] = {}
    for nombre in ENGINES:
        build = build_for(settings, nombre)
        estado[nombre] = {
            "ready": build.ready,
            "license": build.license,
            "device": ENGINES[nombre]["device"],
            "missing": build.missing,
        }
    return estado


def script_path(name: str) -> Path:
    """El script vendorizado del motor, validado contra fuga de carpeta."""
    carpeta = ENGINE_SCRIPTS_DIR.resolve()
    candidato = (carpeta / name).resolve()
    if candidato.parent != carpeta or candidato.suffix != ".py":
        raise MeshEngineError(f"script fuera de la carpeta vendorizada: {name}")
    if not candidato.exists():
        raise MeshEngineError(f"script inexistente: {name}")
    return candidato


def generate(
    settings: Settings,
    engine: str,
    payload: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Genera una malla con el motor pedido y devuelve lo que reporto.

    Lo que sale de aca NO esta aprobado por salir: pasa por el banco como
    cualquier otra malla. Un generador puede devolver una superficie preciosa
    con doscientas islas sueltas, y "listo" sobre eso es el peor falso
    positivo, el que da confianza.
    """
    build = build_for(settings, engine)
    if not build.ready:
        raise MeshEngineError(f"el motor '{engine}' no esta listo: {build.missing}")

    ficha = ENGINES[engine]
    comando = [
        str(build.python),
        str(script_path(str(ficha["script"]))),
        json.dumps({**payload, "sourceDir": str(build.source)}),
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
        raise MeshEngineError(f"{engine} paso los {timeout:.0f} s") from exc

    salida = (proceso.stdout or "") + (proceso.stderr or "")
    resultado = extract_result(proceso.stdout or "")

    if proceso.returncode != 0 or resultado is None or resultado.get("error"):
        # Entera y al log del servidor: es donde esta el traceback, y trae
        # rutas absolutas que no le sirven a nadie del otro lado.
        logger.error("motor %s fallo (codigo %s): %s", engine, proceso.returncode, salida)

    if resultado is None:
        raise MeshEngineError(
            f"{engine} termino en {proceso.returncode} sin reportar resultado", output=salida
        )
    if resultado.get("error"):
        raise MeshEngineError(str(resultado["error"]), output=salida)

    logger.info("motor %s genero %s", engine, resultado.get("mesh"))
    return {**resultado, "engine": engine, "license": build.license}
