# SPDX-License-Identifier: GPL-2.0-or-later
#
# Parte de Upflow. Corre DENTRO de Blender y usa `bpy`. Ver LICENSE en esta carpeta.
"""Mide una malla y devuelve el veredicto, sin tocarla.

Entrada:  {"mesh": "<ruta>"}
Salida:   el reporte de _common.audit mas blockers/warnings/ok.

Existe aparte de las operaciones que transforman para que se pueda medir ANTES
de decidir que hacer. Un agente que encadena remesh, UV y rig a ciegas no se
entera de que el paso dos arruino lo que hizo el uno.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import audit, emit, fail, import_mesh, payload, reset_scene, verdict  # noqa: E402


def main() -> None:
    datos = payload()
    ruta = datos.get("mesh", "")
    if not ruta:
        fail("falta 'mesh'")
    if not os.path.exists(ruta):
        fail(f"no existe: {ruta}")

    reset_scene()
    objeto = import_mesh(ruta)
    emit({"mesh": ruta, **verdict(audit(objeto))})


if __name__ == "__main__":
    main()
