# SPDX-License-Identifier: GPL-2.0-or-later
#
# Parte de Upflow. Corre DENTRO de Blender y usa `bpy`. Ver LICENSE en esta carpeta.
"""Rehace la topologia de una malla por voxeles y dice que cambio.

Entrada: {"mesh": "<ruta>", "output": "<ruta>", "voxelMeters": 0.01}
Salida:  la auditoria ANTES y DESPUES.

El veredicto viaja con el archivo porque un remesh no es gratis: gana topologia
uniforme y pierde detalle, y cuanto perdio solo se sabe midiendo las dos puntas.

POR QUE NO HAY MODO "CUADRUPLES" (QuadriFlow), que seria el que sirve para
esculpir y animar: no corre acá. Medido el 2026-08-28 en Blender 5.2.1 con
`--background`, sobre el blockout de un personaje:

  - `bpy.ops.object.quadriflow_remesh` devuelve {'CANCELLED'} y NO lanza
    excepcion, asi que un llamador ingenuo reporta "remesh hecho" con la malla
    intacta — mismas caras antes y despues.
  - Blender avisa "the mesh needs to be manifold and have face normals that
    point in a consistent direction", pero la malla pasa TODOS los chequeos:
    0 aristas no-manifold, 0 vertices no-manifold, 0 vertices sueltos, 0 caras
    duplicadas. Recalcular las normales tampoco lo destraba.
  - O sea: la causa NO esta establecida. Lo unico probado es que cancela.

Se ofrece solo lo que funciona. Cuando se sepa por que cancela, vuelve.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

from _common import audit, emit, export_mesh, fail, import_mesh, payload, reset_scene, verdict  # noqa: E402

VOXEL_POR_DEFECTO_M = 0.01
# Debajo de esto una malla de personaje pasa de los millones de caras y el
# remesh deja de terminar en un tiempo util.
VOXEL_MINIMO_M = 0.002


def remesh_por_voxeles(objeto: bpy.types.Object, voxel_m: float) -> None:
    """Funde todo en una cascara cerrada del tamano de voxel pedido.

    Es lo que convierte primitivas que se solapan —o una malla con agujeros—
    en una sola superficie. Uniformiza: no respeta aristas vivas.
    """
    modificador = objeto.modifiers.new("remesh", "REMESH")
    modificador.mode = "VOXEL"
    modificador.voxel_size = voxel_m
    modificador.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=modificador.name)


def main() -> None:
    datos = payload()
    entrada, salida = datos.get("mesh", ""), datos.get("output", "")
    voxel_m = float(datos.get("voxelMeters") or VOXEL_POR_DEFECTO_M)

    if not entrada or not salida:
        fail("faltan 'mesh' y 'output'")
    if not os.path.exists(entrada):
        fail(f"no existe: {entrada}")
    if voxel_m < VOXEL_MINIMO_M:
        fail(f"voxelMeters minimo {VOXEL_MINIMO_M} m: mas fino no termina en un tiempo util")

    reset_scene()
    objeto = import_mesh(entrada)
    antes = verdict(audit(objeto))

    bpy.context.view_layer.objects.active = objeto
    bpy.ops.object.select_all(action="DESELECT")
    objeto.select_set(True)
    remesh_por_voxeles(objeto, voxel_m)

    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    export_mesh(objeto, salida)
    emit({"mesh": salida, "voxelMeters": voxel_m, "before": antes, "after": verdict(audit(objeto))})


if __name__ == "__main__":
    main()
