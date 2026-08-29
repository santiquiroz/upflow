# SPDX-License-Identifier: GPL-2.0-or-later
#
# Parte de Upflow. Este archivo corre DENTRO de Blender y usa `bpy`, por eso
# lleva GPL y no la MIT del resto del repo. Ver LICENSE en esta misma carpeta.
"""Piezas compartidas por los scripts que corren dentro de Blender.

Los scripts se invocan como `blender --background --factory-startup --python
<script> -- <json>` y contestan por una linea de stdout con centinela. Todo eso
vive aca para que cada script sea solo su operacion.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import bmesh
import bpy

RESULT_SENTINEL = "UPFLOW_RESULT "

IMPORTERS = {
    ".stl": lambda ruta: bpy.ops.wm.stl_import(filepath=ruta),
    ".obj": lambda ruta: bpy.ops.wm.obj_import(filepath=ruta),
    ".ply": lambda ruta: bpy.ops.wm.ply_import(filepath=ruta),
    ".glb": lambda ruta: bpy.ops.import_scene.gltf(filepath=ruta),
    ".gltf": lambda ruta: bpy.ops.import_scene.gltf(filepath=ruta),
    ".fbx": lambda ruta: bpy.ops.import_scene.fbx(filepath=ruta),
}

EXPORTERS = {
    ".stl": lambda ruta: bpy.ops.wm.stl_export(filepath=ruta),
    ".obj": lambda ruta: bpy.ops.wm.obj_export(filepath=ruta),
    ".glb": lambda ruta: bpy.ops.export_scene.gltf(filepath=ruta, export_format="GLB"),
    ".fbx": lambda ruta: bpy.ops.export_scene.fbx(filepath=ruta),
}


def payload() -> dict[str, Any]:
    """Lo que mando el servicio, del unico argumento posterior a `--`."""
    if "--" not in sys.argv:
        return {}
    crudo = sys.argv[sys.argv.index("--") + 1 :]
    return json.loads(crudo[0]) if crudo else {}


def emit(datos: dict[str, Any]) -> None:
    print(RESULT_SENTINEL + json.dumps(datos, ensure_ascii=False))
    sys.stdout.flush()


def fail(mensaje: str) -> None:
    """Reporta el fallo por el mismo canal que el exito y termina bien.

    Sale con codigo 0 a proposito: el que llama distingue exito de fallo por el
    contenido del JSON, no por el codigo de salida. Blender devuelve codigos
    propios por motivos que no son el script, y mezclar los dos canales hace
    que un warning de OpenGL parezca un error de la operacion.
    """
    emit({"error": mensaje})
    sys.exit(0)


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(ruta: str) -> bpy.types.Object:
    sufijo = "." + ruta.rsplit(".", 1)[-1].lower()
    importador = IMPORTERS.get(sufijo)
    if importador is None:
        fail(f"formato de entrada no soportado: {sufijo}")
    importador(ruta)
    mallas = [objeto for objeto in bpy.context.scene.objects if objeto.type == "MESH"]
    if not mallas:
        fail(f"el archivo no trajo ninguna malla: {ruta}")
    return join_meshes(mallas)


def join_meshes(mallas: list[bpy.types.Object]) -> bpy.types.Object:
    """Un solo objeto, porque las operaciones siguientes trabajan sobre uno.

    Un GLB de personaje suele venir partido en cuerpo, ropa y pelo; auditar
    solo el primero seria auditar un tercio y decir que esta todo bien.
    """
    principal = mallas[0]
    if len(mallas) == 1:
        return principal
    bpy.ops.object.select_all(action="DESELECT")
    for malla in mallas:
        malla.select_set(True)
    bpy.context.view_layer.objects.active = principal
    bpy.ops.object.join()
    return principal


def export_mesh(objeto: bpy.types.Object, ruta: str) -> None:
    sufijo = "." + ruta.rsplit(".", 1)[-1].lower()
    exportador = EXPORTERS.get(sufijo)
    if exportador is None:
        fail(f"formato de salida no soportado: {sufijo}")
    bpy.ops.object.select_all(action="DESELECT")
    objeto.select_set(True)
    bpy.context.view_layer.objects.active = objeto
    exportador(ruta)


def count_shells(bm: bmesh.types.BMesh) -> int:
    """Islas conectadas, marcando al APILAR y no al desapilar.

    Marcar al desapilar deja la pila crecer con duplicados hasta millones de
    entradas en una malla de 25k caras, y Blender se cae.
    """
    bm.faces.ensure_lookup_table()
    vistas = [False] * len(bm.faces)
    islas = 0
    for cara in bm.faces:
        if vistas[cara.index]:
            continue
        islas += 1
        vistas[cara.index] = True
        pila = [cara]
        while pila:
            actual = pila.pop()
            for arista in actual.edges:
                for vecina in arista.link_faces:
                    if not vistas[vecina.index]:
                        vistas[vecina.index] = True
                        pila.append(vecina)
    return islas


def audit(objeto: bpy.types.Object) -> dict[str, Any]:
    """El estado de la malla, en los terminos que deciden si sirve.

    Va antes Y despues de cada operacion: encadenar operaciones sin medir entre
    medio es como aplicar parches sin compilar.

    NO se reporta la relacion entre el eje mas fino y el mas grueso. Parece el
    detector obvio de "esto salio como una plancha" y NO lo es: medido el
    2026-08-25 sobre dos mallas generadas del mismo objeto, la que salio
    destrozada a partir de arte plano dio 0.767 y la buena 0.649 — o sea, el
    numero fue MAS ALTO en la peor. Lo que si distingue esos dos casos es
    `shells` y `boundaryEdges`, que ya viajan.
    """
    bm = bmesh.new()
    bm.from_mesh(objeto.data)
    try:
        caras = bm.faces
        dims = [round(float(valor), 6) for valor in objeto.dimensions]
        return {
            "vertices": len(bm.verts),
            "faces": len(caras),
            "tris": sum(1 for cara in caras if len(cara.verts) == 3),
            "quads": sum(1 for cara in caras if len(cara.verts) == 4),
            "ngons": sum(1 for cara in caras if len(cara.verts) > 4),
            "nonManifoldEdges": sum(1 for arista in bm.edges if not arista.is_manifold),
            "boundaryEdges": sum(1 for arista in bm.edges if arista.is_boundary),
            "looseVerts": sum(1 for vert in bm.verts if not vert.link_edges),
            "shells": count_shells(bm),
            "dims": dims,
            "hasUvs": bool(objeto.data.uv_layers),
        }
    finally:
        bm.free()


def verdict(reporte: dict[str, Any]) -> dict[str, Any]:
    """Traduce el conteo a las tres preguntas que se hacen de verdad.

    El conteo crudo no le dice nada a quien encadena operaciones sin mirar la
    malla. `blockers` es lo que impide seguir; `warnings` es lo que saldria
    mejor de otra forma. Mezclarlos entrena a ignorar los dos.
    """
    blockers: list[str] = []
    warnings: list[str] = []

    if reporte["faces"] == 0:
        blockers.append("la malla no tiene caras")
    if reporte["nonManifoldEdges"]:
        blockers.append(f"{reporte['nonManifoldEdges']} aristas no-manifold")
    if reporte["boundaryEdges"]:
        warnings.append(f"{reporte['boundaryEdges']} aristas de borde (malla abierta)")
    if reporte["shells"] > 1:
        warnings.append(f"{reporte['shells']} islas sueltas")
    if reporte["looseVerts"]:
        warnings.append(f"{reporte['looseVerts']} vertices sueltos")
    if reporte["ngons"]:
        warnings.append(f"{reporte['ngons']} n-gons")

    return {**reporte, "blockers": blockers, "warnings": warnings, "ok": not blockers}
