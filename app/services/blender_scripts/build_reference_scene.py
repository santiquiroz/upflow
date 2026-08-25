# SPDX-License-Identifier: GPL-2.0-or-later
#
# Parte de Upflow. Corre DENTRO de Blender y usa `bpy`. Ver LICENSE en esta carpeta.
"""Arma la escena de referencia desde las vistas de un turnaround.

Entrada:
    {"views": {"front": "<ruta>", "side": "<ruta>", "back": "<ruta>"},
     "heightMeters": 1.70,
     "output": "<ruta .blend>"}

Salida: rutas, altura aplicada y las vistas efectivamente colocadas.

Esto es lo que ninguna IA generativa da y todo modelador necesita: las vistas
alineadas, a ESCALA REAL, cada una detras de su camara y fuera del camino del
raton. Es determinista — no hay modelo de por medio, y sale igual las mil veces.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

from _common import emit, fail, payload, reset_scene  # noqa: E402

REFERENCE_COLLECTION = "referencias"

# Cada vista va DETRAS de su camara, no en el origen: en el origen se pelean
# entre si por el mismo plano y ademas el modelo las atraviesa.
#
# El plano se construye ya parado en XZ mirando a -Y, asi que la vista frontal
# no rota nada; las demas giran sobre Z, que es el eje vertical del personaje.
PLACEMENTS = {
    # (rotacion, eje de desplazamiento, signo)
    "front": ((0.0, 0.0, 0.0), "y", 1.0),
    "back": ((0.0, 0.0, math.pi), "y", -1.0),
    "side": ((0.0, 0.0, math.pi / 2), "x", 1.0),
    "side_left": ((0.0, 0.0, -math.pi / 2), "x", -1.0),
}

DEFAULT_HEIGHT_M = 1.70
BACKDROP_MARGIN_M = 1.0
REFERENCE_ALPHA = 0.45


def use_metric_units(scene: bpy.types.Scene) -> None:
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene.unit_settings.length_unit = "METERS"


def reference_collection(scene: bpy.types.Scene) -> bpy.types.Collection:
    coleccion = bpy.data.collections.new(REFERENCE_COLLECTION)
    scene.collection.children.link(coleccion)
    return coleccion


def reference_material(nombre: str, imagen: bpy.types.Image) -> bpy.types.Material:
    """Textura plana, sin luz y con alfa.

    Emission y no Principled: una referencia que responde a la iluminacion de
    la escena cambia de tono mientras modelas, y lo unico que se le pide es
    mostrar el dibujo tal cual es.
    """
    material = bpy.data.materials.new(f"ref_{nombre}")
    material.use_nodes = True
    material.blend_method = "BLEND"
    arbol = material.node_tree
    arbol.nodes.clear()

    textura = arbol.nodes.new("ShaderNodeTexImage")
    textura.image = imagen
    emision = arbol.nodes.new("ShaderNodeEmission")
    transparente = arbol.nodes.new("ShaderNodeBsdfTransparent")
    mezcla = arbol.nodes.new("ShaderNodeMixShader")
    salida = arbol.nodes.new("ShaderNodeOutputMaterial")

    arbol.links.new(textura.outputs["Color"], emision.inputs["Color"])
    arbol.links.new(textura.outputs["Alpha"], mezcla.inputs[0])
    arbol.links.new(transparente.outputs[0], mezcla.inputs[1])
    arbol.links.new(emision.outputs[0], mezcla.inputs[2])
    arbol.links.new(mezcla.outputs[0], salida.inputs["Surface"])
    return material


def place_reference(
    coleccion: bpy.types.Collection,
    nombre: str,
    ruta: str,
    height_m: float,
    ink_box: tuple[int, int, int, int] | None,
) -> dict[str, object]:
    """Un plano con la vista, a escala real, detras de su camara.

    Plano de malla y no Empty-image: los Empty solo existen en el viewport y no
    aparecen en ningun render, asi que no hay forma de comprobar que midan lo
    que dicen. Un plano se mide.
    """
    imagen = bpy.data.images.load(ruta)
    ancho_px, alto_px = imagen.size
    x0, y0, x1, y1 = ink_box or (0, 0, ancho_px, alto_px)
    tinta_alto = max(y1 - y0, 1)

    # Se escala por el DIBUJO, no por el archivo. Un recorte con margen —o
    # peor, rellenado a cuadrado— deja al personaje mas bajo que la altura
    # pedida, y la referencia miente por todo ese margen.
    alto_plano = height_m * alto_px / tinta_alto
    ancho_plano = alto_plano * ancho_px / alto_px

    # El origen de la escena queda en los PIES y en el centro del personaje,
    # que es donde todo modelador espera encontrarlo.
    bajo_la_tinta = (alto_px - y1) / alto_px * alto_plano
    corrimiento_x = ((x0 + x1) / 2 - ancho_px / 2) / ancho_px * ancho_plano

    izquierda = -ancho_plano / 2 - corrimiento_x
    derecha = ancho_plano / 2 - corrimiento_x
    abajo = -bajo_la_tinta
    arriba = alto_plano - bajo_la_tinta

    malla = bpy.data.meshes.new(f"ref_{nombre}")
    plano = bpy.data.objects.new(f"ref_{nombre}", malla)
    coleccion.objects.link(plano)

    malla.from_pydata(
        [
            (izquierda, 0.0, abajo),
            (derecha, 0.0, abajo),
            (derecha, 0.0, arriba),
            (izquierda, 0.0, arriba),
        ],
        [],
        [(0, 1, 2, 3)],
    )
    malla.update()

    capa_uv = malla.uv_layers.new(name="UVMap")
    for indice, coordenada in enumerate([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]):
        capa_uv.data[indice].uv = coordenada
    malla.materials.append(reference_material(nombre, imagen))

    rotacion, eje, signo = PLACEMENTS[nombre]
    plano.rotation_euler = rotacion
    # Solo el corrimiento del telon: el alto y el centrado ya viajan en los
    # vertices, asi que el objeto queda en Z=0 y su origen sirve de referencia.
    desplazamiento = signo * (height_m / 2 + BACKDROP_MARGIN_M)
    plano.location = (
        desplazamiento if eje == "x" else 0.0,
        desplazamiento if eje == "y" else 0.0,
        0.0,
    )

    # Fuera del render final: es andamiaje, no parte de la pieza.
    plano.hide_render = True
    # Sin esto el modelador agarra la referencia en vez de la malla, cien veces
    # por sesion. Se puede desbloquear desde el outliner cuando haga falta.
    plano.hide_select = True

    return {
        "view": nombre,
        "image": ruta,
        "inkHeightMeters": round(height_m, 4),
        "planeHeightMeters": round(alto_plano, 4),
        "planeWidthMeters": round(ancho_plano, 4),
        "scaledByInk": ink_box is not None,
    }


def main() -> None:
    datos = payload()
    vistas = datos.get("views") or {}
    salida = datos.get("output", "")
    altura = float(datos.get("heightMeters") or DEFAULT_HEIGHT_M)

    if not vistas:
        fail("falta 'views'")

    cajas = {
        nombre: tuple(dato["inkBox"])
        for nombre, dato in vistas.items()
        if isinstance(dato, dict) and dato.get("inkBox")
    }
    vistas = {
        nombre: (dato["image"] if isinstance(dato, dict) else dato)
        for nombre, dato in vistas.items()
    }
    if not salida:
        fail("falta 'output'")
    if altura <= 0:
        fail(f"heightMeters invalido: {altura}")

    desconocidas = sorted(set(vistas) - set(PLACEMENTS))
    if desconocidas:
        fail(f"vistas no reconocidas: {', '.join(desconocidas)}")

    faltantes = [ruta for ruta in vistas.values() if not os.path.exists(ruta)]
    if faltantes:
        fail(f"no existen: {', '.join(faltantes)}")

    reset_scene()
    escena = bpy.context.scene
    use_metric_units(escena)
    coleccion = reference_collection(escena)

    colocadas = [
        place_reference(coleccion, nombre, ruta, altura, cajas.get(nombre))
        for nombre, ruta in vistas.items()
    ]

    os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=salida)

    emit(
        {
            "blend": salida,
            "heightMeters": altura,
            "placed": colocadas,
            "collection": REFERENCE_COLLECTION,
        }
    )


if __name__ == "__main__":
    main()
