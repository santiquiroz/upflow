# SPDX-License-Identifier: GPL-2.0-or-later
#
# Parte de Upflow. Corre DENTRO de Blender y usa `bpy`. Ver LICENSE en esta carpeta.
"""Saca la silueta de una malla desde las vistas de un turnaround.

Entrada: {"mesh": "<ruta>", "outputDir": "<dir>", "views": ["front","side"],
          "viewWidthMeters": 0.6, "resolution": 512}
Salida:  una PNG por vista, y CUANTO MIDE UN PIXEL en metros.

El pixel metrico es el punto de todo esto. Una silueta suelta no se puede
comparar con nada; una silueta que sabe cuantos milimetros mide cada pixel se
compara contra el dibujo sin depender de que las dos imagenes tengan el mismo
tamano ni el mismo encuadre.

La camara es ORTOGRAFICA a proposito: una perspectiva mete escorzo, y entonces
la silueta deja de ser la proyeccion del objeto y pasa a ser la proyeccion del
objeto MAS la distancia a la que se puso la camara. Eso ya no se puede comparar
contra una hoja de model sheet, que esta dibujada sin fuga.

ORIENTACION: se acepta `upAxis` porque cada motor entrega en SU marco canonico.
Las mallas de `build_reference_scene` vienen Z-arriba; las de un generador
suelen venir Y-arriba, y renderizar una Y-arriba como si fuera Z-arriba la
muestra acostada — entonces el banco mide la rotacion en vez del parecido.
Lo que igual queda como SUPUESTO es hacia donde mira: se toma +Y como frente, y
si no lo es las siluetas salen bien renderizadas y mal rotuladas. Por eso el
resultado devuelve `upAxis` y `assumedFrontAxis` en vez de callarselos.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from _common import emit, fail, import_mesh, payload, reset_scene  # noqa: E402

MEDIA_VUELTA = 3.14159265358979
CUARTO_DE_VUELTA = MEDIA_VUELTA / 2

# Donde se para la camara para cada vista, en (posicion, rotacion). La distancia
# no cambia nada con camara ortografica: solo tiene que dejar la malla delante.
DISTANCIA_M = 10.0
CAMARAS = {
    "front": ((0, -DISTANCIA_M, 0), (CUARTO_DE_VUELTA, 0, 0)),
    "back": ((0, DISTANCIA_M, 0), (CUARTO_DE_VUELTA, 0, MEDIA_VUELTA)),
    "side": ((-DISTANCIA_M, 0, 0), (CUARTO_DE_VUELTA, 0, -CUARTO_DE_VUELTA)),
    "side_left": ((DISTANCIA_M, 0, 0), (CUARTO_DE_VUELTA, 0, CUARTO_DE_VUELTA)),
}

RESOLUCION_POR_DEFECTO = 512
# Debajo de esto un rasgo de 1 cm cae en menos de un pixel y la comparacion
# mide el antialias en vez de la forma.
RESOLUCION_MINIMA = 128

# Como llega la malla, para poder enderezarla antes de mirarla. Cada motor
# entrega en SU marco canonico y el de Blender no es universal: una malla
# Y-arriba renderizada como si fuera Z-arriba sale acostada, y entonces el
# banco mide la rotacion en vez del parecido. La conversion es un giro de
# +90 grados en X, que es la que usan glTF y la mayoria de los generadores.
CUARTO_EN_X = (CUARTO_DE_VUELTA, 0.0, 0.0)
ORIENTACIONES = {
    "z_up": (0.0, 0.0, 0.0),
    "y_up": CUARTO_EN_X,
}


def enderezar(objeto: bpy.types.Object, orientacion: str) -> None:
    """Lleva la malla al marco Z-arriba que asume el resto del carril.

    Se aplica la rotacion a la geometria y no se deja en el objeto: los pasos
    que siguen —medir, comparar— leen coordenadas de mundo, y una rotacion que
    vive solo en la transformada del objeto se pierde al exportar.
    """
    if orientacion == "z_up":
        return
    # El modo de rotacion se fija ANTES de asignar los angulos. El importador
    # de glTF deja los objetos en QUATERNION, y con ese modo asignar
    # `rotation_euler` NO HACE NADA: Blender lee el cuaternion. Medido el
    # 2026-08-28 — `transform_apply` devolvia {'FINISHED'} sobre una rotacion
    # de cero y la malla salia intacta, o sea el peor fallo posible: el que
    # reporta exito.
    objeto.rotation_mode = "XYZ"
    objeto.rotation_euler = ORIENTACIONES[orientacion]
    bpy.context.view_layer.objects.active = objeto
    bpy.ops.object.select_all(action="DESELECT")
    objeto.select_set(True)
    bpy.ops.object.transform_apply(rotation=True)
    bpy.context.view_layer.update()


def centro_de(objeto: bpy.types.Object) -> tuple[float, float, float]:
    """El centro de la caja envolvente, en coordenadas de mundo.

    Se apunta la camara ahi y no al origen: una malla generada puede venir con
    el origen en cualquier lado, y entonces la silueta sale recortada por el
    borde del cuadro sin que nada avise.
    """
    esquinas = [objeto.matrix_world @ Vector(esquina) for esquina in objeto.bound_box]
    ejes = list(zip(*[(punto.x, punto.y, punto.z) for punto in esquinas]))
    return tuple((min(eje) + max(eje)) / 2 for eje in ejes)


def preparar_render(resolucion: int) -> None:
    """Silueta pura: sin luces, sin materiales, sin fondo.

    Workbench con fondo transparente deja la silueta en el canal alfa. Es el
    render mas barato que existe y el unico que no depende de como este
    iluminada o texturizada la malla, que es justo lo que NO queremos medir.
    """
    escena = bpy.context.scene
    escena.render.engine = "BLENDER_WORKBENCH"
    escena.render.resolution_x = resolucion
    escena.render.resolution_y = resolucion
    escena.render.resolution_percentage = 100
    escena.render.film_transparent = True
    escena.render.image_settings.file_format = "PNG"
    escena.render.image_settings.color_mode = "RGBA"
    escena.view_settings.view_transform = "Standard"


def camara_ortografica(ancho_m: float) -> bpy.types.Object:
    datos = bpy.data.cameras.new("silueta")
    datos.type = "ORTHO"
    datos.ortho_scale = ancho_m
    camara = bpy.data.objects.new("silueta", datos)
    bpy.context.scene.collection.objects.link(camara)
    bpy.context.scene.camera = camara
    return camara


def renderizar(camara: bpy.types.Object, vista: str, centro, destino: str) -> None:
    posicion, rotacion = CAMARAS[vista]
    camara.location = (
        posicion[0] + centro[0],
        posicion[1] + centro[1],
        posicion[2] + centro[2],
    )
    camara.rotation_euler = rotacion
    bpy.context.scene.render.filepath = destino
    bpy.ops.render.render(write_still=True)


def main() -> None:
    datos = payload()
    entrada = datos.get("mesh", "")
    carpeta = datos.get("outputDir", "")
    vistas = datos.get("views") or ["front", "side"]
    ancho_m = float(datos.get("viewWidthMeters") or 0)
    resolucion = int(datos.get("resolution") or RESOLUCION_POR_DEFECTO)
    orientacion = datos.get("upAxis") or "z_up"

    if not entrada or not carpeta:
        fail("faltan 'mesh' y 'outputDir'")
    if not os.path.exists(entrada):
        fail(f"no existe: {entrada}")
    if ancho_m <= 0:
        fail("'viewWidthMeters' tiene que ser mayor que cero: es lo que le da escala al pixel")
    if resolucion < RESOLUCION_MINIMA:
        fail(f"'resolution' minima {RESOLUCION_MINIMA}: mas chico mide el antialias, no la forma")
    if orientacion not in ORIENTACIONES:
        fail(f"'upAxis' desconocido: {orientacion}. Se conocen: {', '.join(ORIENTACIONES)}")
    desconocidas = [vista for vista in vistas if vista not in CAMARAS]
    if desconocidas:
        fail(f"vistas que no se saben renderizar: {', '.join(desconocidas)}")

    reset_scene()
    objeto = import_mesh(entrada)
    enderezar(objeto, orientacion)
    centro = centro_de(objeto)

    preparar_render(resolucion)
    camara = camara_ortografica(ancho_m)
    os.makedirs(carpeta, exist_ok=True)

    salidas = {}
    for vista in vistas:
        destino = os.path.join(carpeta, f"silueta_{vista}.png")
        renderizar(camara, vista, centro, destino)
        salidas[vista] = destino

    emit({
        "silhouettes": salidas,
        "metersPerPixel": ancho_m / resolucion,
        "viewWidthMeters": ancho_m,
        "resolution": resolucion,
        "dims": [round(float(valor), 6) for valor in objeto.dimensions],
        # Lo que se ASUMIO, dicho para que se pueda desmentir: una malla mal
        # rotulada sale bien renderizada y mal medida, sin que nada avise.
        "upAxis": orientacion,
        "assumedFrontAxis": "+Y",
    })


if __name__ == "__main__":
    main()
