"""Las operaciones de modelado 3D, encadenables de a una.

Cada funcion hace UNA cosa y devuelve el veredicto junto al archivo — la misma
regla que ya rige el carril de impresion: una malla que no cierra no es una
pieza, y decir "listo" sobre eso es el peor falso positivo, el que da confianza.

Son atomicas a proposito. La persona que usa la pantalla aprieta un boton por
vez; un agente que llega por MCP encadena varias y necesita medir entre medio.
Las dos cosas se sirven con las mismas piezas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.services import blender_service, fit, mesh_engine_service, silhouette
from app.services.turnaround import Box, character_view_boxes, ink_bounds, open_sheet

# Como se llama cada vista en la escena. El orden es el de una hoja de
# turnaround estandar, de izquierda a derecha.
VIEW_ORDER = ("front", "side", "back", "side_left")

AUDIT_SCRIPT = "audit_mesh.py"
REFERENCE_SCRIPT = "build_reference_scene.py"
REMESH_SCRIPT = "remesh.py"
SILHOUETTE_SCRIPT = "render_silhouettes.py"

# Cuanto aire se deja alrededor de la malla al renderizar su silueta. Sin
# margen, un pixel de antialias en el borde se pierde contra el marco y el
# ancho medido sale corto.
MARGEN_ENCUADRE = 1.25


@dataclass(frozen=True, slots=True)
class DetectedView:
    name: str
    image: Path
    ink: Box

    def as_payload(self) -> dict[str, Any]:
        return {"image": str(self.image), "inkBox": list(self.ink.as_tuple())}


def audit_mesh(settings: Settings, mesh_path: Path) -> dict[str, Any]:
    """Mide una malla sin tocarla."""
    return blender_service.run_script(settings, AUDIT_SCRIPT, {"mesh": str(mesh_path)})


def split_views(
    sheet_path: Path,
    out_dir: Path,
    *,
    names: tuple[str, ...] = VIEW_ORDER,
) -> list[DetectedView]:
    """Escribe una imagen por vista y devuelve cada una con su caja de tinta.

    La caja se recalcula sobre el RECORTE y no se hereda de la hoja: son
    sistemas de coordenadas distintos, y pasar la de la hoja al script de
    Blender lo haria escalar por un margen que ya no existe.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = open_sheet(sheet_path)
    detectadas: list[DetectedView] = []
    for nombre, caja in zip(names, character_view_boxes(rgb)):
        recorte = rgb.crop(caja.as_tuple())
        destino = out_dir / f"{nombre}.png"
        recorte.save(destino)
        detectadas.append(DetectedView(name=nombre, image=destino, ink=ink_bounds(recorte)))
    return detectadas


def remesh(
    settings: Settings,
    mesh_path: Path,
    output: Path,
    *,
    voxel_meters: float = 0.01,
) -> dict[str, Any]:
    """Rehace la topologia por voxeles y devuelve el antes y el despues.

    Las dos auditorias viajan juntas a proposito: un remesh gana topologia
    uniforme y pierde detalle, y cuanto perdio solo se ve comparando.
    """
    return blender_service.run_script(
        settings,
        REMESH_SCRIPT,
        {"mesh": str(mesh_path), "output": str(output), "voxelMeters": voxel_meters},
    )


def generate_mesh(
    settings: Settings,
    engine: str,
    image_path: Path,
    output: Path,
    *,
    steps: int = 50,
    guidance: float = 7.0,
    face_limit: int = 0,
) -> dict[str, Any]:
    """Genera una malla desde una imagen con un motor generativo local.

    Existe para que el banco tenga candidatas que no salgan de primitivas. Lo
    que devuelve NO esta aprobado por haber salido: `audited` viaja en false a
    proposito, y el paso siguiente es `score_fit`. Un generador puede devolver
    una superficie preciosa con doscientas islas sueltas.
    """
    return mesh_engine_service.generate(
        settings,
        engine,
        {
            "image": str(image_path),
            "output": str(output),
            "steps": steps,
            "guidance": guidance,
            "faceLimit": face_limit,
        },
    )


class UnknownScaleViewError(ValueError):
    """La vista elegida para fijar la escala no existe entre los recortes."""


def _vista_mas_alta(vistas: list[DetectedView]) -> DetectedView:
    return max(vistas, key=lambda vista: vista.ink.height)


def score_fit(
    settings: Settings,
    mesh_path: Path,
    views_dir: Path,
    render_dir: Path,
    *,
    height_meters: float,
    scale_view: str | None = None,
    resolution: int = 512,
) -> dict[str, Any]:
    """Cuanto se parece una malla al dibujo, y de que tipo es la diferencia.

    Es la balanza del banco: la misma medida para una malla generada por un
    modelo, esculpida a mano o armada con primitivas, asi que comparar motores
    deja de ser cuestion de opinion.

    `scale_view` es el nombre de la vista cuya altura real se conoce, y esa
    UNICA escala vale para todos los recortes de la hoja. Es explicito y no
    inferido porque la alternativa —escalar cada vista por su propia altura de
    tinta— es el error medido que hizo imposible calzar una gorra: de frente el
    punto mas bajo era la banda y de perfil la punta de la visera, asi que las
    dos vistas quedaban a escalas distintas y ningun modelo podia calzar ambas.

    Devuelve la auditoria junto al calce: una malla puede calzar la silueta y
    ser igual inservible por estar rota, y separar las dos preguntas invita a
    responder solo la comoda.
    """
    vistas = views_from_dir(views_dir)
    if not vistas:
        raise UnknownScaleViewError(f"no hay recortes de vista en {views_dir}")

    if scale_view is None:
        referencia_escala = _vista_mas_alta(vistas)
    else:
        elegida = next((vista for vista in vistas if vista.name == scale_view), None)
        if elegida is None:
            disponibles = ", ".join(vista.name for vista in vistas)
            raise UnknownScaleViewError(f"'{scale_view}' no esta entre las vistas: {disponibles}")
        referencia_escala = elegida

    auditoria = audit_mesh(settings, mesh_path)
    if auditoria.get("error"):
        return {"audit": auditoria, "fit": None}

    ancho_encuadre = max(auditoria["dims"]) * MARGEN_ENCUADRE
    render = blender_service.run_script(
        settings,
        SILHOUETTE_SCRIPT,
        {
            "mesh": str(mesh_path),
            "outputDir": str(render_dir),
            "views": [vista.name for vista in vistas],
            "viewWidthMeters": ancho_encuadre,
            "resolution": resolution,
        },
    )
    if render.get("error"):
        return {"audit": auditoria, "fit": None, "error": render["error"]}

    metros_por_pixel_dibujo = fit.metros_por_pixel_de(referencia_escala.image, height_meters)
    calce = fit.comparar(
        {nombre: Path(ruta) for nombre, ruta in render["silhouettes"].items()},
        {vista.name: vista.image for vista in vistas},
        metros_por_pixel_modelo=render["metersPerPixel"],
        metros_por_pixel_dibujo=metros_por_pixel_dibujo,
    )
    return {
        "audit": auditoria,
        "fit": {
            "scaleView": referencia_escala.name,
            "scaleViewHeightMeters": height_meters,
            "metersPerPixelModel": render["metersPerPixel"],
            "metersPerPixelSheet": metros_por_pixel_dibujo,
            "average": calce.promedio,
            "worstView": calce.peor_vista,
            "views": [
                {
                    "view": ajuste.vista,
                    "anchored": ajuste.anclado,
                    "best": ajuste.mejor,
                    "gainFromMoving": ajuste.gana_moviendo,
                    "offsetCm": list(ajuste.corrimiento_cm),
                    "blame": ajuste.culpa,
                    "widthCm": [ajuste.ancho.modelo_cm, ajuste.ancho.dibujo_cm],
                    "heightCm": [ajuste.alto.modelo_cm, ajuste.alto.dibujo_cm],
                }
                for ajuste in calce.ajustes
            ],
        },
    }


def compare_meshes(
    settings: Settings,
    meshes: dict[str, Path],
    views_dir: Path,
    render_root: Path,
    *,
    height_meters: float,
    scale_view: str | None = None,
    resolution: int = 512,
) -> dict[str, Any]:
    """Mide varias mallas contra la misma hoja y las ordena.

    Es el banco: dos formas de llegar a la misma pieza —un modelo generativo,
    un remallado, un blockout a mano— dejan de compararse por impresion y pasan
    a compararse por el mismo numero. Cada candidata trae su auditoria, asi que
    una que calce lindo pero este rota no gana por calzar.

    Una candidata que falla NO tumba el banco: se reporta con su error y las
    demas siguen. Un motor que no arranca es justamente lo que hay que ver en
    la tabla, no un stack trace que la deja vacia.
    """
    resultados: list[dict[str, Any]] = []
    for nombre, ruta in meshes.items():
        try:
            medida = score_fit(
                settings,
                ruta,
                views_dir,
                render_root / nombre,
                height_meters=height_meters,
                scale_view=scale_view,
                resolution=resolution,
            )
        except Exception as exc:  # noqa: BLE001 - el fallo de una candidata es un dato
            resultados.append({"name": nombre, "mesh": str(ruta), "error": str(exc)})
            continue
        resultados.append({"name": nombre, "mesh": str(ruta), **medida})

    medidas = [r for r in resultados if r.get("fit")]
    medidas.sort(key=lambda r: r["fit"]["average"], reverse=True)
    fallidas = [r for r in resultados if not r.get("fit")]
    return {
        "ranked": medidas + fallidas,
        # El ganador solo existe si ADEMAS de calzar mejor la malla esta sana:
        # premiar una silueta linda sobre una malla rota es el falso positivo
        # que este banco existe para no cometer.
        "winner": next((r["name"] for r in medidas if r["audit"].get("ok")), None),
        "measured": len(medidas),
        "failed": [r["name"] for r in fallidas],
    }


def views_from_dir(views_dir: Path, *, names: tuple[str, ...] = VIEW_ORDER) -> list[DetectedView]:
    """Rearma las vistas leyendo recortes ya escritos, midiendo cada uno.

    La caja de tinta se vuelve a medir en vez de recibirse: si viniera del
    cliente, cualquiera podria fijar una escala arbitraria en la escena.
    """
    vistas: list[DetectedView] = []
    for nombre in names:
        recorte = views_dir / f"{nombre}.png"
        if not recorte.exists():
            continue
        with Image.open(recorte) as imagen:
            vistas.append(DetectedView(name=nombre, image=recorte, ink=ink_bounds(imagen.convert("RGB"))))
    return vistas


def _cubre_casi_toda(caja: Box, sheet_path: Path, *, umbral: float = 0.9) -> bool:
    with Image.open(sheet_path) as hoja:
        return caja.width >= hoja.width * umbral


def sheet_warnings(
    sheet_path: Path,
    *,
    expected_views: int = len(VIEW_ORDER),
    names: tuple[str, ...] = VIEW_ORDER,
) -> list[str]:
    """Lo que la hoja tiene de raro, dicho antes de que arruine la escena.

    Contar es la unica senal confiable. Medido sobre una hoja real: dos vistas
    dibujadas tan juntas que se SUPERPONEN no dejan ninguna columna con poca
    tinta entre medio, asi que no hay corte vertical que las separe ni ancho
    sospechoso que las delate — pero el conteo baja de cuatro a tres y eso se
    ve siempre.

    Se pregunta aparte de partir: partir devuelve vistas, esto devuelve dudas,
    y mezclarlas obligaria a revisar el resultado para saber si hubo problema.
    """
    cajas = character_view_boxes(open_sheet(sheet_path))

    # Sobran paneles para los nombres que hay: `zip` los descartaria en
    # silencio y en disco quedarian menos recortes de los que la hoja tiene.
    if len(cajas) > len(names):
        return [
            f"la hoja tiene {len(cajas)} vistas y este carril sabe nombrar "
            f"{len(names)}: las que sobran se descartan. Recorta la hoja a "
            f"{len(names)} vistas y volve a subirla."
        ]
    if len(cajas) == expected_views:
        return []
    # Un solo panel que ocupa casi toda la hoja no es "una vista": es que no
    # hubo fondo blanco por donde cortar. Culpar a vistas superpuestas manda a
    # arreglar lo que no esta roto.
    if len(cajas) == 1 and _cubre_casi_toda(cajas[0], sheet_path):
        return [
            "no se encontraron columnas de fondo entre las vistas: la hoja "
            "salio como una sola imagen. Necesita fondo blanco entre vista y "
            "vista."
        ]
    return [
        f"se detectaron {len(cajas)} vistas y se esperaban {expected_views}. "
        "Dos vistas dibujadas superpuestas no se pueden separar solas: "
        "separalas en la hoja y volve a subirla."
    ]


def build_reference_scene(
    settings: Settings,
    views: list[DetectedView],
    output: Path,
    *,
    height_meters: float = 1.70,
) -> dict[str, Any]:
    """La escena de Blender lista para modelar encima."""
    return blender_service.run_script(
        settings,
        REFERENCE_SCRIPT,
        {
            "views": {vista.name: vista.as_payload() for vista in views},
            "heightMeters": height_meters,
            "output": str(output),
        },
    )


def measure_proportions(views_dir: Path, *, height_meters: float = 1.70) -> dict[str, Any]:
    """Las proporciones del personaje, leidas de las vistas ya partidas.

    Es lo que un modelador mira antes de empezar: cuantas cabezas mide y cuanto
    de ancho tiene a cada altura. Lo que las dos vistas NO ubican en el mismo
    lugar se marca dudoso en vez de promediarse en silencio.
    """
    frente, lado = views_dir / "front.png", views_dir / "side.png"
    if not frente.exists() or not lado.exists():
        raise FileNotFoundError("hacen falta las vistas 'front' y 'side' para medir")

    medidas = silhouette.medir(frente, lado, height_meters)
    return {
        "heightMeters": medidas.altura_m,
        "headMeters": medidas.cabeza_m,
        "headsTall": medidas.cabezas_de_alto,
        "landmarks": [
            {
                "name": altura.nombre,
                "z": altura.z,
                "front": altura.frente,
                "side": altura.lado,
                "agrees": altura.acuerdo,
                "disagreementCm": altura.desacuerdo_cm,
            }
            for altura in medidas.alturas
        ],
        "uncertain": list(medidas.dudosas),
        "widths": [
            {
                "z": round(z, 3),
                "frontCm": round(silhouette.ancho_en(medidas.bandas_frente, z) * 100, 1),
                "sideCm": round(silhouette.ancho_en(medidas.bandas_lado, z) * 100, 1),
            }
            # Cada 5 cm: mas fino que eso es ruido de trazo y llena la pantalla.
            for z in [i * 0.05 for i in range(int(height_meters / 0.05), -1, -1)]
        ],
    }


class UnknownViewNameError(ValueError):
    """Un nombre de vista que el carril no sabe colocar en la escena."""


def rename_views(views_dir: Path, names: list[str]) -> list[str]:
    """Reasigna que vista es cada recorte, de izquierda a derecha.

    Nombrar por posicion es una CONVENCION, no una deduccion: mirando los
    pixeles no hay forma de saber si el tercer panel es la espalda o un tres
    cuartos. Cuando la hoja viene en otro orden, la escena sale con el dibujo
    equivocado en cada plano y nada lo delata — por eso se puede corregir.
    """
    conocidos = set(VIEW_ORDER)
    desconocidos = [n for n in names if n not in conocidos]
    if desconocidos:
        raise UnknownViewNameError(
            f"nombres que el carril no sabe colocar: {', '.join(desconocidos)}. "
            f"Validos: {', '.join(VIEW_ORDER)}."
        )
    if len(set(names)) != len(names):
        raise UnknownViewNameError("hay nombres repetidos: cada vista va una sola vez")

    actuales = [nombre for nombre in VIEW_ORDER if (views_dir / f"{nombre}.png").exists()]
    if len(names) != len(actuales):
        raise UnknownViewNameError(
            f"la hoja tiene {len(actuales)} vistas y se pasaron {len(names)} nombres"
        )

    # Se pasa por nombres temporales: intercambiar frente y espalda directamente
    # pisaria un archivo con el otro.
    for indice, viejo in enumerate(actuales):
        (views_dir / f"{viejo}.png").rename(views_dir / f"_{indice}.tmp")
    for indice, nuevo in enumerate(names):
        (views_dir / f"_{indice}.tmp").rename(views_dir / f"{nuevo}.png")
    return names
