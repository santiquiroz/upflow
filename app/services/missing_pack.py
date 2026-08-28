"""Que decirle al usuario cuando falta un paquete, sin mandarlo a una terminal.

El 2026-08-05 un barrido encontro 36 mensajes del backend que le dictaban al
usuario un comando de PowerShell. La app sabe correr esos scripts sola desde
hace varias versiones — el boton existia, pero solo en una pantalla, y todos los
demas caminos seguian escupiendo el comando.

Aca vive la unica descripcion de cada paquete. Los mensajes se arman con ella,
asi que agregar un paquete nuevo sin describirlo rompe una prueba en vez de
llegar al usuario como un comando.
"""

from __future__ import annotations

from dataclasses import dataclass

# Que ES cada paquete, dicho como lo diria alguien que no sabe que existe un
# script. Nada de nombres de archivo ni de rutas: eso es detalle nuestro.
PACK_LABELS: dict[str, str] = {
    "realesrgan": "el motor de escalado",
    "realesrgan-onnx": "los modelos de escalado en formato ONNX",
    "rife": "el motor de interpolacion RIFE",
    "gmfss": "los modelos de interpolacion GMFSS",
    "deepfilternet": "el limpiador de ruido DeepFilterNet",
    "apollo": "el modelo de restauracion Apollo",
    "audiosr": "los modelos de restauracion AudioSR",
    "ffmpeg": "las herramientas de video FFmpeg",
    "mobilesam": "el modelo de seleccion por toque",
    "migan": "el modelo de borrado rapido",
    "sdcpp": "el motor de generacion por Vulkan",
    "openscad": "OpenSCAD, que convierte el codigo en la pieza",
    # No se baja con un script: pesa mas que toda la app y quien modela en 3D
    # ya lo tiene. Lo instala el usuario y Upflow lo encuentra solo.
    "blender": "Blender 4.2 o superior, que hace el trabajo de mallas",
    "wan-video": "los modelos de generacion de video",
    "magpie": "el overlay de tiempo real",
    "ceca": "las descargas de YouTube",
    "openvoice": "cambiar una voz por otra",
    "shap-e": "el modelo de generacion 3D",
    "shap-e-img2img": "el modelo de foto a 3D",
    "karaoke": "el modelo de separacion de voz e instrumental",
    "kokoro": "el modelo de voz",
    "voice-conversion": "el modelo de conversion de voz",
    "translation": "el par de idiomas para traducir",
}


# Los que NO baja la app. Blender pesa mas que Upflow entero y no se vendoriza,
# asi que el mensaje tiene que mandar al sitio oficial y no a un boton que no
# existe. OpenSCAD NO va aca aunque tambien sea GPL y externo: la app SI lo
# baja (`download-openscad.ps1` en PACK_SCRIPTS), y decirle al usuario que se
# lo instale el mismo lo manda a hacer a mano algo que ya tiene a un click.
USER_SUPPLIED_PACKS: frozenset[str] = frozenset({"blender"})

_HOW_TO_GET = {
    False: "Se baja desde la app, con el boton de descargar.",
    True: "Lo instalas vos desde su sitio oficial; Upflow lo encuentra solo despues.",
}


class MissingPack(RuntimeError):
    """Falta un paquete, y la app lo puede bajar sola.

    Lleva `pack` para que la capa HTTP lo mande al frontend: sin eso la pantalla
    solo tiene una frase y no sabe que boton ofrecer.
    """

    def __init__(self, pack: str, *, variant: str | None = None, detail: str = "") -> None:
        self.pack = pack
        # Cual del paquete, cuando el paquete es una familia (un modelo por par
        # de idiomas). Sin esto el boton no sabria cual de todos bajar.
        self.variant = variant
        super().__init__(missing_pack_message(pack, variant=variant, detail=detail))


@dataclass(frozen=True, slots=True)
class UnknownPackLabel(KeyError):
    pack: str


def label_for(pack: str) -> str:
    try:
        return PACK_LABELS[pack]
    except KeyError as exc:
        raise UnknownPackLabel(pack) from exc


def missing_pack_message(
    pack: str, *, variant: str | None = None, detail: str = ""
) -> str:
    """La frase que lee el usuario. Termina donde puede actuar, no en un comando."""
    que = f"{label_for(pack)} ({variant})" if variant else label_for(pack)
    frase = f"Falta {que}. {_HOW_TO_GET[pack in USER_SUPPLIED_PACKS]}"
    return f"{frase} {detail}".strip() if detail else frase
