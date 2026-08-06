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
    "wan-video": "los modelos de generacion de video",
    "magpie": "el overlay de tiempo real",
    "shap-e": "el modelo de generacion 3D",
    "kokoro": "el modelo de voz",
    "voice-conversion": "el modelo de conversion de voz",
    "translation": "el par de idiomas para traducir",
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
    frase = f"Falta {que}. Se baja desde la app, con el boton de descargar."
    return f"{frase} {detail}".strip() if detail else frase
