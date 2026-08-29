from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import uuid4

from app.config import Settings, resolve_against_project_root
from app.services.capabilities import CATALOG, PathRequirement
from app.services.engines.separation_models import SEPARATION_MODELS
from app.services.install_queue_base import SingleWorkerJobQueue
from app.services.process_runner import run_guarded_process

logger = logging.getLogger(__name__)

# Los paquetes vendorizados se bajaban a mano corriendo estos scripts. La app
# ahora los corre por el usuario, y la regla que mantiene esto acotado es REUSAR
# el script, no reimplementar la descarga: el riesgo de regresion esta en tocar
# codigo que funciona desde muchas versiones, no en el boton.
PACK_SCRIPTS: dict[str, str] = {
    "realesrgan": "download-realesrgan.ps1",
    "realesrgan-onnx": "download-realesrgan-onnx.ps1",
    "rife": "download-rife.ps1",
    "gmfss": "download-gmfss-onnx.ps1",
    "deepfilternet": "download-deepfilternet.ps1",
    "apollo": "download-apollo.ps1",
    "audiosr": "download-audiosr-onnx.ps1",
    "ffmpeg": "download-ffmpeg.ps1",
    "mobilesam": "download-mobilesam.ps1",
    "migan": "download-migan.ps1",
    "sdcpp": "download-sdcpp.ps1",
    # GPL-2.0: se baja aparte y corre como proceso separado, nunca enlazado.
    "openscad": "download-openscad.ps1",
    # El pack de video trae ADEMAS el motor Vulkan: download-wan-video.ps1 baja
    # los pesos y el binario ya viene con el pack sdcpp.
    "wan-video": "download-wan-video.ps1",
    "magpie": "download-magpie.ps1",
    # Motor JS + acuñador de PO Token: las dos cosas que YouTube empezo a exigir
    # para entregar el medio. Un solo pack porque una sin la otra no sirve.
    "ceca": "download-ceca.ps1",
    # Conversion de timbre con OpenVoice (pesos MIT, a diferencia de casi todos
    # los clonadores de voz). Convive con el pack de SpeechT5 en vez de
    # reemplazarlo: quien ya lo tenia bajado no se queda sin la funcion.
    "openvoice": "download-openvoice.ps1",
    "shap-e": "download-shap-e.ps1",
    # Foto a 3D. Es OTRO repo HF que el de texto (openai/shap-e-img2img):
    # comparte el renderer pero cambia el prior y el encoder de entrada.
    "shap-e-img2img": "download-shap-e-img2img.ps1",
    # Separacion voz/instrumental. Pack-familia: un .onnx por modelo del
    # catalogo (separation_models.py) y el script recibe cual por parametro.
    "karaoke": "download-karaoke.ps1",
    # Transcripcion por stem a MIDI/MusicXML/tab (F3a): Basic Pitch ONNX, un
    # solo archivo, sin variante.
    "music-transcription": "download-music-transcription.ps1",
    # El nombre sigue al directorio vendorizado (vendor/kokoro), no al script:
    # es la convencion que ya usan gmfss y audiosr.
    "kokoro": "download-kokoro-tts.ps1",
    "voice-conversion": "download-voice-conversion.ps1",
    "translation": "download-translation.ps1",
}

# Packs que no son UN modelo sino una familia. La traduccion es uno por par de
# idiomas, que es como OPUS-MT los publica; karaoke es un modelo de separacion
# por variante del catalogo. El script recibe cual por parametro.
PACK_PARAMETERS: dict[str, str] = {
    "translation": "-Pair",
    "karaoke": "-Model",
    # audiosr no es una familia: es EL mismo modelo en dos precisiones. fp16 pesa
    # la mitad (2.51 -> 1.26 GiB) y corre 9% mas rapido con la salida a 59.4 dB de
    # la fp32, pero solo en GPU: el EP de CPU tiene muchos menos kernels fp16.
    "audiosr": "-Precision",
}

# La variante llega desde una peticion HTTP y termina en una linea de comandos:
# se acepta la forma exacta que cada script declara, y nada mas.
VARIANT_PATTERNS: dict[str, re.Pattern[str]] = {
    "translation": re.compile(r"^[a-z]{2,3}-[a-z]{2,3}$"),
    "karaoke": re.compile(r"^[a-z0-9_]{1,32}$"),
    "audiosr": re.compile(r"^fp(16|32)$"),
}

# Catalogos CERRADOS por pack: la variante tiene que existir, no solo tener la
# forma. La traduccion queda fuera a proposito: es una familia abierta (OPUS-MT
# publica un modelo por par de idiomas, la URL se arma con el par).
VARIANT_CATALOGS: dict[str, frozenset[str]] = {
    "karaoke": frozenset(SEPARATION_MODELS),
    "audiosr": frozenset({"fp16", "fp32"}),
}

# Los scripts descargan cientos de MB desde GitHub releases. El techo es un
# limite de seguridad contra un proceso colgado, no una expectativa de duracion.
PROVISION_TIMEOUT_SECONDS = 45 * 60

SCRIPTS_DIRNAME = "scripts"


class ProvisionStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


@dataclass(slots=True, kw_only=True)
class ProvisionJob:
    id: str
    pack: str
    # Cual del pack, cuando el pack es una familia (un modelo por par de idiomas).
    variant: str | None = None
    status: ProvisionStatus = ProvisionStatus.queued
    error: str | None = None


class UnknownPackError(ValueError):
    pass


Platform = Literal["win32", "other"]


def packs_required_by_catalog() -> frozenset[str]:
    """Los packs del catalogo que la app TIENE que saber bajar.

    Los de `USER_SUPPLIED_PACKS` quedan afuera a proposito: son programas
    completos que instala el usuario (hoy Blender), y exigirles un script de
    descarga obligaria a inventar uno que no queremos tener.
    """
    from app.services.missing_pack import USER_SUPPLIED_PACKS

    return frozenset(
        requirement.pack
        for capability in CATALOG
        for requirement in capability.requirements
        if isinstance(requirement, PathRequirement)
        and requirement.pack not in USER_SUPPLIED_PACKS
    )


def script_for(pack: str) -> str:
    script = PACK_SCRIPTS.get(pack)
    if script is None:
        raise UnknownPackError(f"No hay script de descarga para el paquete {pack!r}.")
    return script


def script_path(pack: str) -> Path:
    return resolve_against_project_root(SCRIPTS_DIRNAME) / script_for(pack)


def provisioning_supported(platform: str = sys.platform) -> bool:
    """Los scripts de descarga son PowerShell.

    En otras plataformas la capacidad queda en needs_setup con un motivo
    explicito, en vez de fallar de forma rara a mitad de una descarga.
    """
    return platform == "win32"


def _powershell_safe_env() -> dict[str, str]:
    """Entorno para los scripts .ps1 SIN PSModulePath heredado.

    Pasado real (2026-08-09): el server relanzado desde pwsh 7 hereda un
    PSModulePath ajeno, y el powershell 5.1 hijo deja de resolver cmdlets de
    modulo (Get-FileHash → CommandNotFound en plena descarga del pack de
    karaoke). Sin la variable, 5.1 reconstruye su default completo.
    """
    env = dict(os.environ)
    env.pop("PSModulePath", None)
    return env


def default_variant(pack: str, default_device: str) -> str | None:
    """Que variante bajar cuando el usuario aprieta el boton y no elige nada.

    Solo audiosr tiene una respuesta que dependa de la maquina: fp16 pesa la
    mitad y corre un 9% mas rapido con la salida a 59.4 dB de la fp32
    (inaudible), pero el EP de CPU tiene muchos menos kernels fp16 y los que
    hay suelen ser mas lentos. Asi que la precision sigue al dispositivo por
    defecto, que es el que va a correr el modelo en la practica.
    """
    if pack != "audiosr":
        return None
    return "fp32" if default_device.strip().lower().startswith("cpu") else "fp16"


def build_command(pack: str, variant: str | None = None) -> list[str]:
    comando = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path(pack)),
    ]
    if variant is None:
        return comando
    return comando + [_parameter_for(pack), _checked_variant(pack, variant)]


def _parameter_for(pack: str) -> str:
    try:
        return PACK_PARAMETERS[pack]
    except KeyError as exc:
        raise ValueError(
            f"El paquete '{pack}' no acepta variantes."
        ) from exc


def _checked_variant(pack: str, variant: str) -> str:
    pattern = VARIANT_PATTERNS.get(pack)
    if pattern is None:
        raise ValueError(f"El paquete '{pack}' no declara forma de variante.")
    if not pattern.match(variant):
        raise ValueError(
            f"La variante no tiene la forma que el paquete '{pack}' espera. Recibido: {variant!r}"
        )
    catalogo = VARIANT_CATALOGS.get(pack)
    if catalogo is not None and variant not in catalogo:
        raise ValueError(
            f"La variante {variant!r} no existe en el catalogo del paquete "
            f"'{pack}'. Conocidas: {', '.join(sorted(catalogo))}"
        )
    return variant


def _tail(raw: bytes, limit: int = 600) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    return text[-limit:] if len(text) > limit else text


class PackProvisioner(SingleWorkerJobQueue[ProvisionJob]):
    _error_status = ProvisionStatus.error

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings

    async def provision(self, pack: str, variant: str | None = None) -> str:
        # Se valida ANTES de encolar: un pack desconocido, o una variante con
        # forma rara, tienen que fallar la request y no aparecer como un job que
        # despues se muere solo.
        script_for(pack)
        build_command(pack, variant)
        return await self._enqueue(ProvisionJob(id=uuid4().hex, pack=pack, variant=variant))

    async def _run(self, job: ProvisionJob) -> None:
        # Un paquete a la vez: dos descargas del mismo destino se pisarian los
        # archivos a medio escribir.
        async with self._lock_for(job.pack):
            try:
                await self._execute(job)
            except Exception as exc:  # noqa: BLE001 - el job reporta, no propaga
                logger.warning("pack provisioning failed", extra={"pack": job.pack})
                self._fail_job(job, exc)

    async def _execute(self, job: ProvisionJob) -> None:
        if not provisioning_supported():
            raise RuntimeError(
                "La descarga automatica de paquetes solo esta disponible en Windows: "
                "los scripts son PowerShell. Corre el script a mano en esta plataforma."
            )

        path = script_path(job.pack)
        if not path.exists():
            raise FileNotFoundError(f"No existe el script de descarga {path}.")

        job.status = ProvisionStatus.running
        _stdout, stderr, returncode = await run_guarded_process(
            build_command(job.pack, job.variant),
            timeout=PROVISION_TIMEOUT_SECONDS,
            env=_powershell_safe_env(),
        )
        if returncode != 0:
            detail = _tail(stderr)
            raise RuntimeError(
                f"El script {path.name} termino con codigo {returncode}."
                + (f" {detail}" if detail else "")
            )
        job.status = ProvisionStatus.done
