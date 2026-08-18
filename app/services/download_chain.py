"""Empalme entre una descarga y la separacion en pistas.

Pegar un link y recibir las pistas es UN pedido para el usuario y DOS trabajos
para el servidor. Este modulo es la union, y vive aparte del manager de descargas
para que ese siga sin saber que existe el audio: recibe una funcion y la llama.

El archivo bajado no se le entrega directo al trabajo de audio. Ese borra su
fuente al terminar (`AudioJobManager._cleanup_source`), asi que encadenar directo
dejaria la descarga sin resultado justo cuando la separacion sale bien.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.models import AudioJob, DownloadJob
from app.services.auth.identity import AuthenticatedUser

logger = logging.getLogger(__name__)


class AudioJobCreator(Protocol):
    async def create_job(self, **kwargs: object) -> AudioJob: ...


def link_or_copy(source: Path, destination: Path) -> None:
    """Dos nombres para el mismo contenido; una copia solo si el disco no puede.

    El hardlink es lo correcto aca: los dos nombres caen en uploads/, o sea el
    mismo volumen, y cada trabajo borra el suyo sin tocar el del otro. La copia es
    el plan B para un FAT32 o un montaje de red que no los soporta, donde pagar el
    doble de disco es mejor que no poder encadenar.
    """
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stage_for_audio_job(source: Path, uploads_path: Path) -> tuple[str, Path]:
    """Prepara una entrada propia para el trabajo de audio y devuelve su token.

    El token es el nombre del archivo Y el id del trabajo, igual que en una subida
    manual: asi el archivo staged se rastrea hasta el trabajo que lo consume.
    """
    token = uuid4().hex
    destination = uploads_path / f"{token}-{source.name}"
    link_or_copy(source, destination)
    return token, destination


async def start_separation_followups(
    job: DownloadJob,
    *,
    audio_jobs: AudioJobCreator,
    uploads_path: Path,
    owner: AuthenticatedUser | None,
) -> list[str]:
    """Un trabajo de separacion por archivo bajado, en el orden en que se bajaron.

    Una playlist produce varios archivos, y cada uno es una cancion distinta: son
    trabajos separados, no uno con varias entradas.
    """
    created: list[str] = []
    for path in job.output_paths:
        token, staged = stage_for_audio_job(path, uploads_path)
        try:
            audio_job = await audio_jobs.create_job(
                source_path=staged,
                original_filename=path.name,
                separate=True,
                separation_model=job.then_separation_model,
                job_id=token,
                owner=owner,
            )
        except Exception:
            # Sin esto un rechazo (cuota, modelo sin instalar) deja el archivo
            # staged en uploads/ para siempre: nadie mas conoce ese nombre.
            staged.unlink(missing_ok=True)
            raise
        created.append(audio_job.id)
    return created
