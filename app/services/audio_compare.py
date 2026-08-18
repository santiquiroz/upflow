"""Probar varios separadores sobre un fragmento antes de comprometer el tema.

Que separador anda mejor depende del material: una voz limpia sobre pocos
instrumentos y una mezcla densa no se ordenan igual. Los rankings publicados
promedian sobre un dataset que no es el del usuario, asi que la unica respuesta
util es correrlos sobre SU archivo.

El fragmento se corta UNA vez y se comparte: cortarlo por modelo daria tres
archivos identicos y tres pasadas de ffmpeg, y —peor— si el corte variara, la
comparacion dejaria de ser entre modelos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.services.audio_excerpt import (
    build_excerpt_command,
    centered_offset,
    probe_duration_seconds,
)
from app.services.auth.identity import AuthenticatedUser
from app.services.download_chain import AudioJobCreator, link_or_copy
from app.services.process_runner import run_guarded_process

logger = logging.getLogger(__name__)

# Mas de tres a la vez deja de ser una comparacion y pasa a ser una cola: el
# usuario espera igual que si corriera el tema entero, que es lo que se evita.
MAX_MODELS_PER_COMPARISON = 3


@dataclass(frozen=True, slots=True)
class ComparisonEntry:
    model_id: str
    job_id: str


@dataclass(frozen=True, slots=True)
class Comparison:
    entries: list[ComparisonEntry]
    offset_seconds: float
    excerpt_seconds: int


def validate_models(model_ids: list[str], installed: set[str]) -> list[str]:
    """Los ids pedidos, sin repetidos y en el orden en que llegaron."""
    unicos = list(dict.fromkeys(model_ids))
    if len(unicos) < 2:
        raise ValueError("Comparar necesita al menos dos modelos distintos.")
    if len(unicos) > MAX_MODELS_PER_COMPARISON:
        raise ValueError(
            f"Se pueden comparar hasta {MAX_MODELS_PER_COMPARISON} modelos a la vez."
        )
    faltantes = [model_id for model_id in unicos if model_id not in installed]
    if faltantes:
        raise ValueError(
            "Estos modelos no estan instalados: " + ", ".join(faltantes)
        )
    return unicos


async def cut_excerpt(
    source: Path,
    destination: Path,
    settings: Settings,
    *,
    excerpt_seconds: int,
    offset_seconds: float | None,
) -> float:
    if offset_seconds is None:
        offset_seconds = centered_offset(
            await probe_duration_seconds(source, settings), excerpt_seconds
        )
    command = build_excerpt_command(
        ffmpeg=settings.ffmpeg_binary_path,
        source=source,
        destination=destination,
        offset_seconds=offset_seconds,
        excerpt_seconds=excerpt_seconds,
    )
    _stdout, stderr, returncode = await run_guarded_process(
        command, settings.subprocess_timeout
    )
    if returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
        detail = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(
            detail.splitlines()[-1] if detail else "No se pudo cortar el fragmento."
        )
    return offset_seconds


async def start_comparison(
    source: Path,
    original_filename: str,
    model_ids: list[str],
    *,
    audio_jobs: AudioJobCreator,
    settings: Settings,
    excerpt_seconds: int,
    offset_seconds: float | None = None,
    owner: AuthenticatedUser | None = None,
) -> Comparison:
    excerpt = settings.uploads_path / f"{uuid4().hex}-fragmento.wav"
    usado = await cut_excerpt(
        source,
        excerpt,
        settings,
        excerpt_seconds=excerpt_seconds,
        offset_seconds=offset_seconds,
    )
    entries: list[ComparisonEntry] = []
    try:
        for model_id in model_ids:
            token = uuid4().hex
            propio = settings.uploads_path / f"{token}-fragmento.wav"
            link_or_copy(excerpt, propio)
            try:
                job = await audio_jobs.create_job(
                    source_path=propio,
                    original_filename=f"{Path(original_filename).stem} [{model_id}].wav",
                    separate=True,
                    separation_model=model_id,
                    job_id=token,
                    owner=owner,
                )
            except Exception:
                propio.unlink(missing_ok=True)
                raise
            entries.append(ComparisonEntry(model_id=model_id, job_id=job.id))
    finally:
        # El corte compartido ya no hace falta: cada trabajo tiene su nombre para
        # el mismo contenido y borra el suyo al terminar.
        excerpt.unlink(missing_ok=True)
    return Comparison(
        entries=entries, offset_seconds=usado, excerpt_seconds=excerpt_seconds
    )
