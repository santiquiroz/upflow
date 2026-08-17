"""Modelo unificado de jobs sobre las 7 familias de la API.

La API expone 7 familias job-based con rutas y formas ligeramente distintas
(`jobId` vs `id`). Este módulo normaliza todo a un solo contrato para las
tools MCP: status/wait/cancel/download genéricos.

Las 7 listan sus jobs en su `base_path` (GET), así que no hay familias
"solo consultables por id".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.mcp import client

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


@dataclass(frozen=True)
class JobFamily:
    name: str
    base_path: str
    id_field: str
    default_output_name: str


FAMILIES: dict[str, JobFamily] = {
    "image": JobFamily("image", "/api/v1/jobs", "jobId", "upscaled.png"),
    "video": JobFamily("video", "/api/v1/video/jobs", "jobId", "upscaled.mp4"),
    "audio": JobFamily("audio", "/api/v1/audio/jobs", "id", "audio.flac"),
    "generation": JobFamily("generation", "/api/v1/generation/jobs", "id", "generated.png"),
    "transcribe": JobFamily("transcribe", "/api/v1/transcribe/jobs", "id", "transcript.txt"),
    "download": JobFamily("download", "/api/v1/download/jobs", "id", "media.bin"),
    "shape3d": JobFamily("shape3d", "/api/v1/print/generate", "id", "pieza.stl"),
}

FAMILY_NAMES = sorted(FAMILIES)


def family_or_raise(name: str) -> JobFamily:
    family = FAMILIES.get(name)
    if family is None:
        raise ValueError(
            f"Familia de job desconocida: '{name}'. Válidas: {', '.join(FAMILY_NAMES)}"
        )
    return family


def normalize_job(family: JobFamily, payload: dict[str, Any]) -> dict[str, Any]:
    """Reduce el payload crudo de cada familia a un contrato común y compacto."""
    normalized: dict[str, Any] = {
        "family": family.name,
        "jobId": payload.get(family.id_field) or payload.get("id") or payload.get("jobId"),
        "status": payload.get("status"),
        "progressPct": payload.get("progressPct"),
        "error": payload.get("error"),
        "downloadUrl": payload.get("downloadUrl"),
        "createdAt": payload.get("createdAt"),
        "finishedAt": payload.get("finishedAt"),
    }
    extras: dict[str, Any] = {}
    if family.name == "audio":
        extras["vocalsDownloadUrl"] = payload.get("vocalsDownloadUrl")
        extras["stems"] = payload.get("stems")
    if family.name == "transcribe":
        extras["text"] = payload.get("text")
        extras["videoUrl"] = payload.get("videoUrl")
    if family.name == "download":
        extras["outputFiles"] = payload.get("outputFiles")
        extras["mediaTitle"] = payload.get("mediaTitle")
    if family.name == "shape3d":
        extras["canPrint"] = payload.get("canPrint")
        extras["sizeMm"] = payload.get("sizeMm")
        extras["blockers"] = payload.get("blockers")
        extras["code"] = payload.get("code")
    if family.name == "video":
        extras["stage"] = (payload.get("metadata") or {}).get("stage")
    normalized.update({key: value for key, value in extras.items() if value is not None})
    return normalized


async def fetch_job(family: JobFamily, job_id: str) -> dict[str, Any]:
    payload = await client.api_get(f"{family.base_path}/{job_id}")
    return normalize_job(family, payload)


async def cancel_job(family: JobFamily, job_id: str) -> dict[str, Any]:
    payload = await client.api_post(f"{family.base_path}/{job_id}/cancel")
    return normalize_job(family, payload)


def is_terminal(job: dict[str, Any]) -> bool:
    return str(job.get("status", "")).lower() in TERMINAL_STATUSES
