"""Lista blanca de artefactos de un job CCTV (spec §5.4 `job_artifacts`, familia video).

Un nombre de la lista se resuelve contra `job.metadata["cctv"]["outputs"]` (rutas
relativas a `outputs/{id}.cctv/`) o contra los archivos fijos del paquete. Todo lo
demas, y cualquier ruta que escape del directorio del job, se rechaza. Los artefactos de la
familia image (restauracion de fotos) viven en `job_artifacts.py`: las rutas de CCTV salen de la
metadata del job, no de una tabla de nombres fijos.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.models import JobStatus, VideoUpscaleJob
from app.services.cctv_ingest import FRAME_INDEX_NAME
from app.services.cctv_report import REPORT_HTML_NAME, REPORT_JSON_NAME, SHA256SUMS_NAME
from app.services.handover_package import REPRODUCE_NAME

UNKNOWN_ARTIFACT = "cctv.error.unknownArtifact"

FIXED_ARTIFACTS: Mapping[str, str] = {
    "report_json": REPORT_JSON_NAME,
    "report_html": REPORT_HTML_NAME,
    "sha256sums": SHA256SUMS_NAME,
    "reproduce": REPRODUCE_NAME,
    "frame_index": FRAME_INDEX_NAME,
}
OUTPUT_ARTIFACTS = ("analysis", "viewing", "enhanced", "comparison", "package")
STILL_ROLES = ("original", "processed")
MEDIA_TYPES: Mapping[str, str] = {
    ".mkv": "video/x-matroska",
    ".mp4": "video/mp4",
    ".zip": "application/zip",
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".png": "image/png",
}
INLINE_ARTIFACTS = frozenset({"report_html", "report_json"})
DOWNLOAD_MEDIA_TYPE = "application/octet-stream"

_STILL = re.compile(r"still:(\d{1,9}):(original|processed)")
_ROI = re.compile(r"roi:([A-Za-z0-9_-]{1,64})")


class ArtifactNotFound(LookupError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Artifact {name!r} is not available for this job.")
        self.key = UNKNOWN_ARTIFACT


def cctv_outputs(job: VideoUpscaleJob) -> Mapping[str, Any]:
    outputs = job.metadata.get("cctv", {}).get("outputs")
    return outputs if isinstance(outputs, Mapping) else {}


def still_file(outputs: Mapping[str, Any], frame: int, role: str) -> str | None:
    pairs = outputs.get("stills") or []
    pair = next((item for item in pairs if isinstance(item, Mapping) and item.get("frame") == frame), None)
    entry = None if pair is None else pair.get(role)
    return entry.get("file") if isinstance(entry, Mapping) else None


def roi_file(outputs: Mapping[str, Any], name: str) -> str | None:
    rois = outputs.get("roi")
    return rois.get(name) if isinstance(rois, Mapping) else None


def pattern_artifact(name: str, outputs: Mapping[str, Any]) -> str | None:
    still = _STILL.fullmatch(name)
    if still is not None:
        return still_file(outputs, int(still.group(1)), still.group(2))
    roi = _ROI.fullmatch(name)
    return None if roi is None else roi_file(outputs, roi.group(1))


def artifact_relative_path(name: str, outputs: Mapping[str, Any]) -> str | None:
    if name in FIXED_ARTIFACTS:
        return FIXED_ARTIFACTS[name]
    if name in OUTPUT_ARTIFACTS:
        value = outputs.get(name)
        return value if isinstance(value, str) else None
    return pattern_artifact(name, outputs)


def contained_file(job_dir: Path, relative: str) -> Path | None:
    base = job_dir.resolve()
    candidate = (base / relative).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        return None
    return candidate


def resolve_artifact(job_dir: Path, name: str, outputs: Mapping[str, Any]) -> Path:
    relative = artifact_relative_path(name, outputs)
    path = None if relative is None else contained_file(job_dir, relative)
    if path is None:
        raise ArtifactNotFound(name)
    return path


def media_type_for(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), DOWNLOAD_MEDIA_TYPE)


def is_inline(name: str) -> bool:
    return name in INLINE_ARTIFACTS


def still_names(outputs: Mapping[str, Any]) -> list[str]:
    pairs = [item for item in outputs.get("stills") or [] if isinstance(item, Mapping)]
    return [f"still:{pair.get('frame')}:{role}" for pair in pairs for role in STILL_ROLES if pair.get(role)]


def roi_names(outputs: Mapping[str, Any]) -> list[str]:
    rois = outputs.get("roi")
    return [f"roi:{name}" for name in rois] if isinstance(rois, Mapping) else []


def listed_artifacts(job: VideoUpscaleJob) -> list[str]:
    # Sin tocar el disco: el resumen del job sale en cada listado de jobs.
    if job.cctv is None or job.status != JobStatus.completed:
        return []
    outputs = cctv_outputs(job)
    produced = [name for name in OUTPUT_ARTIFACTS if isinstance(outputs.get(name), str)]
    return [*produced, *FIXED_ARTIFACTS, *still_names(outputs), *roi_names(outputs)]
