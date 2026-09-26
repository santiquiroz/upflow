from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from app.services.restore_outputs import RestoreOutputPaths, restore_output_paths
from app.services.restore_provenance import DownloadNames, download_names, sanitize_stem
from app.services.restore_recompose import load_faces_manifest

IMAGE_FAMILY = "image"
VIDEO_FAMILY = "video"
CCTV_DIR_SUFFIX = ".cctv"

IMAGE_FILE_ROLES: Mapping[str, Callable[[RestoreOutputPaths], Path]] = MappingProxyType(
    {
        "preview": lambda paths: paths.preview,
        "view": lambda paths: paths.view,
        "beforeafter": lambda paths: paths.before_after,
        "uncolored": lambda paths: paths.uncolored,
        "sidecar": lambda paths: paths.sidecar,
    }
)
FACE_ARTIFACT = re.compile(r"face:(?P<index>\d{1,3}):(?P<side>before|after)")

# Nombres fijos dentro de outputs/{id}.cctv/ (§4.1, §4.11); P2 los escribe con estos nombres.
VIDEO_FILES: Mapping[str, str] = MappingProxyType(
    {
        "analysis": "analysis.mkv",
        "viewing": "viewing.mp4",
        "enhanced": "enhanced.mp4",
        "comparison": "comparison.mp4",
        "report_json": "report.json",
        "report_html": "report.html",
        "sha256sums": "SHA256SUMS.txt",
        "reproduce": "reproduce.cmd",
        "package": "handover.zip",
        "frame_index": "frame_index.csv",
    }
)
STILL_ARTIFACT = re.compile(r"still:(?P<frame>\d{1,9}):(?P<side>original|processed)")
ROI_ARTIFACT = re.compile(r"roi:(?P<name>[a-z0-9][a-z0-9_-]{0,63})")

MEDIA_TYPES: Mapping[str, str] = MappingProxyType(
    {
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".json": "application/json",
        ".mkv": "video/x-matroska",
        ".mp4": "video/mp4",
        ".html": "text/html",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".cmd": "text/plain",
        ".zip": "application/zip",
    }
)


class UnknownArtifact(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: Path
    download_name: str
    media_type: str


def image_artifact_path(outputs_dir: Path, job_id: str, fmt: str, name: str) -> Path:
    paths = restore_output_paths(outputs_dir, job_id, fmt)
    role = IMAGE_FILE_ROLES.get(name)
    if role is not None:
        return role(paths)
    face = FACE_ARTIFACT.fullmatch(name)
    if face is None:
        raise UnknownArtifact(f"Unknown artifact {name!r}")
    return face_artifact_path(paths.artifact_dir, int(face["index"]), face["side"])


def face_artifact_path(artifact_dir: Path, index: int, side: str) -> Path:
    try:
        manifest = load_faces_manifest(artifact_dir)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise UnknownArtifact("This result has no restored faces") from exc
    for face in manifest.faces:
        if face.index == index:
            return inside(artifact_dir, face.aligned if side == "before" else face.restored)
    raise UnknownArtifact(f"Face {index} was not restored in this result")


def video_artifact_path(outputs_dir: Path, job_id: str, name: str) -> Path:
    directory = outputs_dir / f"{job_id}{CCTV_DIR_SUFFIX}"
    fixed = VIDEO_FILES.get(name)
    if fixed is not None:
        return inside(directory, fixed)
    still = STILL_ARTIFACT.fullmatch(name)
    if still is not None:
        return inside(directory, f"{still['side']}_f{int(still['frame'])}.png")
    roi = ROI_ARTIFACT.fullmatch(name)
    if roi is not None:
        return inside(directory, f"roi_{roi['name']}.png")
    raise UnknownArtifact(f"Unknown artifact {name!r}")


def inside(directory: Path, name: str) -> Path:
    candidate = (directory / name).resolve()
    if Path(name).name != name or not candidate.is_relative_to(directory.resolve()):
        raise UnknownArtifact(f"Artifact {name!r} is outside the job folder")
    return directory / name


def image_download_name(name: str, original_name: str, fmt: str, colorized: bool) -> str:
    names = download_names(original_name, fmt, colorized)
    return _image_names(names, sanitize_stem(original_name)).get(name) or _face_download_name(name, original_name)


def _image_names(names: DownloadNames, stem: str) -> dict[str, str]:
    return {
        "preview": f"{stem}_preview.jpg",
        "view": f"{stem}_view.jpg",
        "beforeafter": names.before_after,
        "uncolored": names.uncolored,
        "sidecar": names.sidecar,
    }


def _face_download_name(name: str, original_name: str) -> str:
    face = FACE_ARTIFACT.fullmatch(name)
    if face is None:
        raise UnknownArtifact(f"Unknown artifact {name!r}")
    return f"{sanitize_stem(original_name)}_face-{int(face['index'])}-{face['side']}.png"


def media_type_for(path: Path) -> str:
    return MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream")


def restore_artifact(
    outputs_dir: Path, job_id: str, fmt: str, name: str, original_name: str, restore: Mapping[str, Any]
) -> ArtifactFile:
    path = image_artifact_path(outputs_dir, job_id, fmt, name)
    if not path.is_file():
        raise UnknownArtifact(f"This result has no {name!r} artifact")
    colorized = restore.get("colorize") is not None
    return ArtifactFile(path, image_download_name(name, original_name, fmt, colorized), media_type_for(path))


def restored_download_name(restore: Mapping[str, Any] | None) -> str | None:
    names = (restore or {}).get("downloadNames") or {}
    value = names.get("restored")
    return value if isinstance(value, str) and value else None
