from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.engines.colorize import AB_ARTIFACT_NAME, ColorizeOptions, apply_ab, load_ab_artifact
from app.services.engines.face_restore import (
    FACES_MANIFEST_NAME,
    MANIFEST_SCHEMA_VERSION,
    FacePatch,
    blend_face,
    paste_faces,
    read_unit_png16,
)
from app.services.engines.tiled_restore_runner import RestoreCancelled

RECOMPOSE_UNAVAILABLE_CODE = "restore.error.recomposeUnavailable"


class RecomposeUnavailable(RuntimeError):
    code = RECOMPOSE_UNAVAILABLE_CODE


@dataclass(frozen=True, slots=True)
class FaceChoice:
    enabled: bool
    blend: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.blend) and 0.0 <= self.blend <= 1.0):
            raise ValueError(f"Face blend must be within [0, 1], got {self.blend}")


@dataclass(frozen=True, slots=True)
class ManifestFace:
    index: int
    matrix: np.ndarray
    blend: float
    aligned: str
    restored: str


@dataclass(frozen=True, slots=True)
class FacesManifest:
    scale: float
    before_faces: str
    faces: tuple[ManifestFace, ...]


@dataclass(frozen=True, slots=True)
class RecomposeResult:
    image: np.ndarray
    uncolored: np.ndarray | None
    face_states: Mapping[int, tuple[bool, float]]
    ab_reused: bool


def recompose(
    directory: Path,
    choices: Mapping[int, FaceChoice],
    *,
    colorize_options: ColorizeOptions | None = None,
    cancel_event: threading.Event | None = None,
) -> RecomposeResult:
    manifest = load_faces_manifest(directory)
    states = face_states(manifest, choices)
    ab_512 = _saved_ab(directory) if colorize_options is not None else None
    _raise_if_cancelled(cancel_event)
    before = read_unit_png16(directory / manifest.before_faces)
    pasted = paste_faces(before, face_patches(directory, manifest, states, cancel_event), manifest.scale)
    if ab_512 is None:
        return RecomposeResult(pasted, None, states, ab_reused=False)
    colored = apply_ab(pasted, ab_512, colorize_options, cancel_event=cancel_event)
    return RecomposeResult(colored, pasted, states, ab_reused=True)


def load_faces_manifest(directory: Path) -> FacesManifest:
    path = directory / FACES_MANIFEST_NAME
    if not path.is_file():
        raise RecomposeUnavailable("This result has no saved faces to recompose.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schemaVersion") != MANIFEST_SCHEMA_VERSION:
        raise RecomposeUnavailable(f"Unsupported faces manifest version {raw.get('schemaVersion')!r}.")
    return FacesManifest(
        scale=float(raw["scale"]),
        before_faces=_plain_name(raw["beforeFaces"]),
        faces=tuple(_manifest_face(entry) for entry in raw["faces"]),
    )


def face_states(manifest: FacesManifest, choices: Mapping[int, FaceChoice]) -> dict[int, tuple[bool, float]]:
    known = {face.index for face in manifest.faces}
    unknown = sorted(set(choices) - known)
    if unknown:
        raise ValueError(f"Faces {unknown} were not restored in this result; restored faces: {sorted(known)}")
    return {face.index: _state(face, choices.get(face.index)) for face in manifest.faces}


def face_patches(
    directory: Path,
    manifest: FacesManifest,
    states: Mapping[int, tuple[bool, float]],
    cancel_event: threading.Event | None = None,
) -> list[FacePatch]:
    patches = []
    for face in manifest.faces:
        enabled, blend = states[face.index]
        if not enabled:
            continue
        _raise_if_cancelled(cancel_event)
        aligned = read_unit_png16(directory / face.aligned)
        restored = read_unit_png16(directory / face.restored)
        patches.append(FacePatch(blend_face(restored, aligned, blend), face.matrix))
    return patches


def _state(face: ManifestFace, choice: FaceChoice | None) -> tuple[bool, float]:
    return (True, face.blend) if choice is None else (choice.enabled, choice.blend)


def _manifest_face(entry: Mapping[str, Any]) -> ManifestFace:
    return ManifestFace(
        index=int(entry["index"]),
        matrix=np.asarray(entry["matrix"], dtype=np.float64),
        blend=float(entry["blend"]),
        aligned=_plain_name(entry["aligned"]),
        restored=_plain_name(entry["restored"]),
    )


def _plain_name(name: str) -> str:
    # El manifiesto vive en disco del usuario: un nombre con ruta no sale del directorio del job.
    if Path(name).name != name or name in {"", ".", ".."}:
        raise RecomposeUnavailable(f"Invalid artifact name in the faces manifest: {name!r}")
    return name


def _saved_ab(directory: Path) -> np.ndarray:
    if not (directory / AB_ARTIFACT_NAME).is_file():
        raise RecomposeUnavailable("The saved colors are missing; colorize the photo again to recompose it.")
    return load_ab_artifact(directory)


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RestoreCancelled("Recompose cancelled")
