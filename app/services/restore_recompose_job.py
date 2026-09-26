from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.services.engines.colorize import ColorizeOptions
from app.services.image_io import LoadedImage, load_image_for_restore, save_restored
from app.services.restore_outputs import (
    RestoreOutputPaths,
    restore_output_paths,
    save_views,
    write_jpeg,
)
from app.services.restore_provenance import (
    BEFORE_AFTER_DIVIDER_PX,
    OutputFile,
    badge_applies,
    before_after_image,
    facts_from_sidecar,
    is_composite,
    recomposed_sidecar,
    uncolored_marks,
    with_badge,
    write_sidecar,
    xmp_fields,
)
from app.services.restore_recompose import (
    FaceChoice,
    FacesManifest,
    RecomposeResult,
    RecomposeUnavailable,
    load_faces_manifest,
    recompose,
)
from app.services.xmp_packet import build_xmp_packet


@dataclass(frozen=True, slots=True)
class RecomposeTarget:
    outputs_dir: Path
    job_id: str
    fmt: str
    options: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RecomposeOutcome:
    sidecar: dict[str, Any]
    badge: bool


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def recompose_outputs(
    target: RecomposeTarget,
    choices: Mapping[int, FaceChoice],
    *,
    now: Callable[[], str] = utc_timestamp,
    cancel_event: threading.Event | None = None,
) -> RecomposeOutcome:
    paths = restore_output_paths(target.outputs_dir, target.job_id, target.fmt)
    sidecar = with_manifest_faces(read_sidecar(paths.sidecar), load_faces_manifest(paths.artifact_dir))
    result = recompose(
        paths.artifact_dir, choices, colorize_options=colorize_options_of(sidecar), cancel_event=cancel_event
    )
    recomposed_at = now()
    draft = recomposed_sidecar(sidecar, result.face_states, (), recomposed_at)
    badge = badge_applies(facts_from_sidecar(draft), bool(target.options.get("badge", True)))
    outputs = _rewrite_outputs(paths, result, draft, badge, target.options)
    final_sidecar = recomposed_sidecar(sidecar, result.face_states, outputs, recomposed_at)
    write_sidecar(paths.sidecar, final_sidecar)
    return RecomposeOutcome(final_sidecar, badge)


def read_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RecomposeUnavailable("This result has no restoration details to recompose.")
    return json.loads(path.read_text(encoding="utf-8"))


def with_manifest_faces(sidecar: Mapping[str, Any], manifest: FacesManifest) -> dict[str, Any]:
    # Las caras que detecto el job (sin sesion) no estan en el sidecar: el manifiesto es la
    # verdad de cuales se restauraron, y sin ellas el recuento de caras daria cero.
    faces = list(sidecar.get("faces") or [])
    known = {int(face["index"]) for face in faces}
    extra = [_restored_face(face.index, face.blend) for face in manifest.faces if face.index not in known]
    return {**sidecar, "faces": faces + extra}


def _restored_face(index: int, blend: float) -> dict[str, Any]:
    return {"index": index, "enabled": True, "restored": True, "blend": blend, "recomposedAt": None}


def colorize_options_of(sidecar: Mapping[str, Any]) -> ColorizeOptions | None:
    colorize = sidecar.get("colorize")
    if colorize is None:
        return None
    return ColorizeOptions(
        strength=float(colorize.get("strength", 1.0)), saturation=float(colorize.get("saturation", 1.0))
    )


def before_half(before_after: np.ndarray, after_shape: tuple[int, int]) -> np.ndarray:
    height = before_after.shape[0]
    after_width = (
        after_shape[1] if after_shape[0] == height else max(1, round(after_shape[1] * height / after_shape[0]))
    )
    width = before_after.shape[1] - BEFORE_AFTER_DIVIDER_PX - after_width
    if width <= 0:
        raise RecomposeUnavailable("The saved before/after image does not match this result.")
    return before_after[:, :width].astype(np.float32) / np.float32(255.0)


def _rewrite_outputs(
    paths: RestoreOutputPaths,
    result: RecomposeResult,
    draft: Mapping[str, Any],
    badge: bool,
    options: Mapping[str, Any],
) -> tuple[OutputFile, ...]:
    previous = load_image_for_restore(paths.final)
    photo_date = None if options.get("photo_date") is None else str(options["photo_date"])
    xmp = build_xmp_packet(xmp_fields(draft, photo_date))
    keep_gps = bool(options.get("keep_gps", False))
    final = with_badge(result.image) if badge else result.image
    _save_like(previous, final, paths.final, xmp, keep_gps)
    outputs = [OutputFile("restored", paths.final)]
    if result.uncolored is not None:
        uncolored_xmp, uncolored_badge = uncolored_marks(draft, bool(options.get("badge", True)), photo_date)
        uncolored = with_badge(result.uncolored) if uncolored_badge else result.uncolored
        _save_like(previous, uncolored, paths.uncolored, uncolored_xmp, keep_gps)
        outputs.append(OutputFile("uncolored", paths.uncolored))
    outputs.extend(save_views(final, paths, previous.icc))
    before = before_half(_read_rgb8(paths.before_after), result.image.shape[:2])
    composite = is_composite(facts_from_sidecar(draft))
    write_jpeg(before_after_image(before, result.image, badge=composite), paths.before_after, previous.icc)
    outputs.append(OutputFile("beforeafter", paths.before_after))
    return tuple(outputs)


def _save_like(previous: LoadedImage, rgb: np.ndarray, path: Path, xmp: str, keep_gps: bool) -> None:
    # Mismo formato, profundidad, ICC y EXIF (ya sin GPS) que la salida que se reemplaza.
    fmt = path.suffix.lstrip(".")
    save_restored(rgb, path, fmt, previous.bit_depth, previous.icc, previous.exif, xmp, keep_gps=keep_gps)


def _read_rgb8(path: Path) -> np.ndarray:
    if not path.is_file():
        raise RecomposeUnavailable("The saved before/after image is missing.")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def recomposed_summary(restore: Mapping[str, Any], outcome: RecomposeOutcome) -> dict[str, Any]:
    sidecar = outcome.sidecar
    return {
        **restore,
        "faces": sidecar.get("faces", []),
        "digitalSourceType": sidecar["digitalSourceType"],
        "compositeReasons": list(sidecar["compositeReasons"]),
        "badge": outcome.badge,
    }
