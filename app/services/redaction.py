"""Copia anonimizada para compartir (P4-REDACT: research-forense §3, guia de la SIC, Ley 1581).

Cajas puestas a mano con keyframes: entre dos keyframes la caja se interpola cuadro a
cuadro y, dentro del tramo de la caja, antes del primero y despues del ultimo se sostiene.
Cada caja se tapa con un relleno solido, se pixela o se desenfoca. Solo el relleno es
irreversible: el pixelado y el desenfoque de un sujeto en movimiento pueden dejar rasgos. La copia lleva el rotulo
"Redacted copy", vive en su propia carpeta del job y nunca entra en el paquete de entrega.
No hay detector automatico de caras: las cajas las decide la persona.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.models import CctvOptions, RedactionKeyframe, RedactionRequest, RedactionTrack
from app.services.cctv_chain import CctvChainError, ResolvedStep
from app.services.ffmpeg_filters import FrameGeometry
from app.services.label_band import LabelTexts

REDACT_TASK = "redact"
REDACTED_DIRNAME = "05_redacted"
REDACTION_LOG_NAME = "redaction.json"
REDACTED_TAG = "__upflow-redacted__"
REDACTED_EXTENSION = ".mp4"
LOG_SCHEMA_VERSION = 1

STYLES = ("fill", "pixelate", "blur")
FILL_RGB = (0, 0, 0)
MAX_TRACKS = 32
MAX_KEYFRAMES = 256
MIN_SIDE = 2
# Pocas celdas sobre el lado largo: una cara o una placa quedan ilegibles aunque la caja sea grande.
PIXELATE_CELLS = 6
PIXELATE_MIN_CELL_PX = 8
BLUR_CELLS = 3
ALLOWED_STEPS = frozenset({"trim"})

REDACTION_REQUIRED = "cctv.error.redactionRequired"
REDACTION_UNEXPECTED = "cctv.error.redactionUnexpected"
REDACTION_INVALID = "cctv.error.redactionInvalid"
REDACTION_OUTSIDE_FRAME = "cctv.error.redactionOutsideFrame"
REDACTION_FRAMES = "cctv.error.redactionFrames"
REDACTION_STEPS = "cctv.error.redactionSteps"

REDACTED_LABEL = LabelTexts(
    "REDACTED COPY — NOT THE ORIGINAL RECORDING",
    "COPIA ANONIMIZADA — NO ES LA GRABACIÓN ORIGINAL",
    "Redacted · ",
    "Redacted copy made by Upflow {version} (job {job_id}); not the original recording",
)
NOTICE = (
    "Redacted copy for sharing. It is not the original recording and it is not part of the handover package. "
    "Keep the original and hand it over separately."
)
LIMITATIONS = (
    "Boxes were placed by hand. Between keyframes they move in straight lines; check the whole copy before sharing it.",
    "Only the solid fill removes everything under a box. Pixelation and blur in video can leave a moving face, "
    "plate or text recognizable across frames; use the solid fill for plates and text.",
    "Frames listed under uncoveredFrames have no box at all and are shared as recorded.",
    "Audio is not included in the redacted copy.",
    "Frames keep their order and count, but play back at a constant rate.",
)

Box = tuple[int, int, int, int]


# --- Validacion del pedido ---


def redaction_error(key: str, message: str) -> CctvChainError:
    return CctvChainError(key, message)


def check_style(style: str) -> None:
    if style not in STYLES:
        raise redaction_error(REDACTION_INVALID, f"Redaction style must be one of: {', '.join(STYLES)}.")


def check_track_count(tracks: Sequence[RedactionTrack]) -> None:
    if not 1 <= len(tracks) <= MAX_TRACKS:
        raise redaction_error(REDACTION_INVALID, f"Draw between 1 and {MAX_TRACKS} boxes to redact.")


def check_track_span(track: RedactionTrack, frame_count: int) -> None:
    if not 0 <= track.first_frame <= track.last_frame < frame_count:
        raise redaction_error(
            REDACTION_FRAMES,
            f"Each box must start and end inside the video (frames 0-{frame_count - 1}).",
        )


def keyframe_frames(track: RedactionTrack) -> list[int]:
    return [keyframe.frame for keyframe in track.keyframes]


def is_strictly_increasing(values: Sequence[int]) -> bool:
    return all(earlier < later for earlier, later in zip(values, values[1:]))


def check_keyframes(track: RedactionTrack) -> None:
    frames = keyframe_frames(track)
    if not 1 <= len(frames) <= MAX_KEYFRAMES or not is_strictly_increasing(frames):
        raise redaction_error(
            REDACTION_FRAMES,
            f"Each box needs 1 to {MAX_KEYFRAMES} keyframes in increasing frame order, without repeats.",
        )
    if frames[0] < track.first_frame or frames[-1] > track.last_frame:
        raise redaction_error(REDACTION_FRAMES, "Keyframes must be inside the frames the box covers.")


def check_box(box: Sequence[int], geometry: FrameGeometry) -> None:
    x, y, w, h = box
    inside = x >= 0 and y >= 0 and x + w <= geometry.width and y + h <= geometry.height
    if w < MIN_SIDE or h < MIN_SIDE or not inside:
        raise redaction_error(
            REDACTION_OUTSIDE_FRAME,
            f"The box {tuple(box)!r} is outside the {geometry.width}x{geometry.height} frame.",
        )


def check_track(track: RedactionTrack, geometry: FrameGeometry, frame_count: int) -> None:
    check_track_span(track, frame_count)
    check_keyframes(track)
    for keyframe in track.keyframes:
        check_box(keyframe.box, geometry)


def check_request(request: RedactionRequest, geometry: FrameGeometry, frame_count: int) -> None:
    check_style(request.style)
    check_track_count(request.tracks)
    for track in request.tracks:
        check_track(track, geometry, frame_count)


def check_redaction(options: CctvOptions, geometry: FrameGeometry, frame_count: int) -> None:
    if options.task != REDACT_TASK:
        if options.redaction is not None:
            raise redaction_error(REDACTION_UNEXPECTED, "Redaction boxes only apply to the redacted copy task.")
        return
    if options.redaction is None:
        raise redaction_error(REDACTION_REQUIRED, "The redacted copy needs at least one box.")
    check_request(options.redaction, geometry, frame_count)


def overlaps(track: RedactionTrack, first: int, last: int) -> bool:
    return track.first_frame <= last and first <= track.last_frame


def check_redaction_window(options: CctvOptions, first: int, last: int) -> None:
    if options.task != REDACT_TASK or options.redaction is None:
        return
    for number, track in enumerate(options.redaction.tracks, start=1):
        if not overlaps(track, first, last):
            raise redaction_error(
                REDACTION_FRAMES,
                f"Box {number} covers frames {track.first_frame}-{track.last_frame}, outside the copy ({first}-{last}).",
            )


def check_redaction_steps(options: CctvOptions, steps: Sequence[ResolvedStep]) -> None:
    if options.task != REDACT_TASK:
        return
    extra = [step.id for step in steps if step.id not in ALLOWED_STEPS]
    if extra or options.still_frames:
        raise redaction_error(
            REDACTION_STEPS,
            "The redacted copy only accepts a trim: no filters, on-screen text boxes or still frames.",
        )


# --- Cajas por cuadro ---


def round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def lerp_box(start: Box, end: Box, fraction: float) -> Box:
    x, y, w, h = (round_half_up(a + (b - a) * fraction) for a, b in zip(start, end))
    return (x, y, w, h)


def covers(track: RedactionTrack, frame: int) -> bool:
    return track.first_frame <= frame <= track.last_frame


def held_or_between(keyframes: Sequence[RedactionKeyframe], frame: int) -> Box:
    frames = [keyframe.frame for keyframe in keyframes]
    if frame <= frames[0]:
        return tuple(keyframes[0].box)
    if frame >= frames[-1]:
        return tuple(keyframes[-1].box)
    after = bisect.bisect_right(frames, frame)
    before = keyframes[after - 1]
    fraction = (frame - before.frame) / (keyframes[after].frame - before.frame)
    return lerp_box(tuple(before.box), tuple(keyframes[after].box), fraction)


def box_at(track: RedactionTrack, frame: int) -> Box | None:
    return held_or_between(track.keyframes, frame) if covers(track, frame) else None


def boxes_at(tracks: Sequence[RedactionTrack], frame: int) -> tuple[Box, ...]:
    return tuple(box for box in (box_at(track, frame) for track in tracks) if box is not None)


def clipped_spans(tracks: Sequence[RedactionTrack], first: int, last: int) -> list[tuple[int, int]]:
    spans = ((max(first, track.first_frame), min(last, track.last_frame)) for track in tracks)
    return sorted(span for span in spans if span[0] <= span[1])


def uncovered_ranges(tracks: Sequence[RedactionTrack], first: int, last: int) -> list[tuple[int, int]]:
    gaps: list[tuple[int, int]] = []
    following = first
    for start, end in clipped_spans(tracks, first, last):
        if start > following:
            gaps.append((following, start - 1))
        following = max(following, end + 1)
    return [*gaps, (following, last)] if following <= last else gaps


# --- Relleno, pixelado y desenfoque ---


@dataclass(frozen=True, slots=True)
class Region:
    box: Box
    cell: int


def pixelate_cell(width: int, height: int) -> int:
    return max(PIXELATE_MIN_CELL_PX, math.ceil(max(width, height) / PIXELATE_CELLS))


def track_cell(track: RedactionTrack) -> int:
    return pixelate_cell(max(key.box[2] for key in track.keyframes), max(key.box[3] for key in track.keyframes))


def region_at(track: RedactionTrack, frame: int) -> Region | None:
    box = box_at(track, frame)
    return Region(box, track_cell(track)) if box is not None else None


def regions_at(tracks: Sequence[RedactionTrack], frame: int) -> tuple[Region, ...]:
    return tuple(region for region in (region_at(track, frame) for track in tracks) if region is not None)


def reduced_size(width: int, height: int, cell: float) -> tuple[int, int]:
    return max(1, math.ceil(width / cell)), max(1, math.ceil(height / cell))


def pixelate_region(region: np.ndarray) -> np.ndarray:
    height, width = region.shape[:2]
    small = cv2.resize(region, reduced_size(width, height, pixelate_cell(width, height)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def aligned_span(start: int, length: int, cell: int) -> tuple[int, int]:
    return (start // cell) * cell, -(-(start + length) // cell) * cell


def block_means(block: np.ndarray, cell: int) -> np.ndarray:
    rows, cols = block.shape[0] // cell, block.shape[1] // cell
    means = block.reshape(rows, cell, cols, cell, -1).mean(axis=(1, 3))
    return np.repeat(np.repeat(np.rint(means).astype(block.dtype), cell, axis=0), cell, axis=1)


def grid_pixelated(frame: np.ndarray, box: Box, cell: int) -> np.ndarray:
    # La grilla se ancla al cuadro y no a la caja: la fase de muestreo no cambia cuando la caja o el sujeto se mueven.
    x, y, w, h = box
    top, bottom = aligned_span(y, h, cell)
    left, right = aligned_span(x, w, cell)
    pad = ((0, max(0, bottom - frame.shape[0])), (0, max(0, right - frame.shape[1])), (0, 0))
    block = np.pad(frame[top:bottom, left:right], pad, mode="edge")
    return block_means(block, cell)[y - top : y - top + h, x - left : x - left + w]


def blur_region(region: np.ndarray) -> np.ndarray:
    # Reducir a unas pocas celdas antes de ampliar suave: un gaussiano solo deja rasgos recuperables.
    height, width = region.shape[:2]
    small = cv2.resize(region, reduced_size(width, height, max(width, height) / BLUR_CELLS), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def filled(frame: np.ndarray, region: Region) -> np.ndarray:
    _, _, w, h = region.box
    return np.broadcast_to(np.array(FILL_RGB, dtype=frame.dtype), (h, w, frame.shape[2]))


def pixelated(frame: np.ndarray, region: Region) -> np.ndarray:
    return grid_pixelated(frame, region.box, region.cell)


def blurred(frame: np.ndarray, region: Region) -> np.ndarray:
    x, y, w, h = region.box
    return blur_region(frame[y : y + h, x : x + w])


REGION_RENDERERS = {"fill": filled, "pixelate": pixelated, "blur": blurred}


def redact_regions(frame: np.ndarray, regions: Sequence[Region], style: str) -> np.ndarray:
    if not regions:
        return frame
    render = REGION_RENDERERS[style]
    redacted = frame.copy()
    for region in regions:
        x, y, w, h = region.box
        redacted[y : y + h, x : x + w] = render(frame, region)
    return redacted


def redact_frame(frame: np.ndarray, boxes: Sequence[Box], style: str) -> np.ndarray:
    return redact_regions(frame, [Region(box, pixelate_cell(box[2], box[3])) for box in boxes], style)


# --- Registro de la copia ---


def track_json(track: RedactionTrack) -> dict[str, Any]:
    return {
        "firstFrame": track.first_frame,
        "lastFrame": track.last_frame,
        "keyframes": [{"frame": key.frame, "box": list(key.box)} for key in track.keyframes],
    }


@dataclass(frozen=True, slots=True)
class RedactionLogFacts:
    app_version: str
    job_id: str
    source_name: str
    source_sha256: str
    first_frame: int
    last_frame: int
    frames_out: int
    rate: str
    output_file: str
    output_sha256: str


def redaction_log(request: RedactionRequest, facts: RedactionLogFacts) -> dict[str, Any]:
    return {
        "schemaVersion": LOG_SCHEMA_VERSION,
        "kind": "redacted-copy",
        "upflow": {"version": facts.app_version},
        "jobId": facts.job_id,
        "source": {"name": facts.source_name, "sha256": facts.source_sha256},
        "style": request.style,
        "frames": {"first": facts.first_frame, "last": facts.last_frame, "out": facts.frames_out},
        "rate": facts.rate,
        "tracks": [track_json(track) for track in request.tracks],
        "uncoveredFrames": [list(span) for span in uncovered_ranges(request.tracks, facts.first_frame, facts.last_frame)],
        "output": {"file": facts.output_file, "sha256": facts.output_sha256, "audio": False},
        "notice": NOTICE,
        "limitations": list(LIMITATIONS),
    }
