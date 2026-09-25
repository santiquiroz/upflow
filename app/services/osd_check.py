"""Cajas del OSD de un video CCTV (spec §4.8).

El job no arranca sin cajas confirmadas por el usuario o sin "No on-screen
text": `validate_osd_selection` lo rechaza con `cctv.error.osdUnconfirmed`. El
chequeo barato `osd_box_looks_like_text` mira 10 cuadros espaciados 1 s: la
caja tiene que tener alto contraste local y ser casi estatica (el segundero
cambia, el resto no). Si no, el aviso `cctv.osd.notText` no bloquea el job,
pero queda en el informe junto con las cajas, la confirmacion y los filtros
temporales de la cadena, que son los que mezclan los digitos si la caja esta
mal puesta. Los umbrales son propuestas: se calibran en P2-VAL.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.cctv_chain import CctvChainError, ResolvedStep
from app.services.ffmpeg_filters import OSD_BOX_OUTSIDE_FRAME
from app.services.process_runner import run_guarded_process
from app.services.video_analysis import ProcessRunner, VideoAnalysisError, run_pass

OSD_UNCONFIRMED = "cctv.error.osdUnconfirmed"
OSD_CONFLICT = "cctv.error.osdConflict"
OSD_BOX_ODD = "cctv.error.osdBoxOdd"
TOO_MANY_OSD_BOXES = "cctv.error.tooManyOsdBoxes"
OSD_NOT_TEXT = "cctv.osd.notText"

MIN_CONTRAST = 60.0
LOW_PERCENTILE = 5
HIGH_PERCENTILE = 95
STATIC_MAX_DIFF = 8.0
MIN_STATIC_FRACTION = 0.70
SAMPLE_FRAMES = 10
SAMPLE_SPACING_SECONDS = 1.0
SPACING_TOLERANCE_SECONDS = 0.01
SAMPLE_TIMEOUT_SECONDS = 300.0
MAX_OSD_BOXES = 8
REPORT_DECIMALS = 3

ALWAYS_TEMPORAL_FILTERS = frozenset({"hqdn3d", "atadenoise", "tmedian"})
NEIGHBOR_FRAME_PARAMS = ("prev", "next")
SAMPLE_PASS = "osd-sample"

Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class OsdSelection:
    boxes: tuple[Box, ...]
    confirmed: bool
    no_osd: bool


@dataclass(frozen=True, slots=True)
class OsdBoxCheck:
    box: Box
    frames: int
    contrast: float
    static_fraction: float | None
    looks_like_text: bool


def _outside(box: Any, width: int, height: int) -> CctvChainError:
    return CctvChainError(OSD_BOX_OUTSIDE_FRAME, f"OSD box {box!r} is outside the {width}x{height} frame.")


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_box_shape(box: Any) -> bool:
    return isinstance(box, (tuple, list)) and len(box) == 4 and all(_is_plain_int(value) for value in box)


def check_box_inside(box: Any, width: int, height: int) -> Box:
    if not _is_box_shape(box):
        raise _outside(box, width, height)
    x, y, w, h = box
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
        raise _outside(box, width, height)
    return (x, y, w, h)


def _check_even(box: Box) -> None:
    if box[2] % 2 or box[3] % 2:
        raise CctvChainError(OSD_BOX_ODD, f"OSD box {box!r} must have an even width and height.")


def _check_box_count(boxes: Sequence[Box]) -> None:
    if len(boxes) > MAX_OSD_BOXES:
        raise CctvChainError(TOO_MANY_OSD_BOXES, f"At most {MAX_OSD_BOXES} on-screen text boxes are allowed.")


def _reject_conflict(selection: OsdSelection) -> None:
    if selection.no_osd and (selection.boxes or selection.confirmed):
        raise CctvChainError(OSD_CONFLICT, "'No on-screen text' cannot be combined with on-screen text boxes.")


def _require_confirmation(selection: OsdSelection) -> None:
    if not selection.no_osd and not (selection.boxes and selection.confirmed):
        raise CctvChainError(
            OSD_UNCONFIRMED,
            "Confirm the on-screen text boxes on a frame where the time is visible, or choose 'No on-screen text'.",
        )


def validate_osd_selection(selection: OsdSelection, width: int, height: int) -> None:
    _reject_conflict(selection)
    _require_confirmation(selection)
    _check_box_count(selection.boxes)
    for box in selection.boxes:
        _check_even(check_box_inside(box, width, height))


def box_patches(frames: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    return frames[:, y : y + h, x : x + w]


def local_contrast(patches: np.ndarray) -> float:
    flat = patches.reshape(patches.shape[0], -1)
    spread = np.percentile(flat, HIGH_PERCENTILE, axis=1) - np.percentile(flat, LOW_PERCENTILE, axis=1)
    return float(np.median(spread))


def static_fraction(patches: np.ndarray) -> float | None:
    if patches.shape[0] < 2:
        return None
    changes = np.abs(np.diff(patches.astype(np.int16), axis=0))
    return float((np.median(changes, axis=0) <= STATIC_MAX_DIFF).mean())


def _is_text(contrast: float, stillness: float | None) -> bool:
    is_still = stillness is None or stillness >= MIN_STATIC_FRACTION
    return contrast >= MIN_CONTRAST and is_still


def osd_box_looks_like_text(frames: np.ndarray, box: Box) -> OsdBoxCheck:
    if frames.shape[0] == 0:
        raise ValueError("The on-screen text check needs at least one frame.")
    checked = check_box_inside(box, frames.shape[2], frames.shape[1])
    patches = box_patches(frames, checked)
    contrast = local_contrast(patches)
    stillness = static_fraction(patches)
    return OsdBoxCheck(checked, int(frames.shape[0]), contrast, stillness, _is_text(contrast, stillness))


def check_osd_boxes(frames: np.ndarray, boxes: Sequence[Box]) -> tuple[OsdBoxCheck, ...]:
    return tuple(osd_box_looks_like_text(frames, box) for box in boxes)


def osd_warnings(checks: Sequence[OsdBoxCheck]) -> tuple[str, ...]:
    return (OSD_NOT_TEXT,) if any(not check.looks_like_text for check in checks) else ()


def _mixes_frames(step: ResolvedStep) -> bool:
    if step.filter in ALWAYS_TEMPORAL_FILTERS:
        return True
    return step.filter == "fftdnoiz" and any(step.params.get(name) for name in NEIGHBOR_FRAME_PARAMS)


def temporal_filters_in(steps: Sequence[ResolvedStep]) -> tuple[str, ...]:
    return tuple(step.filter for step in steps if _mixes_frames(step))


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, REPORT_DECIMALS)


def _check_entry(check: OsdBoxCheck) -> dict[str, Any]:
    return {
        "box": list(check.box),
        "looksLikeText": check.looks_like_text,
        "contrast": _rounded(check.contrast),
        "staticFraction": _rounded(check.static_fraction),
        "framesChecked": check.frames,
    }


def osd_report(
    selection: OsdSelection, checks: Sequence[OsdBoxCheck], steps: Sequence[ResolvedStep]
) -> dict[str, Any]:
    return {
        "boxes": [list(box) for box in selection.boxes],
        "osdBoxesConfirmed": selection.confirmed,
        "noOsd": selection.no_osd,
        "checks": [_check_entry(check) for check in checks],
        "temporalFilters": list(temporal_filters_in(steps)),
        "warnings": list(osd_warnings(checks)),
    }


def _seek_args(start_seconds: float) -> list[str]:
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError(f"The sample start must be a finite, non-negative time, got {start_seconds!r}.")
    return ["-ss", f"{start_seconds:.3f}"] if start_seconds > 0 else []


def sample_filter() -> str:
    # select y no fps: fps duplica cuadros en los huecos de un VFR, y un cuadro
    # repetido haria parecer estatica cualquier caja.
    spacing = SAMPLE_SPACING_SECONDS - SPACING_TOLERANCE_SECONDS
    return f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{spacing:g}),format=gray"


def build_osd_sample_command(ffmpeg: Path, video: Path, start_seconds: float = 0.0) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-nostats",
        *_seek_args(start_seconds),
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        sample_filter(),
        "-fps_mode",
        "passthrough",
        "-frames:v",
        str(SAMPLE_FRAMES),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]


def split_gray_frames(buffer: bytes, width: int, height: int) -> np.ndarray:
    count = len(buffer) // (width * height)
    return np.frombuffer(buffer, dtype=np.uint8, count=count * width * height).reshape(count, height, width)


async def sample_osd_frames(
    ffmpeg: Path,
    video: Path,
    width: int,
    height: int,
    start_seconds: float = 0.0,
    run: ProcessRunner = run_guarded_process,
    timeout: float = SAMPLE_TIMEOUT_SECONDS,
) -> np.ndarray:
    command = build_osd_sample_command(ffmpeg, video, start_seconds)
    stdout, _ = await run_pass(SAMPLE_PASS, command, run, timeout)
    frames = split_gray_frames(stdout, width, height)
    if frames.shape[0] == 0:
        raise VideoAnalysisError(SAMPLE_PASS, "No frames were decoded for the on-screen text check.")
    return frames


async def run_osd_check(
    ffmpeg: Path,
    video: Path,
    width: int,
    height: int,
    boxes: Sequence[Box],
    start_seconds: float = 0.0,
    run: ProcessRunner = run_guarded_process,
) -> tuple[OsdBoxCheck, ...]:
    for box in boxes:
        check_box_inside(box, width, height)
    frames = await sample_osd_frames(ffmpeg, video, width, height, start_seconds, run)
    return check_osd_boxes(frames, boxes)
