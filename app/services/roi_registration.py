"""Registro de la ROI para la fusion multi-cuadro (spec §4.9 pasos 1, 3, 4 y 5).

Antes de registrar, las copias casi identicas del GOP (P-frames que repiten los
bloques de la referencia) se agrupan y cuentan como una sola muestra: la
ventana de busqueda de cada cuadro se compara con la de la ultima muestra
unica, y si la diferencia media absoluta queda bajo la mitad del piso de ruido
de la ROI de referencia, es copia. El piso de ruido es la sigma robusta (MAD)
del detalle diagonal de Haar de la ROI, para que la textura no cuente como
ruido. Cada muestra unica se registra con `cv2.phaseCorrelate` y
`cv2.findTransformECC` sobre la ventana ampliada x2 con Lanczos; bajo el umbral
de ECC se descarta y queda listada. La dispersion subpixel es circular (0,95 y
0,05 px estan cerca). Con menos de 3 muestras efectivas o todas en la misma fase
subpixel sale `cctv.roi.nearCopies`. Todo corre en CPU con un hilo de OpenCV.
Los umbrales son propuestas hasta P3-VAL.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

from app.models import RoiKind

Motion = Literal["translation", "affine", "homography"]
SampleStatus = Literal["reference", "accepted", "copy", "rejected"]
Matrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

NEAR_COPIES = "cctv.roi.nearCopies"
ROI_OUTSIDE_FRAME = "cctv.error.roiOutsideFrame"
ROI_REFERENCE_MISSING = "cctv.error.roiReferenceMissing"
ROI_NO_FRAMES = "cctv.error.roiFrames"

MIN_REGISTRATION_SIDE = 24
SATURATION_LEVEL = 250.0
MAX_REFERENCE_SATURATION = 0.05
MAD_TO_SIGMA = 0.6745
# Con una ROI sin ruido medible, dos cuadros identicos igual tienen que agruparse.
MIN_COPY_THRESHOLD = 0.05
MAX_CIRCULAR_SPREAD = 0.5
REPORT_DECIMALS = 6

ECC_MOTIONS: dict[Motion, int] = {
    "translation": cv2.MOTION_TRANSLATION,
    "affine": cv2.MOTION_AFFINE,
    "homography": cv2.MOTION_HOMOGRAPHY,
}
KIND_MOTIONS: dict[RoiKind, Motion] = {"plate": "homography", "face_or_object": "affine"}


class RoiRegistrationError(ValueError):
    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key


@dataclass(frozen=True, slots=True)
class RoiBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return self.x + (self.w - 1) / 2, self.y + (self.h - 1) / 2

    def fits(self, shape: tuple[int, ...]) -> bool:
        height, width = shape[:2]
        inside_x = 0 <= self.x and self.x + self.w <= width
        inside_y = 0 <= self.y and self.y + self.h <= height
        return self.w > 0 and self.h > 0 and inside_x and inside_y

    def crop(self, pixels: np.ndarray) -> np.ndarray:
        return pixels[self.y : self.y + self.h, self.x : self.x + self.w]


@dataclass(frozen=True, slots=True, eq=False)
class RoiFrame:
    n: int
    pict_type: str
    pixels: np.ndarray


@dataclass(frozen=True, slots=True)
class RegistrationSettings:
    ecc_min: float = 0.8
    upsample: int = 2
    gauss_filt_size: int = 5
    iterations: int = 100
    eps: float = 1e-6
    window_margin: float = 0.5
    copy_ratio: float = 0.5
    min_effective: int = 3
    near_copy_spread: float = 0.1


@dataclass(frozen=True, slots=True)
class Alignment:
    ecc: float | None
    matrix: Matrix | None

    def accepted(self, ecc_min: float) -> bool:
        return self.matrix is not None and self.ecc is not None and self.ecc >= ecc_min


@dataclass(frozen=True, slots=True)
class RoiSample:
    n: int
    pict_type: str
    copy_group: int
    status: SampleStatus
    ecc: float | None
    shift: tuple[float, float] | None
    matrix: Matrix | None

    def to_json(self) -> dict[str, Any]:
        return {
            "frame": self.n,
            "pictType": self.pict_type,
            "copyGroup": self.copy_group,
            "status": self.status,
            "ecc": rounded(self.ecc),
            "shift": None if self.shift is None else [rounded(value) for value in self.shift],
            "matrix": None if self.matrix is None else [[rounded(v) for v in row] for row in self.matrix],
        }


@dataclass(frozen=True, slots=True)
class SubpixelSpread:
    fractions_x: tuple[float, ...]
    fractions_y: tuple[float, ...]
    spread_x: float
    spread_y: float

    @property
    def spread(self) -> float:
        return max(self.spread_x, self.spread_y)

    def to_json(self) -> dict[str, Any]:
        return {
            "fractionsX": list(self.fractions_x),
            "fractionsY": list(self.fractions_y),
            "spreadXPx": self.spread_x,
            "spreadYPx": self.spread_y,
            "spreadPx": self.spread,
        }


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    motion: Motion
    noise_floor: float
    copy_threshold: float
    samples: tuple[RoiSample, ...]
    spread: SubpixelSpread
    near_copies: bool
    warnings: tuple[str, ...] = field(default=())

    @property
    def effective_samples(self) -> int:
        return sum(1 for sample in self.samples if sample.status in ("reference", "accepted"))

    @property
    def rejected_frames(self) -> tuple[int, ...]:
        return tuple(sample.n for sample in self.samples if sample.status == "rejected")

    def to_json(self) -> dict[str, Any]:
        return {
            "motion": self.motion,
            "noiseFloor": self.noise_floor,
            "copyThreshold": self.copy_threshold,
            "effectiveSamples": self.effective_samples,
            "rejectedFrames": list(self.rejected_frames),
            "nearCopies": self.near_copies,
            "warnings": list(self.warnings),
            "spread": self.spread.to_json(),
            "samples": [sample.to_json() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class ReferenceCandidate:
    n: int
    sharpness: float
    saturated_fraction: float

    @property
    def clipped(self) -> bool:
        return self.saturated_fraction >= MAX_REFERENCE_SATURATION


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, REPORT_DECIMALS)


def motion_for(kind: RoiKind, box: RoiBox) -> Motion:
    if min(box.w, box.h) < MIN_REGISTRATION_SIDE:
        return "translation"
    return KIND_MOTIONS[kind]


def search_window(box: RoiBox, shape: tuple[int, ...], margin: float) -> RoiBox:
    height, width = shape[:2]
    pad_x, pad_y = round(box.w * margin / 2), round(box.h * margin / 2)
    left, top = max(0, box.x - pad_x), max(0, box.y - pad_y)
    right, bottom = min(width, box.x + box.w + pad_x), min(height, box.y + box.h + pad_y)
    return RoiBox(left, top, right - left, bottom - top)


@contextmanager
def single_cv_thread() -> Iterator[None]:
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


# --- Piso de ruido y copias del GOP ---


def haar_diagonal(patch: np.ndarray) -> np.ndarray:
    even = patch[: patch.shape[0] // 2 * 2, : patch.shape[1] // 2 * 2].astype(np.float64)
    return (even[0::2, 0::2] - even[0::2, 1::2] - even[1::2, 0::2] + even[1::2, 1::2]) / 2


def noise_floor(patch: np.ndarray) -> float:
    return float(np.median(np.abs(haar_diagonal(patch)))) / MAD_TO_SIGMA


def copy_threshold(floor: float, ratio: float) -> float:
    return max(ratio * floor, MIN_COPY_THRESHOLD)


def mean_abs_diff(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float64) - second.astype(np.float64))))


def group_copies(patches: Sequence[np.ndarray], threshold: float) -> tuple[int, ...]:
    groups: list[int] = []
    unique: np.ndarray | None = None
    for patch in patches:
        if unique is not None and mean_abs_diff(patch, unique) < threshold:
            groups.append(groups[-1])
            continue
        groups.append(groups[-1] + 1 if groups else 0)
        unique = patch
    return tuple(groups)


def representatives(groups: Sequence[int], reference_pos: int) -> dict[int, int]:
    chosen: dict[int, int] = {}
    for pos, group in enumerate(groups):
        chosen.setdefault(group, pos)
    chosen[groups[reference_pos]] = reference_pos
    return chosen


# --- Registro ---


def upsampled(patch: np.ndarray, factor: int) -> np.ndarray:
    size = (patch.shape[1] * factor, patch.shape[0] * factor)
    return cv2.resize(patch.astype(np.float32), size, interpolation=cv2.INTER_LANCZOS4)


@dataclass(frozen=True, slots=True, eq=False)
class RegistrationTarget:
    box: RoiBox
    window: RoiBox
    template: np.ndarray
    reference_window: np.ndarray


def registration_target(reference: np.ndarray, box: RoiBox, window: RoiBox, factor: int) -> RegistrationTarget:
    template = upsampled(box.crop(reference), factor)
    return RegistrationTarget(box, window, template, upsampled(window.crop(reference), factor))


def phase_shift(reference: np.ndarray, patch: np.ndarray) -> tuple[float, float]:
    hann = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
    # Con ventana de Hann, phaseCorrelate de OpenCV 4.13 escribe sobre sus entradas.
    (dx, dy), _ = cv2.phaseCorrelate(reference.copy(), patch.copy(), hann)
    return dx, dy


def roi_offset(target: RegistrationTarget, factor: int) -> tuple[float, float]:
    return float(factor * (target.box.x - target.window.x)), float(factor * (target.box.y - target.window.y))


def ecc_starts(target: RegistrationTarget, patch: np.ndarray, factor: int) -> tuple[tuple[float, float], ...]:
    # La ROI (plantilla) se busca dentro de la ventana del cuadro. La fase puede
    # dar un pico falso en texturas repetidas, asi que tambien se prueba sin mover.
    ox, oy = roi_offset(target, factor)
    dx, dy = phase_shift(target.reference_window, patch)
    return ((ox + dx, oy + dy), (ox, oy))


def initial_warp(motion: Motion, start: tuple[float, float]) -> np.ndarray:
    rows = 3 if motion == "homography" else 2
    warp = np.eye(3, dtype=np.float32)[:rows].copy()
    warp[0, 2], warp[1, 2] = start
    return warp


def ecc_criteria(settings: RegistrationSettings) -> tuple[int, int, float]:
    return (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, settings.iterations, settings.eps)


def ecc_from(
    target: RegistrationTarget,
    patch: np.ndarray,
    start: tuple[float, float],
    motion: Motion,
    settings: RegistrationSettings,
) -> tuple[float, np.ndarray] | None:
    warp = initial_warp(motion, start)
    try:
        ecc, warp = cv2.findTransformECC(
            target.template,
            patch,
            warp,
            ECC_MOTIONS[motion],
            ecc_criteria(settings),
            None,
            settings.gauss_filt_size,
        )
    except cv2.error:
        return None
    return float(ecc), warp


def run_ecc(
    target: RegistrationTarget, patch: np.ndarray, motion: Motion, settings: RegistrationSettings
) -> tuple[float, np.ndarray] | None:
    starts = ecc_starts(target, patch, settings.upsample)
    outcomes = [ecc_from(target, patch, start, motion, settings) for start in starts]
    converged = [outcome for outcome in outcomes if outcome is not None]
    return max(converged, key=lambda outcome: outcome[0], default=None)


def as_homogeneous(warp: np.ndarray) -> np.ndarray:
    full = warp.astype(np.float64)
    return full if full.shape[0] == 3 else np.vstack([full, [0.0, 0.0, 1.0]])


def upsample_map(factor: int) -> np.ndarray:
    # Centros de pixel de cv2.resize: x_ampliado = factor·x + (factor - 1) / 2.
    offset = (factor - 1) / 2
    return np.array([[factor, 0, offset], [0, factor, offset], [0, 0, 1]], dtype=np.float64)


def offset_map(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def frame_matrix(warp: np.ndarray, box: RoiBox, window: RoiBox, factor: int) -> np.ndarray:
    scale = upsample_map(factor)
    to_input = offset_map(window.x, window.y) @ np.linalg.inv(scale)
    matrix = to_input @ as_homogeneous(warp) @ scale @ offset_map(-box.x, -box.y)
    return matrix / matrix[2, 2]


def as_matrix(matrix: np.ndarray) -> Matrix:
    return tuple(tuple(float(value) for value in row) for row in matrix)  # type: ignore[return-value]


def identity_alignment() -> Alignment:
    return Alignment(1.0, as_matrix(np.eye(3)))


def align_patch(
    target: RegistrationTarget, patch: np.ndarray, motion: Motion, settings: RegistrationSettings
) -> Alignment:
    outcome = run_ecc(target, upsampled(patch, settings.upsample), motion, settings)
    if outcome is None:
        return Alignment(None, None)
    ecc, warp = outcome
    return Alignment(ecc, as_matrix(frame_matrix(warp, target.box, target.window, settings.upsample)))


def center_shift(matrix: Matrix, box: RoiBox) -> tuple[float, float]:
    cx, cy = box.center
    x, y, w = np.array(matrix) @ np.array([cx, cy, 1.0])
    return float(x / w - cx), float(y / w - cy)


# --- Dispersion subpixel ---


def fractional_parts(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(round(value - math.floor(value), REPORT_DECIMALS) % 1.0 for value in values)


def circular_spread(fractions: Sequence[float]) -> float:
    if not fractions:
        return 0.0
    angles = 2 * np.pi * np.asarray(fractions, dtype=np.float64)
    resultant = float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))
    if resultant <= 0.0:
        return MAX_CIRCULAR_SPREAD
    spread = math.sqrt(-2 * math.log(min(resultant, 1.0))) / (2 * math.pi)
    return min(spread, MAX_CIRCULAR_SPREAD)


def subpixel_spread(shifts: Sequence[tuple[float, float]]) -> SubpixelSpread:
    fractions_x = fractional_parts([dx for dx, _ in shifts])
    fractions_y = fractional_parts([dy for _, dy in shifts])
    return SubpixelSpread(fractions_x, fractions_y, circular_spread(fractions_x), circular_spread(fractions_y))


def is_near_copies(effective: int, spread: SubpixelSpread, settings: RegistrationSettings) -> bool:
    return effective < settings.min_effective or spread.spread < settings.near_copy_spread


# --- Orquestacion ---


def reference_position(frames: Sequence[RoiFrame], reference_n: int) -> int:
    for pos, frame in enumerate(frames):
        if frame.n == reference_n:
            return pos
    raise RoiRegistrationError(ROI_REFERENCE_MISSING, f"reference frame {reference_n} is not in the range")


def check_inputs(frames: Sequence[RoiFrame], reference_n: int, box: RoiBox) -> int:
    if not frames:
        raise RoiRegistrationError(ROI_NO_FRAMES, "the ROI range has no frames")
    pos = reference_position(frames, reference_n)
    if not box.fits(frames[pos].pixels.shape):
        raise RoiRegistrationError(ROI_OUTSIDE_FRAME, "the ROI box is outside the frame")
    return pos


def group_status(pos: int, reference_pos: int, is_representative: bool, accepted: bool) -> SampleStatus:
    if not accepted:
        return "rejected"
    if not is_representative:
        return "copy"
    return "reference" if pos == reference_pos else "accepted"


def build_sample(
    frame: RoiFrame, group: int, status: SampleStatus, alignment: Alignment, box: RoiBox
) -> RoiSample:
    matrix = None if status == "rejected" else alignment.matrix
    shift = None if matrix is None else center_shift(matrix, box)
    return RoiSample(frame.n, frame.pict_type, group, status, alignment.ecc, shift, matrix)


def align_group(
    pos: int,
    reference_pos: int,
    patches: Sequence[np.ndarray],
    target: RegistrationTarget,
    motion: Motion,
    settings: RegistrationSettings,
) -> Alignment:
    if pos == reference_pos:
        return identity_alignment()
    return align_patch(target, patches[pos], motion, settings)


def align_groups(
    patches: Sequence[np.ndarray],
    chosen: dict[int, int],
    reference_pos: int,
    target: RegistrationTarget,
    motion: Motion,
    settings: RegistrationSettings,
) -> dict[int, Alignment]:
    return {
        group: align_group(pos, reference_pos, patches, target, motion, settings)
        for group, pos in chosen.items()
    }


@dataclass(frozen=True, slots=True, eq=False)
class GroupedAlignment:
    groups: tuple[int, ...]
    chosen: dict[int, int]
    alignments: dict[int, Alignment]
    reference_pos: int


def sample_at(pos: int, frame: RoiFrame, grouped: GroupedAlignment, box: RoiBox, ecc_min: float) -> RoiSample:
    group = grouped.groups[pos]
    alignment = grouped.alignments[group]
    is_representative = grouped.chosen[group] == pos
    status = group_status(pos, grouped.reference_pos, is_representative, alignment.accepted(ecc_min))
    return build_sample(frame, group, status, alignment, box)


def effective_shifts(samples: Sequence[RoiSample]) -> list[tuple[float, float]]:
    return [
        sample.shift
        for sample in samples
        if sample.status in ("reference", "accepted") and sample.shift is not None
    ]


def align_roi_frames(
    frames: Sequence[RoiFrame], reference_pos: int, box: RoiBox, motion: Motion, settings: RegistrationSettings
) -> tuple[float, float, GroupedAlignment]:
    reference = frames[reference_pos].pixels
    window = search_window(box, reference.shape, settings.window_margin)
    floor = noise_floor(box.crop(reference))
    threshold = copy_threshold(floor, settings.copy_ratio)
    patches = [window.crop(frame.pixels).astype(np.float32) for frame in frames]
    groups = group_copies(patches, threshold)
    chosen = representatives(groups, reference_pos)
    target = registration_target(reference, box, window, settings.upsample)
    alignments = align_groups(patches, chosen, reference_pos, target, motion, settings)
    return floor, threshold, GroupedAlignment(groups, chosen, alignments, reference_pos)


def register_roi_frames(
    frames: Sequence[RoiFrame],
    reference_n: int,
    box: RoiBox,
    motion: Motion,
    settings: RegistrationSettings = RegistrationSettings(),
) -> RegistrationResult:
    reference_pos = check_inputs(frames, reference_n, box)
    with single_cv_thread():
        floor, threshold, grouped = align_roi_frames(frames, reference_pos, box, motion, settings)
    samples = tuple(sample_at(pos, frame, grouped, box, settings.ecc_min) for pos, frame in enumerate(frames))
    shifts = effective_shifts(samples)
    spread = subpixel_spread(shifts)
    near = is_near_copies(len(shifts), spread, settings)
    return RegistrationResult(motion, floor, threshold, samples, spread, near, (NEAR_COPIES,) if near else ())


# --- Cuadro de referencia sugerido ---


def saturated_fraction(patch: np.ndarray) -> float:
    return float(np.mean(patch >= SATURATION_LEVEL))


def laplacian_variance(patch: np.ndarray) -> float:
    return float(cv2.Laplacian(patch.astype(np.float64), cv2.CV_64F).var())


def reference_candidate(frame: RoiFrame, box: RoiBox) -> ReferenceCandidate:
    patch = box.crop(frame.pixels)
    return ReferenceCandidate(frame.n, laplacian_variance(patch), saturated_fraction(patch))


def reference_candidates(frames: Sequence[RoiFrame], box: RoiBox) -> tuple[ReferenceCandidate, ...]:
    with single_cv_thread():
        candidates = [reference_candidate(frame, box) for frame in frames]
    return tuple(sorted(candidates, key=lambda c: (c.clipped, -c.sharpness, c.n)))


def suggest_reference_frame(frames: Sequence[RoiFrame], box: RoiBox) -> int:
    if not frames:
        raise RoiRegistrationError(ROI_NO_FRAMES, "the ROI range has no frames")
    if not box.fits(frames[0].pixels.shape):
        raise RoiRegistrationError(ROI_OUTSIDE_FRAME, "the ROI box is outside the frame")
    return reference_candidates(frames, box)[0].n
