"""Fusion multi-cuadro de la ROI (spec §4.9 pasos 6 a 9).

Cada muestra efectiva (la referencia y los representantes aceptados de cada
grupo de copias) se deforma con Lanczos sobre la grilla xk de la ROI de
referencia y se combina por mediana o media recortada. Las copias del GOP y los
cuadros rechazados no entran: ni en la fusion ni en el mapa de acuerdo, que es
la MAD por pixel de esa misma pila. La densidad se mide en pixeles guardados
(la caja ya viene en esas coordenadas; con SAR 2 el ancho mostrado es el
doble). Todo es numpy + OpenCV en CPU con un hilo y no toca el disco.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

import cv2
import numpy as np

from app.models import RoiFusionMethod, RoiKind
from app.services.roi_registration import (
    NEAR_COPIES,
    SATURATION_LEVEL,
    Matrix,
    Motion,
    RegistrationResult,
    RoiBox,
    RoiFrame,
    RoiSample,
    single_cv_thread,
    upsample_map,
)

DENSITY_FACE = "cctv.roi.densityFace"
DENSITY_PLATE = "cctv.roi.densityPlate"
CLIPPED = "cctv.roi.clipped"
FRAMES_USED = "cctv.roi.framesUsed"

FACE_MIN_WIDTH_PX = 40
CLIPPED_ROI_SHARE = 0.2
# "Descarta el 20% de los extremos": la mitad de cada lado.
TRIMMED_SHARE = 0.2
# MAD (niveles de gris) que se pinta como desacuerdo total; fijo para que los mapas se comparen entre jobs.
AGREEMENT_FULL_SCALE = 16.0
MAX_LEVEL = 255.0
UINT16_PER_LEVEL = 257
PERCENT_DECIMALS = 1
EFFECTIVE_STATUSES = frozenset({"reference", "accepted"})

FUSED_NAME = "roi_fused_x{scale}.png"
REFERENCE_NAME = "roi_reference_nearest_x{scale}.png"
AGREEMENT_NAME = "roi_agreement.png"
STACK_NAME = "roi_stack.mkv"
SAMPLES_NAME = "roi_samples.csv"
SAMPLES_HEADER = ("frame", "pict_type", "copy_group", "status", "ecc", "shift_x", "shift_y")

DensityAxis = Literal["width", "height"]


@dataclass(frozen=True, slots=True)
class FusionSettings:
    scale: int = 2
    method: RoiFusionMethod = "median"
    trimmed_share: float = TRIMMED_SHARE
    agreement_full_scale: float = AGREEMENT_FULL_SCALE


@dataclass(frozen=True, slots=True)
class RoiNotice:
    key: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "params": dict(self.params)}


@dataclass(frozen=True, slots=True)
class RoiDensity:
    kind: RoiKind
    axis: DensityAxis
    stored_px: int
    display_px: int

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "axis": self.axis, "storedPx": self.stored_px, "displayPx": self.display_px}


@dataclass(frozen=True, slots=True, eq=False)
class RoiFusion:
    fused: np.ndarray
    reference_nearest: np.ndarray
    agreement: np.ndarray
    mad: np.ndarray
    stack: np.ndarray
    stack_frames: tuple[int, ...]


def output_names(scale: int) -> dict[str, str]:
    return {
        "fused": FUSED_NAME.format(scale=scale),
        "reference": REFERENCE_NAME.format(scale=scale),
        "agreement": AGREEMENT_NAME,
        "stack": STACK_NAME,
        "samples": SAMPLES_NAME,
    }


# --- Muestras que entran ---


def effective_pairs(frames: Sequence[RoiFrame], samples: Sequence[RoiSample]) -> tuple[tuple[RoiFrame, RoiSample], ...]:
    return tuple(
        (frame, sample)
        for frame, sample in zip(frames, samples, strict=True)
        if sample.status in EFFECTIVE_STATUSES and sample.matrix is not None
    )


def frames_used(samples: Sequence[RoiSample]) -> int:
    return sum(1 for sample in samples if sample.status != "rejected")


# --- Deformacion sobre la grilla xk ---


def grid_matrix(matrix: Matrix, box: RoiBox, scale: int) -> np.ndarray:
    # Grilla xk -> referencia (centros de pixel de cv2.resize) -> cuadro.
    to_reference = np.array([[1, 0, box.x], [0, 1, box.y], [0, 0, 1]], dtype=np.float64) @ np.linalg.inv(
        upsample_map(scale)
    )
    return np.array(matrix, dtype=np.float64) @ to_reference


def grid_size(box: RoiBox, scale: int) -> tuple[int, int]:
    return box.w * scale, box.h * scale


def warp_to_grid(pixels: np.ndarray, matrix: Matrix, box: RoiBox, scale: int, motion: Motion) -> np.ndarray:
    mapping = grid_matrix(matrix, box, scale)
    flags = cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP
    source = pixels.astype(np.float32)
    size = grid_size(box, scale)
    if motion == "homography":
        warped = cv2.warpPerspective(source, mapping, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
    else:
        warped = cv2.warpAffine(source, mapping[:2], size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(warped, 0.0, MAX_LEVEL)


def aligned_stack(
    pairs: Sequence[tuple[RoiFrame, RoiSample]], box: RoiBox, scale: int, motion: Motion
) -> np.ndarray:
    return np.stack([warp_to_grid(frame.pixels, sample.matrix, box, scale, motion) for frame, sample in pairs])


# --- Combinacion ---


def median_fuse(stack: np.ndarray) -> np.ndarray:
    return np.median(stack.astype(np.float64), axis=0)


def trim_per_side(count: int, share: float) -> int:
    return math.floor(count * share / 2)


def trimmed_mean_fuse(stack: np.ndarray, share: float = TRIMMED_SHARE) -> np.ndarray:
    ordered = np.sort(stack.astype(np.float64), axis=0)
    cut = trim_per_side(len(ordered), share)
    return ordered[cut : len(ordered) - cut].mean(axis=0)


def fuse_stack(stack: np.ndarray, settings: FusionSettings) -> np.ndarray:
    if settings.method == "trimmed_mean":
        return trimmed_mean_fuse(stack, settings.trimmed_share)
    return median_fuse(stack)


# --- Mapa de acuerdo ---


def agreement_mad(stack: np.ndarray) -> np.ndarray:
    values = stack.astype(np.float64)
    return np.median(np.abs(values - np.median(values, axis=0)), axis=0)


def agreement_image(mad: np.ndarray, full_scale: float = AGREEMENT_FULL_SCALE) -> np.ndarray:
    # Verde = los cuadros coinciden; amarillo a rojo = cada vez menos. Salida BGR de 8 bits.
    level = np.clip(mad / full_scale, 0.0, 1.0)
    red = np.minimum(1.0, 2.0 * level)
    green = np.minimum(1.0, 2.0 * (1.0 - level))
    bgr = np.stack([np.zeros_like(level), green, red], axis=-1)
    return np.round(bgr * MAX_LEVEL).astype(np.uint8)


# --- Referencia sin "mejorar" ---


def reference_nearest(reference: np.ndarray, box: RoiBox, scale: int) -> np.ndarray:
    patch = np.clip(np.round(box.crop(reference)), 0, MAX_LEVEL).astype(np.uint8)
    return cv2.resize(patch, grid_size(box, scale), interpolation=cv2.INTER_NEAREST)


def reference_pixels(frames: Sequence[RoiFrame], samples: Sequence[RoiSample]) -> np.ndarray:
    return next(frame.pixels for frame, sample in zip(frames, samples, strict=True) if sample.status == "reference")


def fuse_roi(
    frames: Sequence[RoiFrame], registration: RegistrationResult, box: RoiBox, settings: FusionSettings
) -> RoiFusion:
    pairs = effective_pairs(frames, registration.samples)
    with single_cv_thread():
        stack = aligned_stack(pairs, box, settings.scale, registration.motion)
        nearest = reference_nearest(reference_pixels(frames, registration.samples), box, settings.scale)
    mad = agreement_mad(stack)
    return RoiFusion(
        fused=fuse_stack(stack, settings),
        reference_nearest=nearest,
        agreement=agreement_image(mad, settings.agreement_full_scale),
        mad=mad,
        stack=stack,
        stack_frames=tuple(frame.n for frame, _ in pairs),
    )


# --- Codificacion (en memoria) ---


def to_uint16(image: np.ndarray) -> np.ndarray:
    return np.round(np.clip(image, 0.0, MAX_LEVEL) * UINT16_PER_LEVEL).astype(np.uint16)


def png_bytes(image: np.ndarray) -> bytes:
    # imencode y no imwrite: imwrite no abre rutas con caracteres fuera de ASCII en Windows.
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("could not encode the PNG")
    return encoded.tobytes()


def stack_gray16le(stack: np.ndarray) -> bytes:
    return to_uint16(stack).astype("<u2").tobytes()


def csv_number(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def sample_row(sample: RoiSample) -> list[str]:
    shift_x, shift_y = (None, None) if sample.shift is None else sample.shift
    return [
        str(sample.n),
        sample.pict_type,
        str(sample.copy_group),
        sample.status,
        csv_number(sample.ecc),
        csv_number(shift_x),
        csv_number(shift_y),
    ]


def samples_csv_text(samples: Sequence[RoiSample]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(SAMPLES_HEADER)
    writer.writerows(sample_row(sample) for sample in samples)
    return buffer.getvalue()


# --- Avisos ---


def roi_density(kind: RoiKind, box: RoiBox, sar: Fraction) -> RoiDensity:
    if kind == "face_or_object":
        return RoiDensity(kind, "width", box.w, round(box.w * sar))
    return RoiDensity(kind, "height", box.h, box.h)


def density_notices(density: RoiDensity) -> tuple[RoiNotice, ...]:
    if density.kind == "plate":
        return (RoiNotice(DENSITY_PLATE, {"px": density.stored_px}),)
    if density.stored_px < FACE_MIN_WIDTH_PX:
        return (RoiNotice(DENSITY_FACE, {"px": density.stored_px}),)
    return ()


def roi_saturated_share(pixels: np.ndarray, box: RoiBox) -> float:
    return float(np.mean(box.crop(pixels) >= SATURATION_LEVEL))


def clipped_frames_pct(frames: Sequence[RoiFrame], box: RoiBox) -> float:
    if not frames:
        return 0.0
    clipped = sum(1 for frame in frames if roi_saturated_share(frame.pixels, box) > CLIPPED_ROI_SHARE)
    return round(100.0 * clipped / len(frames), PERCENT_DECIMALS)


def clipped_notices(pct: float) -> tuple[RoiNotice, ...]:
    return (RoiNotice(CLIPPED, {"pct": pct}),) if pct > 0 else ()


def frames_used_notice(registration: RegistrationResult) -> RoiNotice:
    params = {
        "used": frames_used(registration.samples),
        "total": len(registration.samples),
        "effective": registration.effective_samples,
    }
    return RoiNotice(FRAMES_USED, params)


def near_copies_notices(registration: RegistrationResult) -> tuple[RoiNotice, ...]:
    return (RoiNotice(NEAR_COPIES),) if registration.near_copies else ()


def fusion_notices(registration: RegistrationResult, density: RoiDensity, clipped_pct: float) -> tuple[RoiNotice, ...]:
    return (
        frames_used_notice(registration),
        *near_copies_notices(registration),
        *density_notices(density),
        *clipped_notices(clipped_pct),
    )
