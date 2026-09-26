from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from app.services.photo_dsp import LUMA_WEIGHTS
from app.services.photo_geometry import (
    MIN_PERSPECTIVE_SIDE_PX,
    Geometry,
    Quad,
    order_corners,
    rotate_quarter_turns,
)

# Umbrales de P4-CAPTURE, todos [propuesta]: se recalibran en P4-BENCH con escaneos
# y fotos de celular reales.
CAPTURE_ANALYSIS_SIDE = 800
BORDER_STRIP_FRACTION = 0.02
BORDER_STRIP_MIN_PX = 2
BACKGROUND_MIN_DISTANCE = 0.08
BACKGROUND_MIN_SHARE = 0.5
MIN_PHOTO_FRACTION = 0.02
FULL_FRAME_FRACTION = 0.95
CLOSE_KERNEL_FRACTION = 0.01
EDGE_INSET_FRACTION = 0.004
EDGE_INSET_MIN_PX = 3
MIN_TILT_DEG = 0.2
KEYSTONE_MIN_DEG = 3.0
QUAD_APPROX_FRACTION = 0.02
MIN_PRINT_FRACTION = 0.2
MAX_SPLIT_PHOTOS = 12
ROW_OVERLAP = 0.5
GLARE_LEVEL = 0.97
GLARE_MIN_BLOB_FRACTION = 0.0002
GLARE_MIN_FRACTION = 0.003
GLARE_HALO_PX = 3
GLARE_MIN_HALO_LUMA = 0.8


@dataclass(frozen=True, slots=True)
class SheetRegion:
    # Rectangulo girado minimo y contorno de 4 lados, en coordenadas de borde de la imagen analizada.
    rect: Quad
    outline: Quad | None
    area_fraction: float

    @property
    def keystone_deg(self) -> float:
        return 0.0 if self.outline is None else keystone_deg(self.outline)

    @property
    def center(self) -> tuple[float, float]:
        points = np.asarray(self.rect)
        return float(points[:, 0].mean()), float(points[:, 1].mean())

    @property
    def height(self) -> float:
        points = np.asarray(self.rect)
        return float(points[:, 1].max() - points[:, 1].min())


@dataclass(frozen=True, slots=True)
class CaptureSuggestions:
    auto_crop: Geometry | None = None
    photos: tuple[Geometry, ...] = ()
    perspective: Geometry | None = None
    # Alto y ancho de la foto girada: el marco de las esquinas y de los recortes sugeridos.
    frame: tuple[int, int] = (0, 0)


@dataclass(frozen=True, slots=True)
class CaptureSigns:
    perspective: bool
    glare: bool
    glare_fraction: float

    @property
    def phone_capture(self) -> bool:
        return self.perspective or self.glare


# Se reduce antes de girar: girar la foto entera obligaria a copiar un escaneo de 24 Mpx en float.
def suggest_capture(rgb: np.ndarray, rotate90: int) -> CaptureSuggestions:
    small, factor = fit_within(rgb, CAPTURE_ANALYSIS_SIDE)
    shape = rotated_shape(rgb.shape[:2], rotate90)
    regions = regions_in(np.ascontiguousarray(rotate_quarter_turns(small, rotate90)), factor, shape)
    return CaptureSuggestions(
        auto_crop=auto_crop_geometry(regions, shape, rotate90),
        photos=split_geometries(regions, shape, rotate90),
        perspective=perspective_geometry(regions, rotate90),
        frame=(int(shape[0]), int(shape[1])),
    )


def capture_signs(rgb: np.ndarray) -> CaptureSigns:
    small, _ = fit_within(rgb, CAPTURE_ANALYSIS_SIDE)
    regions = find_regions(small)
    glare = glare_fraction(small)
    return CaptureSigns(
        perspective=any(is_keystoned_print(region) for region in regions),
        glare=glare >= GLARE_MIN_FRACTION,
        glare_fraction=glare,
    )


def rotated_shape(shape: tuple[int, ...], rotate90: int) -> tuple[int, int]:
    height, width = int(shape[0]), int(shape[1])
    return (width, height) if rotate90 % 2 else (height, width)


def find_regions(rgb: np.ndarray) -> tuple[SheetRegion, ...]:
    small, factor = fit_within(rgb, CAPTURE_ANALYSIS_SIDE)
    return regions_in(small, factor, rgb.shape[:2])


def regions_in(small: np.ndarray, factor: float, shape: tuple[int, int]) -> tuple[SheetRegion, ...]:
    mask = foreground_mask(small)
    if mask is None:
        return ()
    total = float(mask.shape[0] * mask.shape[1])
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [
        region_from_contour(contour, factor, shape, total)
        for contour in contours
        if cv2.contourArea(contour) >= MIN_PHOTO_FRACTION * total
    ]
    return reading_order(regions)


def fit_within(rgb: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    factor = min(1.0, side / max(rgb.shape[:2]))
    if factor >= 1.0:
        return np.ascontiguousarray(rgb, dtype=np.float32), 1.0
    size = (max(1, round(rgb.shape[1] * factor)), max(1, round(rgb.shape[0] * factor)))
    small = cv2.resize(np.ascontiguousarray(rgb, dtype=np.float32), size, interpolation=cv2.INTER_AREA)
    return small, factor


def foreground_mask(small: np.ndarray) -> np.ndarray | None:
    blurred = cv2.GaussianBlur(small, (0, 0), 1.0)
    background = uniform_background(border_pixels(blurred))
    if background is None:
        return None
    distance = np.linalg.norm(blurred - background, axis=2)
    mask = (distance > BACKGROUND_MIN_DISTANCE).astype(np.uint8) * 255
    return filled(cleaned(mask))


def border_pixels(rgb: np.ndarray) -> np.ndarray:
    strip = max(BORDER_STRIP_MIN_PX, round(min(rgb.shape[:2]) * BORDER_STRIP_FRACTION))
    edges = (rgb[:strip], rgb[-strip:], rgb[:, :strip], rgb[:, -strip:])
    return np.concatenate([edge.reshape(-1, 3) for edge in edges])


# La tapa del escaner es pareja; si el borde no lo es, la foto llena el cuadro y no hay nada que recortar.
def uniform_background(border: np.ndarray) -> np.ndarray | None:
    background = np.median(border, axis=0)
    close = np.linalg.norm(border - background, axis=1) <= BACKGROUND_MIN_DISTANCE / 2
    return background if close.mean() >= BACKGROUND_MIN_SHARE else None


def cleaned(mask: np.ndarray) -> np.ndarray:
    side = max(3, round(max(mask.shape) * CLOSE_KERNEL_FRACTION) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


def filled(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask)
    cv2.drawContours(solid, contours, -1, 255, thickness=cv2.FILLED)
    return solid


def region_from_contour(
    contour: np.ndarray, factor: float, shape: tuple[int, int], total: float
) -> SheetRegion:
    (cx, cy), (width, height), angle = cv2.minAreaRect(contour)
    # El contorno pasa por centros de pixel: el borde real esta medio pixel mas afuera.
    box = cv2.boxPoints(((cx + 0.5, cy + 0.5), (width + 1.0, height + 1.0), angle))
    return SheetRegion(
        rect=scaled_quad(box, factor, shape),
        outline=outline_quad(contour, factor, shape),
        area_fraction=float(cv2.contourArea(contour)) / total,
    )


def outline_quad(contour: np.ndarray, factor: float, shape: tuple[int, int]) -> Quad | None:
    hull = cv2.convexHull(contour)
    approx = cv2.approxPolyDP(hull, QUAD_APPROX_FRACTION * cv2.arcLength(hull, True), True)
    if len(approx) != 4:
        return None
    return scaled_quad(approx.reshape(4, 2).astype(np.float64) + 0.5, factor, shape)


def scaled_quad(points: np.ndarray, factor: float, shape: tuple[int, int]) -> Quad:
    height, width = shape
    scaled = np.asarray(points, dtype=np.float64) / factor
    clipped = np.column_stack((np.clip(scaled[:, 0], 0, width), np.clip(scaled[:, 1], 0, height)))
    return order_corners(tuple((float(x), float(y)) for x, y in clipped))  # type: ignore[arg-type]


def reading_order(regions: Sequence[SheetRegion]) -> tuple[SheetRegion, ...]:
    rows: list[list[SheetRegion]] = []
    for region in sorted(regions, key=lambda item: item.center[1]):
        if rows and region.center[1] - rows[-1][0].center[1] < ROW_OVERLAP * rows[-1][0].height:
            rows[-1].append(region)
        else:
            rows.append([region])
    return tuple(region for row in rows for region in sorted(row, key=lambda item: item.center[0]))


def keystone_deg(quad: Quad) -> float:
    points = np.asarray(quad, dtype=np.float64)
    previous = np.roll(points, 1, axis=0) - points
    following = np.roll(points, -1, axis=0) - points
    cosines = (previous * following).sum(axis=1) / (
        np.linalg.norm(previous, axis=1) * np.linalg.norm(following, axis=1)
    )
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return float(np.abs(angles - 90.0).max())


def is_keystoned_print(region: SheetRegion) -> bool:
    return region.area_fraction >= MIN_PRINT_FRACTION and region.keystone_deg >= KEYSTONE_MIN_DEG


def auto_crop_geometry(
    regions: Sequence[SheetRegion], shape: tuple[int, int], rotate90: int
) -> Geometry | None:
    if len(regions) != 1 or is_keystoned_print(regions[0]):
        return None
    if regions[0].area_fraction >= FULL_FRAME_FRACTION:
        return None
    return leveled_crop(regions[0].rect, shape, rotate90)


def split_geometries(
    regions: Sequence[SheetRegion], shape: tuple[int, int], rotate90: int
) -> tuple[Geometry, ...]:
    if len(regions) < 2:
        return ()
    largest = sorted(regions, key=lambda region: region.area_fraction, reverse=True)[:MAX_SPLIT_PHOTOS]
    kept = [region for region in regions if region in largest]
    geometries = (leveled_crop(region.rect, shape, rotate90) for region in kept)
    return tuple(geometry for geometry in geometries if geometry is not None)


def perspective_geometry(regions: Sequence[SheetRegion], rotate90: int) -> Geometry | None:
    if len(regions) != 1 or not is_keystoned_print(regions[0]):
        return None
    try:
        return Geometry(rotate90=rotate90, corners=regions[0].outline)
    except ValueError:
        return None


def leveled_crop(rect: Quad, shape: tuple[int, int], rotate90: int) -> Geometry | None:
    angle = level_angle(rect)
    top_left, top_right, bottom_right, bottom_left = rotated_points(rect, angle, shape)
    inset = max(EDGE_INSET_MIN_PX, EDGE_INSET_FRACTION * min(shape))
    left = max(0, math.ceil(max(top_left[0], bottom_left[0]) + inset))
    top = max(0, math.ceil(max(top_left[1], top_right[1]) + inset))
    right = min(shape[1], math.floor(min(top_right[0], bottom_right[0]) - inset))
    bottom = min(shape[0], math.floor(min(bottom_left[1], bottom_right[1]) - inset))
    if min(right - left, bottom - top) < MIN_PERSPECTIVE_SIDE_PX:
        return None
    return Geometry(rotate90=rotate90, crop=(left, top, right - left, bottom - top), angle=angle)


# cv2 gira en sentido antihorario con angulos positivos: un borde que baja hacia la
# derecha (pendiente positiva con y hacia abajo) se nivela girando su mismo angulo.
def level_angle(rect: Quad) -> float:
    (x0, y0), (x1, y1) = rect[0], rect[1]
    slope = math.degrees(math.atan2(y1 - y0, x1 - x0))
    angle = (slope + 45.0) % 90.0 - 45.0
    return 0.0 if abs(angle) < MIN_TILT_DEG else round(angle, 2)


def rotated_points(quad: Quad, angle: float, shape: tuple[int, int]) -> Quad:
    height, width = shape
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    centers = np.asarray(quad, dtype=np.float64) - 0.5
    moved = centers @ matrix[:, :2].T + matrix[:, 2] + 0.5
    return order_corners(tuple((float(x), float(y)) for x, y in moved))  # type: ignore[arg-type]


def glare_fraction(rgb: np.ndarray) -> float:
    clipped = (rgb.min(axis=2) >= GLARE_LEVEL).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clipped, connectivity=8)
    luma = np.ascontiguousarray(rgb, dtype=np.float32) @ LUMA_WEIGHTS
    total = float(clipped.size)
    glare = sum(
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, count)
        if is_glare_blob(labels, stats[label], label, luma, total)
    )
    return glare / total


def is_glare_blob(labels: np.ndarray, stats: np.ndarray, label: int, luma: np.ndarray, total: float) -> bool:
    x, y, width, height, area = (int(value) for value in stats[:5])
    if area < GLARE_MIN_BLOB_FRACTION * total or touches_border(x, y, width, height, labels.shape):
        return False
    return halo_luma(labels, (x, y, width, height), label, luma) >= GLARE_MIN_HALO_LUMA


def touches_border(x: int, y: int, width: int, height: int, shape: tuple[int, ...]) -> bool:
    return x == 0 or y == 0 or x + width >= shape[1] or y + height >= shape[0]


# Un reflejo se desvanece hacia afuera; un objeto blanco con borde neto tiene alrededor su fondo.
def halo_luma(labels: np.ndarray, box: tuple[int, int, int, int], label: int, luma: np.ndarray) -> float:
    x, y, width, height = box
    pad = GLARE_HALO_PX
    top, left = max(0, y - pad), max(0, x - pad)
    bottom, right = min(labels.shape[0], y + height + pad), min(labels.shape[1], x + width + pad)
    blob = (labels[top:bottom, left:right] == label).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))
    ring = cv2.dilate(blob, kernel).astype(bool) & ~blob.astype(bool)
    return float(luma[top:bottom, left:right][ring].mean()) if ring.any() else 0.0
