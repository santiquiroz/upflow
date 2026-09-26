from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

MAX_STRAIGHTEN_DEG = 45.0
MIN_PERSPECTIVE_SIDE_PX = 16
# Tolerancia para esquinas que caen justo en el borde por redondeo de la UI.
CORNER_EDGE_TOLERANCE_PX = 0.5

Crop = tuple[int, int, int, int]
Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


@dataclass(frozen=True, slots=True)
class Geometry:
    rotate90: int = 0
    crop: Crop | None = None
    angle: float = 0.0
    corners: Quad | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.angle) or abs(self.angle) > MAX_STRAIGHTEN_DEG:
            raise ValueError(f"Straighten angle must be within ±{MAX_STRAIGHTEN_DEG}°")
        if self.crop is not None and len(self.crop) != 4:
            raise ValueError(f"Crop must be (x, y, width, height), got {self.crop}")
        if self.corners is not None:
            object.__setattr__(self, "corners", checked_quad(self.corners, self.angle))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> Geometry:
        values = dict(raw or {})
        crop = values.get("crop")
        corners = values.get("corners")
        return cls(
            rotate90=int(values.get("rotate90", 0)) % 4,
            crop=None if crop is None else tuple(int(value) for value in crop),
            angle=float(values.get("angle", 0.0)),
            corners=None if corners is None else quad_from_points(corners),
        )

    # Sin perspectiva no se escribe "corners": sidecars y jobs anteriores siguen iguales.
    def to_dict(self) -> dict[str, Any]:
        framing = {"rotate90": self.rotate90, "crop": None if self.crop is None else list(self.crop), "angle": self.angle}
        return framing if self.corners is None else {**framing, "corners": [list(point) for point in self.corners]}

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return apply_geometry(rgb, self.rotate90, self.crop, self.angle, self.corners)

    def output_size(self, height: int, width: int) -> tuple[int, int]:
        rotated = (width, height) if self.rotate90 % 2 else (height, width)
        framed = rotated if self.corners is None else rectified_size(self.corners, rotated)
        if self.crop is None:
            return framed
        x, y, crop_width, crop_height = self.crop
        inside = x >= 0 and y >= 0 and x + crop_width <= framed[1] and y + crop_height <= framed[0]
        if crop_width <= 0 or crop_height <= 0 or not inside:
            raise ValueError(f"Crop {self.crop} is outside the {framed[1]}x{framed[0]} image")
        return crop_height, crop_width


def apply_geometry(
    rgb: np.ndarray,
    rotate90: int = 0,
    crop: Crop | None = None,
    angle_deg: float = 0.0,
    corners: Quad | None = None,
) -> np.ndarray:
    rotated = rotate_quarter_turns(rgb, rotate90)
    straightened = straighten(rotated, angle_deg)
    framed = straightened if corners is None else rectify_perspective(straightened, corners)
    cropped = crop_box(framed, crop) if crop is not None else framed
    return _owned_contiguous(cropped, rgb)


def rotate_quarter_turns(rgb: np.ndarray, turns_clockwise: int) -> np.ndarray:
    turns = turns_clockwise % 4
    return rgb if turns == 0 else np.rot90(rgb, k=-turns)


def straighten(rgb: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) > MAX_STRAIGHTEN_DEG:
        raise ValueError(f"Straighten angle must be within ±{MAX_STRAIGHTEN_DEG}°")
    if angle_deg == 0:
        return rgb
    height, width = rgb.shape[:2]
    center = ((width - 1) / 2, (height - 1) / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    warped = cv2.warpAffine(
        np.ascontiguousarray(rgb),
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )
    # Lanczos overshoots at edges; the working copy must stay in [0, 1].
    return np.clip(warped, 0.0, 1.0)


def _owned_contiguous(result: np.ndarray, source: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(result)
    return contiguous.copy() if np.may_share_memory(contiguous, source) else contiguous


def crop_box(rgb: np.ndarray, crop: Crop) -> np.ndarray:
    x, y, width, height = crop
    image_height, image_width = rgb.shape[:2]
    inside = x >= 0 and y >= 0 and x + width <= image_width and y + height <= image_height
    if width <= 0 or height <= 0 or not inside:
        raise ValueError(f"Crop {crop} is outside the {image_width}x{image_height} image")
    return rgb[y : y + height, x : x + width]


def quad_from_points(points: Sequence[Sequence[float]]) -> Quad:
    if len(points) != 4 or any(len(point) != 2 for point in points):
        raise ValueError("Perspective needs exactly 4 corners given as (x, y)")
    quad = tuple((float(point[0]), float(point[1])) for point in points)
    if not all(math.isfinite(value) for point in quad for value in point):
        raise ValueError("Perspective corners must be finite numbers")
    return quad  # type: ignore[return-value]


def checked_quad(corners: Sequence[Sequence[float]], angle: float) -> Quad:
    if angle != 0:
        raise ValueError("Perspective and straighten can't be combined: the 4 corners already set the rotation")
    quad = order_corners(quad_from_points(corners))
    if not is_convex(quad):
        raise ValueError("Perspective corners must form a convex shape")
    return quad


def order_corners(quad: Quad) -> Quad:
    points = np.asarray(quad, dtype=np.float64)
    center = points.mean(axis=0)
    # Con y hacia abajo, el angulo creciente recorre la imagen en sentido horario.
    clockwise = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    start = int(np.argmin(clockwise.sum(axis=1)))
    ordered = np.roll(clockwise, -start, axis=0)
    return tuple((float(x), float(y)) for x, y in ordered)  # type: ignore[return-value]


def is_convex(quad: Quad) -> bool:
    points = np.asarray(quad, dtype=np.float64)
    edges = np.roll(points, -1, axis=0) - points
    following = np.roll(edges, -1, axis=0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    return bool(np.all(turns > 0) or np.all(turns < 0))


def rectified_size(corners: Quad, image_size: tuple[int, int]) -> tuple[int, int]:
    _require_inside(corners, image_size)
    top_left, top_right, bottom_right, bottom_left = (np.asarray(point) for point in corners)
    width = round(max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left)))
    height = round(max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right)))
    if min(width, height) < MIN_PERSPECTIVE_SIDE_PX:
        raise ValueError(f"Perspective corners enclose less than {MIN_PERSPECTIVE_SIDE_PX} px per side")
    return int(height), int(width)


def _require_inside(corners: Quad, image_size: tuple[int, int]) -> None:
    height, width = image_size
    tolerance = CORNER_EDGE_TOLERANCE_PX
    for x, y in corners:
        if not (-tolerance <= x <= width + tolerance and -tolerance <= y <= height + tolerance):
            raise ValueError(f"Perspective corner ({x:g}, {y:g}) is outside the {width}x{height} image")


# Las esquinas van en coordenadas de borde (el pixel i cubre [i, i+1)), como el recorte;
# cv2 trabaja con centros de pixel, de ahi el corrimiento de medio pixel.
def rectify_perspective(rgb: np.ndarray, corners: Quad) -> np.ndarray:
    height, width = rectified_size(corners, rgb.shape[:2])
    source = np.asarray(corners, dtype=np.float32) - 0.5
    target = np.asarray([(0, 0), (width, 0), (width, height), (0, height)], dtype=np.float32) - 0.5
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(
        np.ascontiguousarray(rgb),
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )
    return np.clip(warped, 0.0, 1.0)
