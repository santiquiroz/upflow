from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

MAX_STRAIGHTEN_DEG = 45.0

Crop = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Geometry:
    rotate90: int = 0
    crop: Crop | None = None
    angle: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.angle) or abs(self.angle) > MAX_STRAIGHTEN_DEG:
            raise ValueError(f"Straighten angle must be within ±{MAX_STRAIGHTEN_DEG}°")
        if self.crop is not None and len(self.crop) != 4:
            raise ValueError(f"Crop must be (x, y, width, height), got {self.crop}")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> Geometry:
        values = dict(raw or {})
        crop = values.get("crop")
        return cls(
            rotate90=int(values.get("rotate90", 0)) % 4,
            crop=None if crop is None else tuple(int(value) for value in crop),
            angle=float(values.get("angle", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"rotate90": self.rotate90, "crop": None if self.crop is None else list(self.crop), "angle": self.angle}

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        return apply_geometry(rgb, self.rotate90, self.crop, self.angle)

    def output_size(self, height: int, width: int) -> tuple[int, int]:
        rotated = (width, height) if self.rotate90 % 2 else (height, width)
        if self.crop is None:
            return rotated
        x, y, crop_width, crop_height = self.crop
        inside = x >= 0 and y >= 0 and x + crop_width <= rotated[1] and y + crop_height <= rotated[0]
        if crop_width <= 0 or crop_height <= 0 or not inside:
            raise ValueError(f"Crop {self.crop} is outside the {rotated[1]}x{rotated[0]} image")
        return crop_height, crop_width


def apply_geometry(
    rgb: np.ndarray,
    rotate90: int = 0,
    crop: Crop | None = None,
    angle_deg: float = 0.0,
) -> np.ndarray:
    rotated = rotate_quarter_turns(rgb, rotate90)
    straightened = straighten(rotated, angle_deg)
    cropped = crop_box(straightened, crop) if crop is not None else straightened
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
