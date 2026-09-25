from __future__ import annotations

import cv2
import numpy as np

MAX_STRAIGHTEN_DEG = 45.0

Crop = tuple[int, int, int, int]


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
