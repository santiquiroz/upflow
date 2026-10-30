from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from app.services.fill_faces import get_face_cascade
from app.services.person_detector import PersonBox

Role = Literal["subject", "background"]

# Calibrados contra 816 clics de 59 fotos de grado (2026-10-30): rol 90.6 %, 1 foto con un
# sujeto grande propuesto como fondo; con relativo 0.65-0.75 se mantiene en ≤ 2 fotos.
HEIGHT_WEIGHT = 0.7
SHARPNESS_WEIGHT = 0.1
CENTER_WEIGHT = 0.2
SUBJECT_RELATIVE = 0.7
SUBJECT_MIN = 0.35


def _clipped_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def _height_score(box: tuple[int, int, int, int], image_height: int) -> float:
    y1 = max(0, min(image_height, box[1]))
    y2 = max(0, min(image_height, box[3]))
    return min(max(0, y2 - y1) / image_height, 1.0)


def _center_score(box: tuple[int, int, int, int], image_width: int) -> float:
    x1 = max(0, min(image_width, box[0]))
    x2 = max(0, min(image_width, box[2]))
    center_x = (x1 + x2) / 2
    return max(
        0.0, 1.0 - abs(center_x - image_width / 2) / (image_width / 2)
    )


def _sharpness_score(
    laplacian: np.ndarray,
    box: tuple[int, int, int, int],
    image_sharpness: float,
) -> float:
    height, width = laplacian.shape[:2]
    x1, y1, x2, y2 = _clipped_box(box, width, height)
    if x2 <= x1 or y2 <= y1 or image_sharpness <= 0:
        return 0.0
    box_sharpness = float(np.var(laplacian[y1:y2, x1:x2]))
    return min(box_sharpness / image_sharpness, 1.0)


def role_scores(rgb: np.ndarray, people: list[PersonBox]) -> list[float]:
    if not people:
        return []

    image_height, image_width = rgb.shape[:2]
    if image_height == 0 or image_width == 0:
        return [0.0] * len(people)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    image_sharpness = float(np.var(laplacian))
    scores = []
    for person in people:
        scores.append(
            HEIGHT_WEIGHT * _height_score(person.box, image_height)
            + SHARPNESS_WEIGHT
            * _sharpness_score(laplacian, person.box, image_sharpness)
            + CENTER_WEIGHT * _center_score(person.box, image_width)
        )
    return scores


def classify_roles(rgb: np.ndarray, people: list[PersonBox]) -> list[Role]:
    scores = role_scores(rgb, people)
    if not scores:
        return []
    max_score = max(scores)
    return [
        "subject"
        if score >= SUBJECT_RELATIVE * max_score and score >= SUBJECT_MIN
        else "background"
        for score in scores
    ]


def face_box(
    rgb: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    image_height, image_width = rgb.shape[:2]
    x1, y1, x2, y2 = _clipped_box(box, image_width, image_height)
    upper_y2 = y1 + (y2 - y1) // 2
    if x2 <= x1 or upper_y2 <= y1:
        return None

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    region = gray[y1:upper_y2, x1:x2]
    detections = get_face_cascade().detectMultiScale(
        region, 1.1, 5, minSize=(20, 20)
    )
    if len(detections) == 0:
        return None

    x, y, width, height = max(detections, key=lambda detected: detected[2] * detected[3])
    return (
        int(x1 + x),
        int(y1 + y),
        int(x1 + x + width),
        int(y1 + y + height),
    )


def _center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) // 2), int((y1 + y2) // 2)


def _upper_quarter_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) // 2), int(y1 + (y2 - y1) // 8)


def anchor_point(
    box: tuple[int, int, int, int],
    face: tuple[int, int, int, int] | None,
) -> tuple[int, int]:
    if face is not None:
        return _center(face)
    return _upper_quarter_center(box)
