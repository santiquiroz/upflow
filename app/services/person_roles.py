from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from app.services.fill_faces import _get_cascade
from app.services.person_detector import PersonBox

Role = Literal["subject", "background"]

HEIGHT_WEIGHT = 0.5
SHARPNESS_WEIGHT = 0.3
CENTER_WEIGHT = 0.2
SUBJECT_RELATIVE = 0.6
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
        x1, y1, x2, y2 = _clipped_box(
            person.box, image_width, image_height
        )
        box_height = max(0, y2 - y1)
        height_score = min(box_height / image_height, 1.0)
        center_x = (x1 + x2) / 2
        center_score = max(
            0.0, 1.0 - abs(center_x - image_width / 2) / (image_width / 2)
        )
        if x2 > x1 and y2 > y1 and image_sharpness > 0:
            box_sharpness = float(np.var(laplacian[y1:y2, x1:x2]))
            sharpness_score = min(box_sharpness / image_sharpness, 1.0)
        else:
            sharpness_score = 0.0
        scores.append(
            HEIGHT_WEIGHT * height_score
            + SHARPNESS_WEIGHT * sharpness_score
            + CENTER_WEIGHT * center_score
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
    detections = _get_cascade().detectMultiScale(
        region, 1.1, 5, minSize=(20, 20)
    )
    if len(detections) == 0:
        return None

    x, y, width, height = max(detections, key=lambda detected: detected[2] * detected[3])
    return x1 + x, y1 + y, x1 + x + width, y1 + y + height


def anchor_point(
    box: tuple[int, int, int, int],
    face: tuple[int, int, int, int] | None,
) -> tuple[int, int]:
    if face is not None:
        x1, y1, x2, y2 = face
    else:
        x1, y1, x2, y2 = box
        return (x1 + x2) // 2, y1 + (y2 - y1) // 8
    return (x1 + x2) // 2, (y1 + y2) // 2
