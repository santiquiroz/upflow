from __future__ import annotations

from collections.abc import Callable

import numpy as np

from app.services.person_detector import PersonBox, PersonDetector

PERSON_MIN_SCORE = 0.3
PERSON_INSIDE_RATIO = 0.6

PeopleFinder = Callable[[np.ndarray, np.ndarray], list[tuple[int, int, int, int]]]


def people_inside(
    boxes: list[PersonBox],
    mask: np.ndarray,
    min_score: float,
    inside_ratio: float,
) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    found = []
    for person in boxes:
        if person.score < min_score:
            continue
        x1, y1, x2, y2 = person.box
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= 0:
            continue
        left, top = max(x1, 0), max(y1, 0)
        right, bottom = min(x2, width), min(y2, height)
        if right <= left or bottom <= top:
            continue
        inside_area = np.count_nonzero(mask[top:bottom, left:right] > 0)
        if inside_area / box_area >= inside_ratio:
            found.append(person.box)
    return found


def make_people_finder(
    detector: PersonDetector,
    device: str = "cpu",
    min_score: float = PERSON_MIN_SCORE,
    inside_ratio: float = PERSON_INSIDE_RATIO,
) -> PeopleFinder:
    def find_people(rgb: np.ndarray, mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        rows, columns = np.nonzero(mask)
        if rows.size == 0:
            return []

        height, width = rgb.shape[:2]
        box_width = int(columns.max()) + 1 - int(columns.min())
        box_height = int(rows.max()) + 1 - int(rows.min())
        pad_x = int(np.ceil(box_width * 0.25))
        pad_y = int(np.ceil(box_height * 0.25))
        left = max(0, int(columns.min()) - pad_x)
        top = max(0, int(rows.min()) - pad_y)
        right = min(width, int(columns.max()) + 1 + pad_x)
        bottom = min(height, int(rows.max()) + 1 + pad_y)

        detections = detector.detect(rgb[top:bottom, left:right], device=device)
        translated = []
        for person in detections:
            x1, y1, x2, y2 = person.box
            translated.append(
                PersonBox(
                    (x1 + left, y1 + top, x2 + left, y2 + top),
                    person.score,
                )
            )
        return people_inside(translated, mask, min_score, inside_ratio)

    return find_people
