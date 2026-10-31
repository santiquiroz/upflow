from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import cv2
import numpy as np

from app.services.fill_faces import get_face_cascade
from app.services.person_detector import PersonBox

Role = Literal["subject", "background"]
FaceBox = tuple[int, int, int, int]
FaceFinder = Callable[[np.ndarray, FaceBox], list[FaceBox]]

# Calibrados contra 816 clics de 59 fotos de grado (2026-10-30): rol 90.6 %, 1 foto con un
# sujeto grande propuesto como fondo; con relativo 0.65-0.75 se mantiene en ≤ 2 fotos.
HEIGHT_WEIGHT = 0.7
SHARPNESS_WEIGHT = 0.1
CENTER_WEIGHT = 0.2
SUBJECT_RELATIVE = 0.7
SUBJECT_MIN = 0.35

# Una cara solo es de una persona si su centro cae en la mitad central de la caja
# (|cx - box_cx| <= 0.3 * box_w) y el punto esperado de la cabeza sale del 15 % superior.
FACE_WINDOW_HALF_WIDTH = 0.3
FACE_HEAD_Y = 0.15
FACE_DUPLICATE_IOU = 0.5

# Dos cajas son la misma persona si la pequeña cabe dentro de la grande, con la misma
# cabeza y el mismo centro horizontal (un hombre detectado dos veces).
DUPLICATE_CONTAINMENT = 0.8
DUPLICATE_TOP_TOLERANCE = 0.05
DUPLICATE_CENTER_TOLERANCE = 0.15


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
    box_sharpness = float(np.var(laplacian[y1:y2, x1:x2], dtype=np.float64))
    return min(box_sharpness / image_sharpness, 1.0)


def role_scores(rgb: np.ndarray, people: list[PersonBox]) -> list[float]:
    if not people:
        return []

    image_height, image_width = rgb.shape[:2]
    if image_height == 0 or image_width == 0:
        return [0.0] * len(people)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_16S)
    image_sharpness = float(np.var(laplacian, dtype=np.float64))
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


def _haar_faces(rgb: np.ndarray, box: FaceBox) -> list[FaceBox]:
    image_height, image_width = rgb.shape[:2]
    x1, y1, x2, y2 = _clipped_box(box, image_width, image_height)
    upper_y2 = y1 + (y2 - y1) // 2
    if x2 <= x1 or upper_y2 <= y1:
        return []

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    region = gray[y1:upper_y2, x1:x2]
    detections = get_face_cascade().detectMultiScale(
        region, 1.1, 5, minSize=(20, 20)
    )
    return [
        (int(x1 + x), int(y1 + y), int(x1 + x + width), int(y1 + y + height))
        for x, y, width, height in detections
    ]


def face_box(
    rgb: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    faces = _haar_faces(rgb, box)
    if not faces:
        return None
    return max(faces, key=_area)


def assign_faces(
    rgb: np.ndarray,
    boxes: list[FaceBox],
    find_faces: FaceFinder = _haar_faces,
) -> list[FaceBox | None]:
    pool = _unique_faces([face for box in boxes for face in find_faces(rgb, box)])
    assigned: list[FaceBox | None] = [None] * len(boxes)
    taken: set[FaceBox] = set()
    for index in sorted(range(len(boxes)), key=lambda item: _area(boxes[item])):
        face = _pick_face(boxes[index], pool, taken)
        assigned[index] = face
        if face is not None:
            taken.add(face)
    return assigned


def _unique_faces(faces: list[FaceBox]) -> list[FaceBox]:
    unique: list[FaceBox] = []
    for face in faces:
        if all(_iou(face, kept) <= FACE_DUPLICATE_IOU for kept in unique):
            unique.append(face)
    return unique


def _pick_face(
    box: FaceBox, pool: list[FaceBox], taken: set[FaceBox]
) -> FaceBox | None:
    candidates = [
        face for face in pool if face not in taken and _face_belongs_to(face, box)
    ]
    if not candidates:
        return None
    head_x, head_y = _box_center_x(box), _head_y(box)
    return min(
        candidates,
        key=lambda face: _distance(_face_center(face), (head_x, head_y)),
    )


def _face_belongs_to(face: FaceBox, box: FaceBox) -> bool:
    face_x, face_y = _face_center(face)
    x1, y1, x2, y2 = box
    upper_y2 = y1 + (y2 - y1) // 2
    return (
        y1 <= face_y < upper_y2
        and abs(face_x - _box_center_x(box)) <= FACE_WINDOW_HALF_WIDTH * (x2 - x1)
    )


def _head_y(box: FaceBox) -> float:
    return box[1] + FACE_HEAD_Y * (box[3] - box[1])


def _face_center(face: FaceBox) -> tuple[float, float]:
    return (face[0] + face[2]) / 2, (face[1] + face[3]) / 2


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(np.hypot(first[0] - second[0], first[1] - second[1]))


def merge_duplicates(people: list[PersonBox]) -> list[PersonBox]:
    order = sorted(range(len(people)), key=lambda index: -_area(people[index].box))
    scores = [person.score for person in people]
    kept: list[int] = []
    for index in order:
        owner = next(
            (
                candidate
                for candidate in kept
                if _is_same_person(people[candidate].box, people[index].box)
            ),
            None,
        )
        if owner is None:
            kept.append(index)
        else:
            scores[owner] = max(scores[owner], scores[index])
    return [
        replace(people[index], score=scores[index]) for index in sorted(kept)
    ]


def _is_same_person(owner: FaceBox, duplicate: FaceBox) -> bool:
    duplicate_area = _area(duplicate)
    if duplicate_area == 0:
        return False
    containment = _intersection_area(owner, duplicate) / duplicate_area
    return (
        containment >= DUPLICATE_CONTAINMENT
        and abs(owner[1] - duplicate[1]) <= DUPLICATE_TOP_TOLERANCE * _height(owner)
        and abs(_box_center_x(owner) - _box_center_x(duplicate))
        <= DUPLICATE_CENTER_TOLERANCE * _width(owner)
    )


def _area(box: FaceBox) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _width(box: FaceBox) -> int:
    return max(0, box[2] - box[0])


def _height(box: FaceBox) -> int:
    return max(0, box[3] - box[1])


def _box_center_x(box: FaceBox) -> float:
    return (box[0] + box[2]) / 2


def _intersection_area(first: FaceBox, second: FaceBox) -> int:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    return max(0, width) * max(0, height)


def _iou(first: FaceBox, second: FaceBox) -> float:
    intersection = _intersection_area(first, second)
    union = _area(first) + _area(second) - intersection
    return intersection / union if union > 0 else 0.0


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
