from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.config import Settings
from app.services.missing_pack import missing_pack_message

YOLOX_STRIDES = (8, 16, 32)


@dataclass(frozen=True)
class PersonBox:
    box: tuple[int, int, int, int]
    score: float


class PersonDetectorUnavailableError(RuntimeError):
    pass


def letterbox(rgb: np.ndarray, side: int = 640) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    ratio = min(side / height, side / width)
    resized = cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        (int(width * ratio), int(height * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((side, side, 3), 114, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    return tensor, ratio


def decode_yolox(raw: np.ndarray, side: int = 640) -> np.ndarray:
    grids = []
    strides = []
    for stride in YOLOX_STRIDES:
        grid_size = side // stride
        x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        grids.append(np.stack((x, y), axis=2).reshape(-1, 2))
        strides.append(np.full((grid_size * grid_size, 1), stride))

    grid = np.concatenate(grids).astype(np.float32)
    stride_values = np.concatenate(strides).astype(np.float32)
    decoded = np.asarray(raw).reshape(-1, raw.shape[-1]).copy()
    decoded[:, :2] = (decoded[:, :2] + grid) * stride_values
    decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * stride_values
    return decoded


def _corner_boxes_in_original(decoded_boxes: np.ndarray, ratio: float) -> np.ndarray:
    centers_x, centers_y, widths, heights = decoded_boxes.T
    return np.column_stack(
        (
            (centers_x - widths / 2) / ratio,
            (centers_y - heights / 2) / ratio,
            (centers_x + widths / 2) / ratio,
            (centers_y + heights / 2) / ratio,
        )
    )


def _nms_indices(
    boxes: np.ndarray, scores: list[float], threshold: float, iou: float
) -> list[int]:
    rectangles = [
        [
            float(box[0]),
            float(box[1]),
            float(box[2] - box[0]),
            float(box[3] - box[1]),
        ]
        for box in boxes
    ]
    kept = cv2.dnn.NMSBoxes(rectangles, scores, threshold, iou)
    if kept is None or len(kept) == 0:
        return []
    return [int(index) for index in np.asarray(kept).reshape(-1)]


def _clip_box(
    box: np.ndarray, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (
        int(np.rint(np.clip(value, 0, limit)))
        for value, limit in zip(box, (width, height, width, height))
    )
    return x1, y1, x2, y2


def select_people(
    decoded: np.ndarray,
    ratio: float,
    image_size: tuple[int, int],
    threshold: float = 0.3,
    iou: float = 0.45,
) -> list[PersonBox]:
    scores = decoded[:, 4] * decoded[:, 5]
    candidate_indices = np.flatnonzero(scores > threshold)
    if candidate_indices.size == 0:
        return []

    scaled_boxes = _corner_boxes_in_original(decoded[candidate_indices, :4], ratio)
    candidate_scores = scores[candidate_indices].tolist()
    kept_indices = _nms_indices(scaled_boxes, candidate_scores, threshold, iou)
    if not kept_indices:
        return []

    width, height = image_size
    people = []
    for selected in kept_indices:
        box = _clip_box(scaled_boxes[selected], width, height)
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        people.append(
            PersonBox(
                box=box,
                score=float(candidate_scores[selected]),
            )
        )
    return sorted(people, key=lambda person: person.box[0])


class PersonDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._sessions: dict[str, Any] = {}
        self._session_locks: dict[str, threading.Lock] = {}

    def available(self) -> bool:
        return self.settings.person_detector_available()

    def detect(self, rgb: np.ndarray, device: str = "cpu") -> list[PersonBox]:
        if not self.available():
            raise PersonDetectorUnavailableError(missing_pack_message("person-detector"))

        session = self._get_session(device)
        tensor, ratio = letterbox(rgb)
        raw = session.run(None, {"images": tensor})[0][0]
        decoded = decode_yolox(raw)
        return select_people(decoded, ratio, (rgb.shape[1], rgb.shape[0]))

    def _get_session(self, device: str) -> Any:
        with self._lock:
            session = self._sessions.get(device)
            if session is not None:
                return session
            session_lock = self._session_locks.setdefault(device, threading.Lock())

        with session_lock:
            with self._lock:
                session = self._sessions.get(device)
                if session is not None:
                    return session

            session = self._create_session(device)
            with self._lock:
                self._sessions[device] = session
            return session

    def _create_session(self, device: str) -> Any:
        from app.services import ep_registry

        return ep_registry.create_session(
            str(self.settings.person_detector_model_path), device, self.settings
        )
