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

    boxes = decoded[candidate_indices, :4]
    centers_x, centers_y, widths, heights = boxes.T
    scaled_boxes = np.column_stack(
        (
            (centers_x - widths / 2) / ratio,
            (centers_y - heights / 2) / ratio,
            (centers_x + widths / 2) / ratio,
            (centers_y + heights / 2) / ratio,
        )
    )
    rectangles = [
        [
            float(box[0]),
            float(box[1]),
            float(box[2] - box[0]),
            float(box[3] - box[1]),
        ]
        for box in scaled_boxes
    ]
    candidate_scores = scores[candidate_indices].tolist()
    kept = cv2.dnn.NMSBoxes(rectangles, candidate_scores, threshold, iou)
    if len(kept) == 0:
        return []

    width, height = image_size
    people = []
    for selected in np.asarray(kept).reshape(-1):
        box = scaled_boxes[int(selected)]
        x1 = int(np.rint(np.clip(box[0], 0, width)))
        y1 = int(np.rint(np.clip(box[1], 0, height)))
        x2 = int(np.rint(np.clip(box[2], 0, width)))
        y2 = int(np.rint(np.clip(box[3], 0, height)))
        people.append(
            PersonBox(
                box=(x1, y1, x2, y2),
                score=float(candidate_scores[int(selected)]),
            )
        )
    return sorted(people, key=lambda person: person.box[0])


class PersonDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._sessions: dict[str, Any] = {}

    def available(self) -> bool:
        return self.settings.person_detector_available()

    def detect(self, rgb: np.ndarray, device: str = "cpu") -> list[PersonBox]:
        if not self.available():
            raise PersonDetectorUnavailableError(missing_pack_message("person-detector"))

        with self._lock:
            session = self._sessions.get(device)
            if session is None:
                session = self._create_session(device)
                self._sessions[device] = session

        tensor, ratio = letterbox(rgb)
        raw = session.run(None, {"images": tensor})[0][0]
        decoded = decode_yolox(raw)
        return select_people(decoded, ratio, (rgb.shape[1], rgb.shape[0]))

    def _create_session(self, device: str) -> Any:
        from app.services import ep_registry

        return ep_registry.create_session(
            str(self.settings.person_detector_model_path), device, self.settings
        )
