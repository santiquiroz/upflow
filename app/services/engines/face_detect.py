# Adapted from yakhyo/retinaface-pytorch@7601e1cba94213b07d4b26a5b9c0a5341078bda4 (MIT, © 2024 Yakhyokhuja Valikhujaev): cfg_re34 priors, box and landmark decoding, NMS and the BGR mean of detect.py
# Adapted from biubug6/Pytorch_Retinaface@b984b4b775b2c4dced95c1eadd195a5c7d32a60b (MIT, © 2019 biubug6): the prior box and decoding scheme yakhyo builds on
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.services.engines.photo_restore_engine import PhotoRestoreEngine, guard_finite
from app.services.face_geometry import eye_distance
from app.services.photo_diagnosis import DetectedFace, FaceDetector

RETINAFACE_MODEL_ID = "retinaface-r34"
DETECT_DEVICE = "cpu"
DETECT_PRECISION = "fp32"
DETECT_MAX_SIDE = 1280
MIN_SIZES = ((16, 32), (64, 128), (256, 512))
STEPS = (8, 16, 32)
CENTER_VARIANCE = 0.1
SIZE_VARIANCE = 0.2
BGR_MEAN = np.array([104.0, 117.0, 123.0], dtype=np.float32)
SCORE_THRESHOLD = 0.9
NMS_IOU = 0.4
PRE_NMS_TOP_K = 5000
OUTPUT_NAMES = ("loc", "conf", "landmarks")
FACE_CLASS = 1
U8_PEAK = 255.0

RetinaOutputs = tuple[np.ndarray, np.ndarray, np.ndarray]
RetinaRun = Callable[[np.ndarray], RetinaOutputs]


@dataclass(frozen=True, slots=True)
class FaceDetection:
    box: tuple[float, float, float, float]
    score: float
    landmarks: tuple[tuple[float, float], ...]

    @property
    def eye_px(self) -> float:
        return eye_distance(np.asarray(self.landmarks))

    def scaled(self, factor: float) -> FaceDetection:
        return FaceDetection(
            box=tuple(value * factor for value in self.box),
            score=self.score,
            landmarks=tuple((x * factor, y * factor) for x, y in self.landmarks),
        )

    def to_detected_face(self) -> DetectedFace:
        return DetectedFace(box=self.box, score=self.score, eye_px=self.eye_px)


def priors(height: int, width: int) -> np.ndarray:
    levels = [_level_priors(height, width, step, sizes) for step, sizes in zip(STEPS, MIN_SIZES, strict=True)]
    return np.concatenate(levels).astype(np.float32)


def decode_boxes(loc: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    centers = anchors[:, :2] + loc[:, :2] * CENTER_VARIANCE * anchors[:, 2:]
    sizes = anchors[:, 2:] * np.exp(loc[:, 2:] * SIZE_VARIANCE)
    return np.concatenate([centers - sizes / 2, centers + sizes / 2], axis=1)


def decode_landmarks(landmarks: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    offsets = landmarks.reshape(-1, 5, 2)
    return anchors[:, None, :2] + offsets * CENTER_VARIANCE * anchors[:, None, 2:]


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> list[int]:
    order = np.argsort(scores, kind="stable")[::-1]
    keep: list[int] = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(int(best))
        order = rest[_overlaps(boxes[best], boxes[rest]) <= iou]
    return keep


def network_input(rgb: np.ndarray) -> np.ndarray:
    bgr = _u8_scale(rgb)[:, :, 2::-1] - BGR_MEAN
    return np.ascontiguousarray(bgr.transpose(2, 0, 1)[None])


def decode_detections(
    outputs: RetinaOutputs, height: int, width: int, threshold: float = SCORE_THRESHOLD
) -> tuple[FaceDetection, ...]:
    loc, conf, landmarks = (np.asarray(output, dtype=np.float32)[0] for output in outputs)
    candidates = _top_candidates(conf[:, FACE_CLASS], threshold)
    anchors = priors(height, width)[candidates]
    size = np.array([width, height], dtype=np.float32)
    boxes = decode_boxes(loc[candidates], anchors) * np.tile(size, 2)
    points = decode_landmarks(landmarks[candidates], anchors) * size
    scores = conf[candidates, FACE_CLASS]
    return tuple(_detection(boxes[i], scores[i], points[i]) for i in nms(boxes, scores))


def detect_faces(
    image: np.ndarray, run: RetinaRun, max_side: int = DETECT_MAX_SIDE, threshold: float = SCORE_THRESHOLD
) -> tuple[FaceDetection, ...]:
    small, factor = fit_within(image, max_side)
    found = decode_detections(run(network_input(small)), small.shape[0], small.shape[1], threshold)
    return tuple(face.scaled(1.0 / factor) for face in found)


def fit_within(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    factor = min(1.0, max_side / max(image.shape[:2]))
    if factor == 1.0:
        return image, 1.0
    size = (max(1, round(image.shape[1] * factor)), max(1, round(image.shape[0] * factor)))
    return cv2.resize(np.ascontiguousarray(image), size, interpolation=cv2.INTER_AREA), factor


def session_run(session: Any) -> RetinaRun:
    input_name = session.get_inputs()[0].name

    def run(batch: np.ndarray) -> RetinaOutputs:
        outputs = session.run(list(OUTPUT_NAMES), {input_name: batch})
        loc, conf, landmarks = (guard_finite(np.asarray(o), RETINAFACE_MODEL_ID, DETECT_PRECISION) for o in outputs)
        return loc, conf, landmarks

    return run


def retinaface_run(engine: PhotoRestoreEngine) -> RetinaRun:
    # Siempre en CPU (§3.4.7): el analisis y el job ven las mismas caras y DML no recompila por forma.
    engine.begin_phase(DETECT_DEVICE)
    return session_run(engine.session(RETINAFACE_MODEL_ID, DETECT_DEVICE, DETECT_PRECISION))


def landmarked_face_detector(engine: PhotoRestoreEngine) -> Callable[[np.ndarray], tuple[FaceDetection, ...]]:
    def detect(image: np.ndarray) -> tuple[FaceDetection, ...]:
        return detect_faces(image, retinaface_run(engine))

    return detect


def face_detector(engine: PhotoRestoreEngine) -> FaceDetector:
    detect_landmarked = landmarked_face_detector(engine)

    def detect(image: np.ndarray) -> Sequence[DetectedFace]:
        return [face.to_detected_face() for face in detect_landmarked(image)]

    return detect


def _level_priors(height: int, width: int, step: int, min_sizes: Sequence[int]) -> np.ndarray:
    rows, cols = math.ceil(height / step), math.ceil(width / step)
    cy, cx = np.meshgrid(
        (np.arange(rows) + 0.5) * step / height, (np.arange(cols) + 0.5) * step / width, indexing="ij"
    )
    centers = np.stack([cx, cy], axis=-1).reshape(-1, 1, 2)
    sizes = np.array([[size / width, size / height] for size in min_sizes])[None]
    centers, sizes = np.broadcast_arrays(centers, sizes)
    return np.concatenate([centers, sizes], axis=-1).reshape(-1, 4)


def _top_candidates(scores: np.ndarray, threshold: float) -> np.ndarray:
    chosen = np.flatnonzero(scores >= threshold)
    ranked = chosen[np.argsort(scores[chosen], kind="stable")[::-1]]
    return ranked[:PRE_NMS_TOP_K]


def _overlaps(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    top_left = np.maximum(box[:2], others[:, :2])
    bottom_right = np.minimum(box[2:], others[:, 2:])
    inter = np.prod(np.maximum(0.0, bottom_right - top_left + 1), axis=1)
    area = np.prod(box[2:] - box[:2] + 1)
    areas = np.prod(others[:, 2:] - others[:, :2] + 1, axis=1)
    return inter / (area + areas - inter)


def _detection(box: np.ndarray, score: float, points: np.ndarray) -> FaceDetection:
    return FaceDetection(
        box=tuple(float(v) for v in box),
        score=float(score),
        landmarks=tuple((float(x), float(y)) for x, y in points),
    )


def _u8_scale(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Face detection needs an RGB image, got shape {rgb.shape}")
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32)
    if rgb.dtype == np.uint16:
        return rgb.astype(np.float32) / 257.0
    return rgb.astype(np.float32) * U8_PEAK
