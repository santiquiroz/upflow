# Adapted from xinntao/facexlib@260620ae93990a300f4b16448df9bb459f1caba9 (MIT, © 2020 Xintao Wang): FFHQ-512 template, similarity alignment with LMEDS, gray border and template-mask paste of face_restoration_helper.py
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

FACE_SIZE = 512
TEMPLATE_FFHQ_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float64,
)
LANDMARK_COUNT = 5
ALIGN_BORDER_RGB = (135, 133, 132)
U8_PEAK = 255.0
U16_PER_U8 = 257
MASK_EDGE_DIVISOR = 20
MASK_EDGE_PX = FACE_SIZE // MASK_EDGE_DIVISOR
PASTE_MARGIN_PX = 1
# Piel, ojos y boca de la plantilla FFHQ: sin fondo, pelo ni el borde gris del warp.
FACE_CORE_BOX = (128, 160, 384, 448)
CONTRAST_EPSILON = 1e-8

# Umbrales de §2.4 [propuesta]; P1-ID y P4-BENCH los validan.
RESTORE_MIN_EYE_PX = 32.0
SMALL_MIN_EYE_PX = 16.0
FORCEABLE_MIN_EYE_PX = 8.0
# 4 caras de dominio publico: nitidas 0,08..0,24; reducidas 2× y vueltas a su tamaño 0,035..0,053.
SHARP_FACE_MIN = 0.06

Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class FacePolicy:
    tier: str
    enabled: bool
    blend: float | None
    selectable: bool
    needs_confirmation: bool
    label_key: str


FACE_POLICY_TEXTS = {
    "restore.face.restore": "AI-restored face",
    "restore.face.alreadyClear": "Already clear — restoring may change it",
    "restore.face.small": "Small face — most detail will be invented",
    "restore.face.tooSmallFaithful": "Too small to restore faithfully",
    "restore.face.tooSmall": "Too small",
}


def align_matrix(landmarks: np.ndarray) -> np.ndarray:
    points = _landmark_array(landmarks)
    matrix, _ = cv2.estimateAffinePartial2D(points, TEMPLATE_FFHQ_512, method=cv2.LMEDS)
    if matrix is None or not np.all(np.isfinite(matrix)) or _is_singular(matrix):
        raise ValueError("Face landmarks are degenerate: no similarity maps them onto the template")
    return matrix


def align_face(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(
        image,
        matrix,
        (FACE_SIZE, FACE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=align_border_value(image.dtype),
    )


def align_border_value(dtype: np.dtype) -> tuple[float, ...]:
    if np.dtype(dtype) == np.uint8:
        return tuple(float(v) for v in ALIGN_BORDER_RGB)
    if np.dtype(dtype) == np.uint16:
        return tuple(float(v * U16_PER_U8) for v in ALIGN_BORDER_RGB)
    return tuple(v / U8_PEAK for v in ALIGN_BORDER_RGB)


def inverse_paste_matrix(matrix: np.ndarray, scale: float) -> np.ndarray:
    if not scale > 0:
        raise ValueError(f"Paste scale must be positive, got {scale}")
    inverse = cv2.invertAffineTransform(matrix) * scale
    # facexlib suma 0,5·s; el centro de pixel de un reescalado s× cae en s·p + 0,5·(s − 1).
    inverse[:, 2] += 0.5 * (scale - 1)
    return inverse


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:, :2].T + matrix[:, 2]


def upscaled_points(points: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) * scale + 0.5 * (scale - 1)


def paste_box(inverse: np.ndarray, out_shape: tuple[int, int], face_size: int = FACE_SIZE) -> Box:
    corners = np.array([[-0.5, -0.5], [face_size - 0.5, -0.5], [-0.5, face_size - 0.5], [face_size - 0.5] * 2])
    reached = transform_points(corners, inverse)
    height, width = out_shape
    x0 = _clip(math.floor(reached[:, 0].min()) - PASTE_MARGIN_PX, width)
    y0 = _clip(math.floor(reached[:, 1].min()) - PASTE_MARGIN_PX, height)
    x1 = _clip(math.ceil(reached[:, 0].max()) + PASTE_MARGIN_PX + 1, width)
    y1 = _clip(math.ceil(reached[:, 1].max()) + PASTE_MARGIN_PX + 1, height)
    return x0, y0, x1, y1


def warp_to_box(crop: np.ndarray, inverse: np.ndarray, box: Box) -> np.ndarray:
    x0, y0, x1, y1 = box
    shifted = inverse.copy()
    shifted[:, 2] -= (x0, y0)
    return cv2.warpAffine(
        crop,
        shifted,
        (max(0, x1 - x0), max(0, y1 - y0)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def template_mask(face_size: int = FACE_SIZE) -> np.ndarray:
    edge = face_size // MASK_EDGE_DIVISOR
    mask = np.zeros((face_size, face_size), dtype=np.float32)
    mask[edge : face_size - edge, edge : face_size - edge] = 1.0
    kernel = 2 * edge + 1
    return cv2.GaussianBlur(mask, (kernel, kernel), 0, borderType=cv2.BORDER_CONSTANT)


def eye_distance(landmarks: np.ndarray) -> float:
    points = np.asarray(landmarks, dtype=np.float64)
    return float(np.hypot(*(points[1] - points[0])))


def face_sharpness(crop: np.ndarray) -> float:
    luma = _core_luma(crop)
    contrast = float(luma.var())
    if contrast < CONTRAST_EPSILON:
        return 0.0
    return float(cv2.Laplacian(luma, cv2.CV_32F).var()) / contrast


def default_face_policy(eye_px: float, sharpness: float) -> FacePolicy:
    _require_measure("eye distance", eye_px)
    _require_measure("face sharpness", sharpness)
    if eye_px >= RESTORE_MIN_EYE_PX:
        return _large_face_policy(sharpness)
    if eye_px >= SMALL_MIN_EYE_PX:
        return _policy("small", blend=0.5)
    if eye_px >= FORCEABLE_MIN_EYE_PX:
        return _policy("tooSmallFaithful", blend=0.4, needs_confirmation=True)
    return _policy("tooSmall", blend=None, selectable=False)


def _large_face_policy(sharpness: float) -> FacePolicy:
    if sharpness >= SHARP_FACE_MIN:
        return _policy("alreadyClear", blend=0.4)
    return _policy("restore", blend=0.6, enabled=True)


def _policy(
    tier: str,
    *,
    blend: float | None,
    enabled: bool = False,
    selectable: bool = True,
    needs_confirmation: bool = False,
) -> FacePolicy:
    return FacePolicy(
        tier=tier,
        enabled=enabled,
        blend=blend,
        selectable=selectable,
        needs_confirmation=needs_confirmation,
        label_key=f"restore.face.{tier}",
    )


def _require_measure(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number, got {value}")


def _landmark_array(landmarks: np.ndarray) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64).reshape(-1, 2)
    if points.shape != (LANDMARK_COUNT, 2):
        raise ValueError(f"Face alignment needs 5 landmarks, got {points.shape[0]}")
    return points


def _is_singular(matrix: np.ndarray) -> bool:
    return abs(float(np.linalg.det(matrix[:, :2]))) < 1e-12


def _clip(value: int, limit: int) -> int:
    return max(0, min(limit, value))


def _core_luma(crop: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = (round(v * crop.shape[0] / FACE_SIZE) for v in FACE_CORE_BOX)
    core = _unit_float(crop[y0:y1, x0:x1])
    if core.ndim == 3 and core.shape[2] >= 3:
        return cv2.cvtColor(np.ascontiguousarray(core[:, :, :3]), cv2.COLOR_RGB2GRAY)
    return np.ascontiguousarray(core.reshape(core.shape[0], core.shape[1]))


def _unit_float(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    return image.astype(np.float32, copy=False)
