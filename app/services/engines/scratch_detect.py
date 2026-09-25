from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from scipy.special import expit

from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.engines.tiled_restore_runner import TileInfer
from app.services.inpaint_mask import dilate_mask
from app.services.photo_diagnosis import DamageDetector
from app.services.photo_dsp import LUMA_WEIGHTS

SCRATCH_MODEL_ID = "bopbtl-scratch-detector"
DETECTOR_DEVICE = "cpu"
DETECTOR_PRECISION = "fp32"
DETECTOR_SHORT_SIDE = 256
DETECTOR_MULTIPLE = 16
# "Detection sensitivity" 0..1 recorre el umbral de 0,6 a 0,2; 0,5 da el 0,4 de BOPBTL (§3.4.2).
LEAST_SENSITIVE_THRESHOLD = 0.6
MOST_SENSITIVE_THRESHOLD = 0.2
DEFAULT_SENSITIVITY = 0.5
MAX_GROW_PX = 3
U8_PEAK = 255.0


def detector_input_size(width: int, height: int) -> tuple[int, int]:
    # Global/detection.py (scale_256): lado corto a 256 y cada lado redondeado a multiplo de 16.
    if width < height:
        return _rounded(DETECTOR_SHORT_SIDE), _rounded(height / width * DETECTOR_SHORT_SIDE)
    return _rounded(width / height * DETECTOR_SHORT_SIDE), _rounded(DETECTOR_SHORT_SIDE)


def detector_input(image: np.ndarray) -> np.ndarray:
    gray = luma(image)
    size = detector_input_size(gray.shape[1], gray.shape[0])
    # PIL filtra con soporte ampliado al reducir, igual que el Image.resize BICUBIC del entrenamiento.
    resized = np.asarray(Image.fromarray(gray).resize(size, Image.Resampling.BICUBIC), dtype=np.float32)
    return np.clip(resized, 0.0, 1.0)


def luma(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.ascontiguousarray(image, dtype=np.float32)
    if image.shape[2] == 1:
        return np.ascontiguousarray(image[:, :, 0], dtype=np.float32)
    return np.ascontiguousarray(image, dtype=np.float32) @ LUMA_WEIGHTS


def scratch_probability(image: np.ndarray, infer: TileInfer) -> np.ndarray:
    logits = infer(detector_input(image)[:, :, np.newaxis])[:, :, 0]
    probability = expit(logits.astype(np.float32, copy=False))
    height, width = image.shape[:2]
    native = cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(native, 0.0, 1.0).astype(np.float32, copy=False)


def sensitivity_threshold(sensitivity: float) -> float:
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError(f"Detection sensitivity must be within [0, 1], got {sensitivity}")
    span = LEAST_SENSITIVE_THRESHOLD - MOST_SENSITIVE_THRESHOLD
    return LEAST_SENSITIVE_THRESHOLD - sensitivity * span


def damage_mask(probability: np.ndarray, sensitivity: float = DEFAULT_SENSITIVITY, grow_px: int = 0) -> np.ndarray:
    return grow_mask(probability > sensitivity_threshold(sensitivity), grow_px)


def grow_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if abs(pixels) > MAX_GROW_PX:
        raise ValueError(f"Grow / shrink must be within ±{MAX_GROW_PX} px, got {pixels}")
    if pixels > 0:
        return dilate_mask(_to_u8(mask), pixels) > 0
    if pixels < 0:
        return _shrink(mask, -pixels)
    return mask


def probability_to_u8(probability: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(probability, 0.0, 1.0) * U8_PEAK).astype(np.uint8)


def scratch_detector(engine: PhotoRestoreEngine) -> DamageDetector:
    # Siempre en CPU: el analisis y el job dan la misma mascara y DML no recompila por cada forma.
    def detect(image: np.ndarray) -> np.ndarray:
        engine.begin_phase(DETECTOR_DEVICE)
        infer = engine.tile_infer(SCRATCH_MODEL_ID, DETECTOR_DEVICE, DETECTOR_PRECISION, clamp=False)
        return scratch_probability(image, infer)

    return detect


def _rounded(value: float) -> int:
    return max(DETECTOR_MULTIPLE, int(round(value / DETECTOR_MULTIPLE)) * DETECTOR_MULTIPLE)


def _to_u8(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, np.uint8(255), np.uint8(0))


def _shrink(mask: np.ndarray, pixels: int) -> np.ndarray:
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), dtype=np.uint8)
    return cv2.erode(_to_u8(mask), kernel) > 0

