from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

NEUTRAL_AXIS_BANDS = 12
LEAST_CHROMATIC_FRACTION = 0.2
FIT_MAX_PIXELS = 1_000_000
FIX_FADED_DEFAULT_STRENGTH = 0.7
LEVELS_LOW_PERCENTILE = 0.5
LEVELS_HIGH_PERCENTILE = 99.5
LEVELS_BLEND = 0.5
MIN_LEVELS_SPAN = 1.0
GRAY_POINT_WINDOW = 5
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILES = (8, 8)
LOCAL_CONTRAST_BLEND = 0.5
LAB_L_MAX = 100.0
UINT16_MAX = 65535.0
CHANNEL_EPSILON = 1e-4

GrayPoint = tuple[int, int]


@dataclass(frozen=True)
class ChannelCurve:
    inputs: np.ndarray
    outputs: np.ndarray


@dataclass(frozen=True)
class ToneSettings:
    strength: float = FIX_FADED_DEFAULT_STRENGTH
    gray_point: GrayPoint | None = None
    keep_tone: bool = True
    neutral_gray: bool = False
    fix_faded: bool = False
    local_contrast: bool = False


def apply_tone(rgb: np.ndarray, settings: ToneSettings) -> np.ndarray:
    balanced = _render_tone(rgb, settings)
    return local_contrast(balanced) if settings.local_contrast else balanced


def _render_tone(rgb: np.ndarray, settings: ToneSettings) -> np.ndarray:
    if settings.neutral_gray:
        return neutral_gray(rgb)
    if settings.fix_faded:
        return fix_faded_colors(rgb, settings.strength, settings.gray_point)
    if settings.keep_tone:
        return keep_tone_levels(rgb)
    return rgb.copy()


def fix_faded_colors(
    rgb: np.ndarray,
    strength: float = FIX_FADED_DEFAULT_STRENGTH,
    gray_point: GrayPoint | None = None,
) -> np.ndarray:
    _require_unit_interval(strength, "Strength")
    balanced = apply_gray_point(rgb, gray_point) if gray_point is not None else rgb
    if strength == 0:
        return balanced.copy()
    curves = fit_neutral_axis_curves(balanced)
    return apply_channel_curves(balanced, curves, strength)


def apply_gray_point(rgb: np.ndarray, point: GrayPoint) -> np.ndarray:
    sample = gray_point_sample(rgb, point)
    gains = sample.mean() / np.maximum(sample, CHANNEL_EPSILON)
    return np.clip(rgb * gains.astype(np.float32), 0.0, 1.0)


def gray_point_sample(rgb: np.ndarray, point: GrayPoint) -> np.ndarray:
    x, y = point
    height, width = rgb.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Gray point {point} is outside the {width}x{height} image")
    radius = GRAY_POINT_WINDOW // 2
    window = rgb[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    return window.reshape(-1, 3).mean(axis=0)


def fit_neutral_axis_curves(rgb: np.ndarray) -> tuple[ChannelCurve, ChannelCurve, ChannelCurve]:
    pixels = _fit_sample(rgb)
    lab = _to_lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    band_means = _least_chromatic_band_means(pixels, lab)
    grays = band_means.mean(axis=1)
    return tuple(_monotone_curve(band_means[:, channel], grays) for channel in range(3))


def _fit_sample(rgb: np.ndarray) -> np.ndarray:
    pixels = rgb.reshape(-1, 3)
    stride = max(1, int(np.ceil(len(pixels) / FIT_MAX_PIXELS)))
    return pixels[::stride]


def _least_chromatic_band_means(pixels: np.ndarray, lab: np.ndarray) -> np.ndarray:
    lightness = lab[:, 0]
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    edges = np.quantile(lightness, np.linspace(0.0, 1.0, NEUTRAL_AXIS_BANDS + 1))
    bands = np.clip(np.searchsorted(edges, lightness, side="right") - 1, 0, NEUTRAL_AXIS_BANDS - 1)
    means = [_band_neutral_mean(pixels, chroma, bands == band) for band in range(NEUTRAL_AXIS_BANDS)]
    return np.array([mean for mean in means if mean is not None], dtype=np.float64)


def _band_neutral_mean(pixels: np.ndarray, chroma: np.ndarray, in_band: np.ndarray) -> np.ndarray | None:
    band_pixels, band_chroma = pixels[in_band], chroma[in_band]
    if len(band_pixels) == 0:
        return None
    keep = max(1, int(len(band_pixels) * LEAST_CHROMATIC_FRACTION))
    least_chromatic = np.argpartition(band_chroma, keep - 1)[:keep]
    return band_pixels[least_chromatic].mean(axis=0)


def _monotone_curve(channel_means: np.ndarray, grays: np.ndarray) -> ChannelCurve:
    order = np.argsort(channel_means, kind="stable")
    inputs = np.clip(channel_means[order], CHANNEL_EPSILON, 1.0 - CHANNEL_EPSILON)
    outputs = np.maximum.accumulate(np.clip(grays[order], 0.0, 1.0))
    distinct = np.concatenate(([True], np.diff(inputs) > CHANNEL_EPSILON))
    return ChannelCurve(
        inputs=np.concatenate(([0.0], inputs[distinct], [1.0])),
        outputs=np.concatenate(([0.0], outputs[distinct], [1.0])),
    )


def apply_channel_curves(
    rgb: np.ndarray,
    curves: tuple[ChannelCurve, ChannelCurve, ChannelCurve],
    strength: float,
) -> np.ndarray:
    curved = np.empty(rgb.shape, dtype=np.float32)
    for channel, curve in enumerate(curves):
        curved[..., channel] = np.interp(rgb[..., channel], curve.inputs, curve.outputs)
    return _blend(rgb, curved, strength)


def keep_tone_levels(rgb: np.ndarray, blend: float = LEVELS_BLEND) -> np.ndarray:
    lab = _to_lab(rgb)
    low, high = lightness_levels(lab[..., 0])
    stretched = (lab[..., 0] - low) * (LAB_L_MAX / (high - low))
    leveled = np.clip(stretched, 0.0, LAB_L_MAX).astype(np.float32)
    return _with_lightness(rgb, lab, _blend(lab[..., 0], leveled, blend))


def lightness_levels(lightness: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(lightness, [LEVELS_LOW_PERCENTILE, LEVELS_HIGH_PERCENTILE])
    if high - low < MIN_LEVELS_SPAN:
        return 0.0, LAB_L_MAX
    return float(low), float(high)


def neutral_gray(rgb: np.ndarray) -> np.ndarray:
    lab = _to_lab(rgb)
    lab[..., 1:] = 0.0
    gray = np.clip(_lab_to_rgb(lab), 0.0, 1.0)
    # Lab -> RGB of an achromatic color can differ per channel in the last float bit.
    return np.repeat(gray.mean(axis=-1, keepdims=True), 3, axis=-1)


def local_contrast(rgb: np.ndarray, blend: float = LOCAL_CONTRAST_BLEND) -> np.ndarray:
    lab = _to_lab(rgb)
    lightness_u16 = np.round(lab[..., 0] * (UINT16_MAX / LAB_L_MAX)).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILES)
    equalized = clahe.apply(lightness_u16).astype(np.float32) * (LAB_L_MAX / UINT16_MAX)
    return _with_lightness(rgb, lab, _blend(lab[..., 0], equalized, blend))


def _with_lightness(rgb: np.ndarray, lab: np.ndarray, lightness: np.ndarray) -> np.ndarray:
    relit = lab.copy()
    relit[..., 0] = lightness
    # Applied as a delta: the float Lab round trip alone drifts ~1e-3 even where L did not change.
    return np.clip(rgb + (_lab_to_rgb(relit) - _lab_to_rgb(lab)), 0.0, 1.0)


def _blend(original: np.ndarray, processed: np.ndarray, amount: float) -> np.ndarray:
    return (original + (processed - original) * np.float32(amount)).astype(np.float32)


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32), cv2.COLOR_Lab2RGB)


def _require_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
