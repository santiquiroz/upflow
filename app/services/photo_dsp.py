from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy import fft as sp_fft

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

PEAK_ANALYSIS_SIDE = 1024
PEAK_SIGMA_THRESHOLD = 3.5
MIN_PEAK_PROMINENCE = 1.5
PEAK_BACKGROUND_SIGMA_BINS = 4.0
MIN_PATTERN_FREQUENCY = 1.0 / 40.0
MAX_PERIODIC_PEAKS = 32
RAY_ANGLE_TOLERANCE_DEG = 2.0
HARMONIC_TOLERANCE = 0.1
LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], np.float32)
NOTCH_PAD_FRACTION = 0.1
NOTCH_DILATE_BINS = 2.0
NOTCH_EDGE_SIGMA_BINS = 1.0
NOTCH_PROTECT_FRACTION = 0.5
HALFTONE_SIGMA_PER_PERIOD = 0.4
HALFTONE_UNSHARP_SIGMA_PER_PERIOD = 1.0
HALFTONE_UNSHARP_AMOUNT = 0.3
ATROUS_LEVELS = 5
ATROUS_TAPS = (0.25, 0.5, 0.25)
BAND_ROWS = 1024

GrayPoint = tuple[int, int]
BandFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ChannelCurve:
    inputs: np.ndarray
    outputs: np.ndarray


@dataclass(frozen=True)
class PeriodicPeaks:
    frequencies: np.ndarray
    strengths: np.ndarray
    resolution: float


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


@dataclass(frozen=True)
class ToneBandPlan:
    settings: ToneSettings
    gains: np.ndarray | None = None
    curves: tuple[ChannelCurve, ChannelCurve, ChannelCurve] | None = None
    levels: tuple[float, float] | None = None
    contrast_lightness: np.ndarray | None = None


def apply_tone_in_bands(rgb: np.ndarray, settings: ToneSettings, band_rows: int = BAND_ROWS) -> np.ndarray:
    # Los parametros globales (curvas, niveles, CLAHE) se miden una vez y se aplican por franjas:
    # mismo resultado que apply_tone con el pico de RAM acotado a la salida mas una franja.
    _require_band_sizes(band_rows, 0)
    plan = _tone_band_plan(rgb, settings, band_rows)
    output = np.empty(rgb.shape, dtype=np.float32)
    for start, stop in _band_spans(rgb.shape[0], band_rows):
        output[start:stop] = _tone_band(rgb[start:stop], plan, start, stop)
    return output


def _tone_band_plan(rgb: np.ndarray, settings: ToneSettings, band_rows: int) -> ToneBandPlan:
    plan = _balance_plan(rgb, settings, band_rows)
    if not settings.local_contrast:
        return plan
    lightness = _banded_lightness(rgb, band_rows, lambda band: _balance_band(band, plan))
    return replace(plan, contrast_lightness=_clahe_lightness(lightness))


def _balance_plan(rgb: np.ndarray, settings: ToneSettings, band_rows: int) -> ToneBandPlan:
    if settings.neutral_gray:
        return ToneBandPlan(settings)
    if settings.fix_faded:
        return _faded_plan(rgb, settings)
    if settings.keep_tone:
        lightness = _banded_lightness(rgb, band_rows, lambda band: band)
        return ToneBandPlan(settings, levels=lightness_levels(lightness))
    return ToneBandPlan(settings)


def _faded_plan(rgb: np.ndarray, settings: ToneSettings) -> ToneBandPlan:
    _require_unit_interval(settings.strength, "Strength")
    gains = None if settings.gray_point is None else gray_point_gains(rgb, settings.gray_point)
    if settings.strength == 0:
        return ToneBandPlan(settings, gains=gains)
    sample = _fit_sample(rgb)
    balanced = sample if gains is None else _apply_gains(sample, gains)
    return ToneBandPlan(settings, gains=gains, curves=fit_neutral_axis_curves(balanced))


def _banded_lightness(rgb: np.ndarray, band_rows: int, balance: BandFunction) -> np.ndarray:
    lightness = np.empty(rgb.shape[:2], dtype=np.float32)
    for start, stop in _band_spans(rgb.shape[0], band_rows):
        lightness[start:stop] = _to_lab(balance(rgb[start:stop]))[..., 0]
    return lightness


def _tone_band(band: np.ndarray, plan: ToneBandPlan, start: int, stop: int) -> np.ndarray:
    balanced = _balance_band(band, plan)
    if plan.contrast_lightness is None:
        return balanced
    lab = _to_lab(balanced)
    equalized = plan.contrast_lightness[start:stop]
    return _with_lightness(balanced, lab, _blend(lab[..., 0], equalized, LOCAL_CONTRAST_BLEND))


def _balance_band(band: np.ndarray, plan: ToneBandPlan) -> np.ndarray:
    settings = plan.settings
    if settings.neutral_gray:
        return neutral_gray(band)
    if settings.fix_faded:
        return _faded_band(band, plan)
    if plan.levels is not None:
        return _leveled(band, plan.levels, LEVELS_BLEND)
    return band.astype(np.float32, copy=True)


def _faded_band(band: np.ndarray, plan: ToneBandPlan) -> np.ndarray:
    balanced = band if plan.gains is None else _apply_gains(band, plan.gains)
    if plan.curves is None:
        return balanced.astype(np.float32, copy=True)
    return apply_channel_curves(balanced, plan.curves, plan.settings.strength)


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
    return _apply_gains(rgb, gray_point_gains(rgb, point))


def gray_point_gains(rgb: np.ndarray, point: GrayPoint) -> np.ndarray:
    sample = gray_point_sample(rgb, point)
    return (sample.mean() / np.maximum(sample, CHANNEL_EPSILON)).astype(np.float32)


def _apply_gains(rgb: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return np.clip(rgb * gains, 0.0, 1.0)


def gray_point_sample(rgb: np.ndarray, point: GrayPoint) -> np.ndarray:
    x, y = point
    height, width = rgb.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Gray point {point} is outside the {width}x{height} image")
    radius = GRAY_POINT_WINDOW // 2
    window = rgb[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    return window.reshape(-1, 3).mean(axis=0)


def fit_neutral_axis_curves(rgb: np.ndarray) -> tuple[ChannelCurve, ChannelCurve, ChannelCurve]:
    band_means = neutral_axis_band_means(rgb)
    grays = band_means.mean(axis=1)
    return tuple(_monotone_curve(band_means[:, channel], grays) for channel in range(3))


def neutral_axis_band_means(rgb: np.ndarray) -> np.ndarray:
    pixels = _fit_sample(rgb)
    lab = _to_lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    return _least_chromatic_band_means(pixels, lab)


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
    return _leveled(rgb, lightness_levels(_to_lab(rgb)[..., 0]), blend)


def _leveled(rgb: np.ndarray, levels: tuple[float, float], blend: float) -> np.ndarray:
    low, high = levels
    lab = _to_lab(rgb)
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
    return _with_lightness(rgb, lab, _blend(lab[..., 0], _clahe_lightness(lab[..., 0]), blend))


def _clahe_lightness(lightness: np.ndarray) -> np.ndarray:
    lightness_u16 = np.round(lightness * (UINT16_MAX / LAB_L_MAX)).astype(np.uint16)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILES)
    return clahe.apply(lightness_u16).astype(np.float32) * (LAB_L_MAX / UINT16_MAX)


def _with_lightness(rgb: np.ndarray, lab: np.ndarray, lightness: np.ndarray) -> np.ndarray:
    relit = lab.copy()
    relit[..., 0] = lightness
    # Applied as a delta: the float Lab round trip alone drifts ~1e-3 even where L did not change.
    delta = _lab_to_rgb(relit)
    del relit
    delta -= _lab_to_rgb(lab)
    delta += rgb
    return np.clip(delta, 0.0, 1.0, out=delta)


def _blend(original: np.ndarray, processed: np.ndarray, amount: float) -> np.ndarray:
    return (original + (processed - original) * np.float32(amount)).astype(np.float32)


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(lab, dtype=np.float32), cv2.COLOR_Lab2RGB)


def _require_unit_interval(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")


def _require_same_shape(first: np.ndarray, second: np.ndarray) -> None:
    if first.shape != second.shape:
        raise ValueError(f"Images must have the same shape, got {first.shape} and {second.shape}")


def find_periodic_peaks(rgb: np.ndarray) -> PeriodicPeaks:
    luma = _center_crop(rgb, PEAK_ANALYSIS_SIDE).astype(np.float32) @ LUMA_WEIGHTS
    residual = _spectral_residual(luma)
    freq_y, freq_x = _centered_frequency_grid(residual.shape)
    candidates = np.hypot(freq_y, freq_x) >= MIN_PATTERN_FREQUENCY
    threshold = _outlier_threshold(residual[candidates])
    is_peak = candidates & (residual > threshold) & _is_local_max(residual) & _upper_half_plane(freq_y, freq_x)
    order = np.argsort(residual[is_peak], kind="stable")[::-1][:MAX_PERIODIC_PEAKS]
    frequencies = np.stack([freq_y[is_peak], freq_x[is_peak]], axis=-1)[order]
    found = PeriodicPeaks(frequencies, residual[is_peak][order], resolution=1.0 / min(luma.shape))
    return harmonic_peaks(found)


def harmonic_peaks(peaks: PeriodicPeaks) -> PeriodicPeaks:
    # A straight edge or a banded gradient leaves a ray of peaks through DC at non-harmonic radii.
    radii = np.hypot(peaks.frequencies[:, 0], peaks.frequencies[:, 1])
    keep = np.zeros(len(radii), dtype=bool)
    for members in _direction_groups(peaks.frequencies):
        keep[members] = _are_harmonics(radii[members])
    return PeriodicPeaks(peaks.frequencies[keep], peaks.strengths[keep], peaks.resolution)


def _direction_groups(frequencies: np.ndarray) -> list[np.ndarray]:
    if len(frequencies) == 0:
        return []
    angles = np.degrees(np.arctan2(frequencies[:, 0], frequencies[:, 1])) % 180.0
    # Lines at 179.9 and 0.1 degrees are the same direction.
    angles = np.where(angles > 180.0 - RAY_ANGLE_TOLERANCE_DEG, angles - 180.0, angles)
    order = np.argsort(angles, kind="stable")
    breaks = np.flatnonzero(np.diff(angles[order]) > RAY_ANGLE_TOLERANCE_DEG) + 1
    return np.split(order, breaks)


def _are_harmonics(radii: np.ndarray) -> bool:
    multiples = radii / radii.min()
    return bool(np.all(np.abs(multiples - np.round(multiples)) <= HARMONIC_TOLERANCE))


def screen_period(peaks: PeriodicPeaks) -> float | None:
    if len(peaks.frequencies) == 0:
        return None
    return float(1.0 / np.hypot(*peaks.frequencies[0]))


def _center_crop(image: np.ndarray, side: int) -> np.ndarray:
    top, left = (max(0, (size - side) // 2) for size in image.shape[:2])
    return image[top : top + side, left : left + side]


def _spectral_residual(luma: np.ndarray) -> np.ndarray:
    window = np.outer(np.hanning(luma.shape[0]), np.hanning(luma.shape[1])).astype(np.float32)
    spectrum = sp_fft.fftshift(sp_fft.fft2((luma - luma.mean()) * window))
    log_magnitude = np.log1p(np.abs(spectrum)).astype(np.float32)
    return log_magnitude - cv2.GaussianBlur(log_magnitude, (0, 0), PEAK_BACKGROUND_SIGMA_BINS)


def _centered_frequency_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    freq_y = sp_fft.fftshift(sp_fft.fftfreq(shape[0]))
    freq_x = sp_fft.fftshift(sp_fft.fftfreq(shape[1]))
    return np.meshgrid(freq_y, freq_x, indexing="ij")


def _outlier_threshold(values: np.ndarray) -> float:
    if values.size == 0:
        return float("inf")
    # A clean scan has a very narrow residual spread: 3.5 sigma alone would flag bumps of 2x over the background.
    return float(values.mean() + max(PEAK_SIGMA_THRESHOLD * values.std(), MIN_PEAK_PROMINENCE))


def _is_local_max(values: np.ndarray) -> np.ndarray:
    return values >= cv2.dilate(values, np.ones((3, 3), np.uint8))


def _upper_half_plane(freq_y: np.ndarray, freq_x: np.ndarray) -> np.ndarray:
    # A real image has a symmetric spectrum: keep one peak of each (f, -f) pair.
    return (freq_y > 0) | ((freq_y == 0) & (freq_x > 0))


def descreen_notch(rgb: np.ndarray, peaks: PeriodicPeaks | None = None, strength: float = 1.0) -> np.ndarray:
    _require_unit_interval(strength, "Strength")
    if strength == 0:
        return _unchanged(rgb)
    found = peaks if peaks is not None else find_periodic_peaks(rgb)
    if len(found.frequencies) == 0:
        return _unchanged(rgb)
    return np.clip(_blend(rgb, _notch_filter(rgb, found), strength), 0.0, 1.0)


def _unchanged(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float32, copy=True)


def _notch_filter(rgb: np.ndarray, peaks: PeriodicPeaks) -> np.ndarray:
    height, width = rgb.shape[:2]
    pad_y, pad_x = (int(np.ceil(NOTCH_PAD_FRACTION * size)) for size in (height, width))
    padded = cv2.copyMakeBorder(
        np.ascontiguousarray(rgb, dtype=np.float32), pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REFLECT_101
    )
    gain = notch_gain(padded.shape[:2], peaks)
    filtered = np.empty((height, width, rgb.shape[2]), np.float32)
    for channel in range(rgb.shape[2]):
        spectrum = sp_fft.rfft2(padded[..., channel]) * gain
        restored = sp_fft.irfft2(spectrum, s=padded.shape[:2])
        filtered[..., channel] = restored[pad_y : pad_y + height, pad_x : pad_x + width]
    return filtered


def notch_gain(shape: tuple[int, int], peaks: PeriodicPeaks) -> np.ndarray:
    height, width = shape
    gain = np.ones((height, width // 2 + 1), np.float32)
    # The peaks come from a smaller analysis crop, so their position here is only known to half its bin.
    radius = NOTCH_DILATE_BINS + 0.5 * peaks.resolution * max(height, width)
    for freq_y, freq_x in peaks.frequencies:
        _stamp_notch(gain, freq_y * height, freq_x * width, radius)
        _stamp_notch(gain, -freq_y * height, -freq_x * width, radius)
    period = screen_period(peaks)
    if period is not None:
        _protect_low_frequencies(gain, shape, NOTCH_PROTECT_FRACTION / period)
    return gain


def _stamp_notch(gain: np.ndarray, center_row: float, center_col: float, radius: float) -> None:
    reach = int(np.ceil(radius + 3.0 * NOTCH_EDGE_SIGMA_BINS))
    rows = np.arange(int(np.floor(center_row)) - reach, int(np.ceil(center_row)) + reach + 1)
    cols = np.arange(int(np.floor(center_col)) - reach, int(np.ceil(center_col)) + reach + 1)
    cols = cols[(cols >= 0) & (cols < gain.shape[1])]
    distance = np.hypot(rows[:, None] - center_row, cols[None, :] - center_col)
    stop = _notch_profile(distance, radius)
    np.minimum.at(gain, (rows[:, None] % gain.shape[0], cols[None, :]), (1.0 - stop).astype(np.float32))


def _notch_profile(distance: np.ndarray, radius: float) -> np.ndarray:
    edge = np.exp(-((distance - radius) ** 2) / (2.0 * NOTCH_EDGE_SIGMA_BINS**2))
    return np.where(distance <= radius, 1.0, edge)


def _protect_low_frequencies(gain: np.ndarray, shape: tuple[int, int], radius: float) -> None:
    freq_y = sp_fft.fftfreq(shape[0]).astype(np.float32)[:, None]
    freq_x = sp_fft.rfftfreq(shape[1]).astype(np.float32)[None, :]
    gain[np.hypot(freq_y, freq_x) < radius] = 1.0


def descreen_halftone(rgb: np.ndarray, period: float | None = None, strength: float = 1.0) -> np.ndarray:
    _require_unit_interval(strength, "Strength")
    _require_positive_period(period)
    if strength == 0:
        return _unchanged(rgb)
    screen = period if period is not None else screen_period(find_periodic_peaks(rgb))
    if screen is None:
        return _unchanged(rgb)
    smoothed = _gaussian_blur(rgb, HALFTONE_SIGMA_PER_PERIOD * screen)
    sharpened = _unsharp(smoothed, HALFTONE_UNSHARP_SIGMA_PER_PERIOD * screen, HALFTONE_UNSHARP_AMOUNT)
    return np.clip(_blend(rgb, sharpened, strength), 0.0, 1.0)


def _require_positive_period(period: float | None) -> None:
    if period is not None and period <= 0:
        raise ValueError(f"Screen period must be positive, got {period}")


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    source = np.ascontiguousarray(image, dtype=np.float32)
    return cv2.GaussianBlur(source, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)


def _unsharp(image: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    return image + np.float32(amount) * (image - _gaussian_blur(image, sigma))


# Clean-room à trous (Holschneider et al. 1989; Starck & Murtagh): linear B-spline taps dilated 2^level.
def wavelet_lowpass(image: np.ndarray, levels: int = ATROUS_LEVELS) -> np.ndarray:
    if levels < 1:
        raise ValueError(f"Wavelet levels must be at least 1, got {levels}")
    smooth = np.ascontiguousarray(image, dtype=np.float32)
    for level in range(levels):
        kernel = _atrous_kernel(level)
        smooth = cv2.sepFilter2D(smooth, -1, kernel, kernel, borderType=cv2.BORDER_REFLECT_101)
    return smooth


def _atrous_kernel(level: int) -> np.ndarray:
    step = 2**level
    kernel = np.zeros(2 * step + 1, np.float32)
    kernel[[0, step, 2 * step]] = ATROUS_TAPS
    return kernel


def atrous_halo(levels: int = ATROUS_LEVELS) -> int:
    return 2**levels - 1


def wavelet_color_transfer(restored: np.ndarray, reference: np.ndarray, levels: int = ATROUS_LEVELS) -> np.ndarray:
    _require_same_shape(restored, reference)
    detail = restored - wavelet_lowpass(restored, levels)
    return np.clip(detail + wavelet_lowpass(reference, levels), 0.0, 1.0)


def keep_grain(original: np.ndarray, denoised: np.ndarray, amount: float) -> np.ndarray:
    _require_unit_interval(amount, "Keep grain")
    _require_same_shape(original, denoised)
    return _blend(denoised, original, amount)


def process_in_bands(
    image: np.ndarray,
    function: BandFunction,
    band_rows: int = BAND_ROWS,
    halo: int = 0,
) -> np.ndarray:
    _require_band_sizes(band_rows, halo)
    output = None
    for start, stop in _band_spans(image.shape[0], band_rows):
        band = _run_band(image, function, start, stop, halo)
        output = output if output is not None else np.empty((image.shape[0], *band.shape[1:]), band.dtype)
        output[start:stop] = band
    return output


def _require_band_sizes(band_rows: int, halo: int) -> None:
    if band_rows < 1 or halo < 0:
        raise ValueError(f"Bands need band_rows >= 1 and halo >= 0, got {band_rows} and {halo}")


def _band_spans(height: int, band_rows: int) -> list[tuple[int, int]]:
    return [(start, min(start + band_rows, height)) for start in range(0, max(height, 1), band_rows)]


def _run_band(image: np.ndarray, function: BandFunction, start: int, stop: int, halo: int) -> np.ndarray:
    low, high = max(0, start - halo), min(image.shape[0], stop + halo)
    result = function(image[low:high])
    if result.shape[0] != high - low:
        raise ValueError(f"Band function must keep the row count: got {result.shape[0]} rows for {high - low}")
    return result[start - low : stop - low]
