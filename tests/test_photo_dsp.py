from __future__ import annotations

import tracemalloc

import cv2
import numpy as np
import pytest

from app.services.photo_dsp import (
    BAND_ROWS,
    PeriodicPeaks,
    ToneSettings,
    apply_gray_point,
    apply_tone,
    apply_tone_in_bands,
    atrous_halo,
    descreen_halftone,
    descreen_notch,
    find_periodic_peaks,
    fit_neutral_axis_curves,
    fix_faded_colors,
    harmonic_peaks,
    keep_grain,
    keep_tone_levels,
    local_contrast,
    neutral_gray,
    notch_gain,
    process_in_bands,
    screen_period,
    wavelet_color_transfer,
    wavelet_lowpass,
)

FADE_SEEDS = range(6)


def smooth_field(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    coarse = rng.random((cells, cells), dtype=np.float32)
    return np.clip(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def natural_scene(seed: int, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hue = smooth_field(rng, size, 6) * 360.0
    saturation = smooth_field(rng, size, 5) ** 2 * 0.8
    value = 0.08 + 0.87 * smooth_field(rng, size, 7)
    hsv = np.stack([hue, saturation, value], axis=-1).astype(np.float32)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)


def density_fade(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Cyan, magenta and yellow dyes keep part of their density, plus a yellow stain in the highlights.
    retention = np.array([rng.uniform(0.45, 0.75), rng.uniform(0.8, 0.95), rng.uniform(0.7, 0.95)], np.float32)
    density = -np.log10(np.maximum(rgb, 1e-3))
    faded = 10.0 ** (-density * retention)
    faded[..., 2] *= 1.0 - rng.uniform(0.08, 0.18) * faded.mean(axis=-1)
    return np.clip(faded, 0.0, 1.0).astype(np.float32)


def curves_fade(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gamma, gain, offset = rng.uniform(0.7, 1.4, 3), rng.uniform(0.75, 1.1, 3), rng.uniform(-0.05, 0.1, 3)
    return np.clip(gain * rgb**gamma + offset, 0.0, 1.0).astype(np.float32)


def lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


def median_delta_e(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(lab(first) - lab(second), axis=-1)))


def chroma(rgb: np.ndarray) -> np.ndarray:
    lab_image = lab(rgb)
    return np.hypot(lab_image[..., 1], lab_image[..., 2])


def sepia_print(size: int = 96) -> np.ndarray:
    ramp = np.tile(np.linspace(0.25, 0.7, size, dtype=np.float32), (size, 1))
    lightness = ramp * 100.0
    lab_image = np.stack([lightness, np.full_like(ramp, 6.0), np.full_like(ramp, 18.0)], axis=-1)
    return np.clip(cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB), 0.0, 1.0)


@pytest.mark.parametrize("fade", [density_fade, curves_fade], ids=["density", "curves"])
def test_tone_fix_faded_colors_lowers_the_median_delta_e(fade) -> None:
    before, after = [], []
    for seed in FADE_SEEDS:
        original = natural_scene(seed)
        faded = fade(original, np.random.default_rng(100 + seed))

        restored = fix_faded_colors(faded)

        before.append(median_delta_e(faded, original))
        after.append(median_delta_e(restored, original))

    assert all(fixed < untouched for fixed, untouched in zip(after, before))
    assert np.median(after) < 0.9 * np.median(before)


def test_tone_fix_faded_colors_barely_moves_an_unfaded_photo() -> None:
    original = natural_scene(3)

    assert median_delta_e(fix_faded_colors(original), original) < 2.0


def test_tone_fix_faded_colors_neutralizes_a_cast_gray_ramp() -> None:
    ramp = np.tile(np.linspace(0.1, 0.9, 64, dtype=np.float32)[None, :, None], (32, 1, 3))
    cast = np.clip(ramp * np.array([1.0, 0.92, 0.75], np.float32), 0.0, 1.0)

    restored = fix_faded_colors(cast, strength=1.0)

    assert np.median(chroma(restored)) < 0.25 * np.median(chroma(cast))


def test_tone_neutral_axis_curves_are_monotone_and_anchored() -> None:
    faded = density_fade(natural_scene(1), np.random.default_rng(7))
    probe = np.linspace(0.0, 1.0, 4097)

    for curve in fit_neutral_axis_curves(faded):
        mapped = np.interp(probe, curve.inputs, curve.outputs)
        assert np.all(np.diff(curve.inputs) > 0)
        assert np.all(np.diff(mapped) >= 0)
        assert mapped[0] == 0.0 and mapped[-1] == 1.0


def test_tone_zero_strength_is_the_identity() -> None:
    faded = curves_fade(natural_scene(2), np.random.default_rng(3))

    result = fix_faded_colors(faded, strength=0.0)

    np.testing.assert_array_equal(result, faded)
    assert result is not faded


def test_tone_zero_strength_still_applies_the_gray_point() -> None:
    faded = curves_fade(natural_scene(2), np.random.default_rng(3))

    result = fix_faded_colors(faded, strength=0.0, gray_point=(20, 20))

    np.testing.assert_array_equal(result, apply_gray_point(faded, (20, 20)))


def test_tone_strength_scales_the_correction() -> None:
    faded = density_fade(natural_scene(4), np.random.default_rng(9))
    full = fix_faded_colors(faded, strength=1.0)

    half = fix_faded_colors(faded, strength=0.5)

    np.testing.assert_allclose(half, (faded + full) / 2, atol=1e-5)


@pytest.mark.parametrize("strength", [-0.1, 1.5])
def test_tone_strength_outside_the_unit_interval_is_rejected(strength: float) -> None:
    with pytest.raises(ValueError, match="Strength"):
        fix_faded_colors(natural_scene(0), strength=strength)


def test_tone_gray_point_makes_the_picked_patch_neutral() -> None:
    rgb = natural_scene(5)
    rgb[40:50, 60:70] = np.array([0.62, 0.5, 0.38], np.float32)

    balanced = apply_gray_point(rgb, (64, 44))

    patch = balanced[42:47, 62:67].reshape(-1, 3).mean(axis=0)
    np.testing.assert_allclose(patch, patch.mean(), atol=1e-4)


def test_tone_gray_point_outside_the_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="Gray point"):
        apply_gray_point(natural_scene(0), (500, 3))


def test_tone_keep_original_tone_preserves_sepia_chroma() -> None:
    sepia = sepia_print()

    leveled = keep_tone_levels(sepia)

    np.testing.assert_allclose(np.median(chroma(leveled)), np.median(chroma(sepia)), rtol=0.05)
    assert np.ptp(lab(leveled)[..., 0]) > np.ptp(lab(sepia)[..., 0])


def test_tone_keep_original_tone_preserves_a_hand_tinted_patch() -> None:
    tinted = sepia_print()
    tinted[30:50, 30:50] = np.array([0.7, 0.35, 0.3], np.float32)

    leveled = keep_tone_levels(tinted)

    patch_before, patch_after = chroma(tinted)[30:50, 30:50], chroma(leveled)[30:50, 30:50]
    np.testing.assert_allclose(patch_after, patch_before, rtol=0.05)


def test_tone_keep_original_tone_leaves_a_flat_image_alone() -> None:
    flat = np.full((16, 16, 3), 0.4, np.float32)

    np.testing.assert_allclose(keep_tone_levels(flat), flat, atol=1e-6)


def test_tone_neutral_gray_removes_the_chroma_and_keeps_lightness() -> None:
    sepia = sepia_print()

    gray = neutral_gray(sepia)

    assert np.max(chroma(gray)) < 0.5
    np.testing.assert_allclose(gray[..., 0], gray[..., 2])
    np.testing.assert_allclose(lab(gray)[..., 0], lab(sepia)[..., 0], atol=0.5)


def test_tone_local_contrast_raises_detail_without_touching_chroma() -> None:
    rng = np.random.default_rng(11)
    texture = 0.45 + 0.03 * rng.standard_normal((128, 128)).astype(np.float32)
    rgb = np.clip(np.stack([texture * 1.1, texture, texture * 0.85], axis=-1), 0.0, 1.0)

    result = local_contrast(rgb)

    assert lab(result)[..., 0].std() > lab(rgb)[..., 0].std()
    np.testing.assert_allclose(lab(result)[..., 1:], lab(rgb)[..., 1:], atol=1.0)
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_tone_default_settings_keep_the_original_tone() -> None:
    sepia = sepia_print()

    np.testing.assert_array_equal(apply_tone(sepia, ToneSettings()), keep_tone_levels(sepia))


def test_tone_neutral_gray_setting_wins_over_the_others() -> None:
    sepia = sepia_print()
    settings = ToneSettings(neutral_gray=True, fix_faded=True)

    np.testing.assert_array_equal(apply_tone(sepia, settings), neutral_gray(sepia))


def test_tone_fix_faded_setting_uses_strength_and_gray_point() -> None:
    faded = density_fade(natural_scene(2), np.random.default_rng(5))
    settings = ToneSettings(fix_faded=True, strength=0.4, gray_point=(10, 10))

    expected = fix_faded_colors(faded, 0.4, (10, 10))

    np.testing.assert_array_equal(apply_tone(faded, settings), expected)


def test_tone_local_contrast_runs_after_the_tone_step() -> None:
    sepia = sepia_print()
    settings = ToneSettings(local_contrast=True)

    np.testing.assert_array_equal(apply_tone(sepia, settings), local_contrast(keep_tone_levels(sepia)))


def test_tone_without_keep_tone_is_the_identity() -> None:
    sepia = sepia_print()

    result = apply_tone(sepia, ToneSettings(keep_tone=False))

    np.testing.assert_array_equal(result, sepia)
    assert result is not sepia


def test_tone_does_not_mutate_the_input() -> None:
    faded = density_fade(natural_scene(0), np.random.default_rng(1))
    snapshot = faded.copy()

    all_settings = (
        ToneSettings(local_contrast=True),
        ToneSettings(fix_faded=True, gray_point=(5, 5)),
        ToneSettings(neutral_gray=True),
    )
    for settings in all_settings:
        result = apply_tone(faded, settings)
        assert result.dtype == np.float32
        assert result.shape == faded.shape

    np.testing.assert_array_equal(faded, snapshot)


def psnr(first: np.ndarray, second: np.ndarray) -> float:
    mse = float(np.mean((first.astype(np.float64) - second.astype(np.float64)) ** 2))
    return 10.0 * np.log10(1.0 / mse)


def gray_scene(seed: int, size: int = 256) -> np.ndarray:
    photo = natural_scene(seed, size)
    return np.repeat(photo.mean(axis=-1, keepdims=True), 3, axis=-1)


def periodic_texture(photo: np.ndarray, period: float = 7.0, levels: float = 18.0) -> np.ndarray:
    # Three cosines 60 degrees apart, like the silk/honeycomb paper finish of research-color-danos §5.3.
    rows, cols = np.mgrid[: photo.shape[0], : photo.shape[1]].astype(np.float32)
    texture = sum(
        np.cos(2.0 * np.pi * (cols * np.cos(angle) + rows * np.sin(angle)) / period)
        for angle in np.deg2rad([0.0, 60.0, 120.0])
    )
    return np.clip(photo + (levels / 255.0 / 3.0) * texture[..., None], 0.0, 1.0).astype(np.float32)


def am_halftone(photo: np.ndarray, period: float = 5.0) -> np.ndarray:
    rows, cols = np.mgrid[: photo.shape[0], : photo.shape[1]].astype(np.float32)
    u, v = (cols + rows) / np.sqrt(2.0), (cols - rows) / np.sqrt(2.0)
    screen = 0.5 + 0.25 * (np.cos(2.0 * np.pi * u / period) + np.cos(2.0 * np.pi * v / period))
    return (photo > screen[..., None]).astype(np.float32)


def test_descreen_notch_recovers_the_periodic_paper_texture() -> None:
    clean = natural_scene(3, 256)
    textured = periodic_texture(clean)

    restored = descreen_notch(textured)

    assert restored.dtype == np.float32
    assert restored.shape == textured.shape
    assert psnr(restored, clean) >= psnr(textured, clean) + 8.0


def test_descreen_peaks_find_the_texture_period() -> None:
    peaks = find_periodic_peaks(periodic_texture(natural_scene(3, 256)))

    assert len(peaks.frequencies) >= 3
    assert screen_period(peaks) == pytest.approx(7.0, abs=0.4)


def test_descreen_notch_gain_touches_a_small_part_of_the_spectrum() -> None:
    peaks = find_periodic_peaks(periodic_texture(natural_scene(3, 256)))

    gain = notch_gain((308, 308), peaks)

    assert gain.shape == (308, 155)
    assert gain.min() == 0.0
    assert np.mean(gain < 0.99) < 0.01
    assert gain[0, 0] == 1.0


@pytest.mark.parametrize("seed", range(4))
def test_descreen_notch_leaves_a_photo_without_a_pattern_alone(seed: int) -> None:
    clean = natural_scene(seed, 256)

    assert len(find_periodic_peaks(clean).frequencies) == 0
    np.testing.assert_array_equal(descreen_notch(clean), clean)


def test_descreen_notch_zero_strength_is_the_identity() -> None:
    textured = periodic_texture(natural_scene(3, 128))

    result = descreen_notch(textured, strength=0.0)

    np.testing.assert_array_equal(result, textured)
    assert result is not textured


def test_descreen_notch_strength_blends_with_the_input() -> None:
    textured = periodic_texture(natural_scene(3, 128))
    full = descreen_notch(textured)

    half = descreen_notch(textured, strength=0.5)

    np.testing.assert_allclose(half, (textured + full) / 2.0, atol=1e-5)


def test_descreen_notch_without_peaks_returns_a_copy() -> None:
    clean = natural_scene(4, 64)
    no_peaks = PeriodicPeaks(np.zeros((0, 2)), np.zeros(0), resolution=1.0 / 64)

    result = descreen_notch(clean, peaks=no_peaks)

    np.testing.assert_array_equal(result, clean)
    assert result is not clean


def test_descreen_halftone_recovers_a_binary_am_screen() -> None:
    photo = gray_scene(5)
    reference = cv2.GaussianBlur(photo, (0, 0), 1.0)
    screened = am_halftone(photo)

    restored = descreen_halftone(screened, period=5.0)

    assert psnr(screened, reference) < 16.0
    assert psnr(restored, reference) >= 22.0


def test_descreen_halftone_estimates_the_screen_period() -> None:
    screened = am_halftone(gray_scene(5))

    assert screen_period(find_periodic_peaks(screened)) == pytest.approx(5.0, abs=0.4)
    np.testing.assert_allclose(descreen_halftone(screened), descreen_halftone(screened, period=5.0), atol=0.02)


def test_descreen_halftone_beats_the_notch_on_binary_dots() -> None:
    photo = gray_scene(5)
    reference = cv2.GaussianBlur(photo, (0, 0), 1.0)
    screened = am_halftone(photo)

    assert psnr(descreen_halftone(screened, period=5.0), reference) > psnr(descreen_notch(screened), reference)


def test_descreen_halftone_zero_strength_is_the_identity() -> None:
    screened = am_halftone(gray_scene(5, 64))

    np.testing.assert_array_equal(descreen_halftone(screened, period=5.0, strength=0.0), screened)


@pytest.mark.parametrize("period", [0.0, -3.0])
def test_descreen_halftone_rejects_a_non_positive_period(period: float) -> None:
    with pytest.raises(ValueError):
        descreen_halftone(gray_scene(5, 32), period=period)


@pytest.mark.parametrize("descreen", [descreen_notch, descreen_halftone])
def test_descreen_rejects_strength_outside_the_unit_interval(descreen) -> None:
    with pytest.raises(ValueError):
        descreen(gray_scene(5, 32), strength=1.2)


def test_descreen_keeps_sixteen_bit_precision_and_the_input() -> None:
    fine = periodic_texture(natural_scene(3, 128)) * np.float32(0.999) + np.float32(1.0 / 65535.0)
    snapshot = fine.copy()

    for result in (descreen_notch(fine), descreen_halftone(fine, period=7.0)):
        assert result.dtype == np.float32
        assert len(np.unique(result)) > 256 * 3

    np.testing.assert_array_equal(fine, snapshot)


def detail_and_cast(reference: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    detail = rng.normal(0.0, 0.04, reference.shape[:2]).astype(np.float32)[..., None]
    cast = np.array([0.08, -0.05, -0.1], np.float32)
    return np.clip(reference + detail + cast, 0.0, 1.0)


def highpass(image: np.ndarray) -> np.ndarray:
    return image - cv2.GaussianBlur(image, (0, 0), 6.0)


def test_wavelet_color_transfer_takes_the_low_frequencies_of_the_reference() -> None:
    reference = natural_scene(7, 128)
    restored = detail_and_cast(reference, 1)

    result = wavelet_color_transfer(restored, reference)

    blurred_reference = cv2.GaussianBlur(reference, (0, 0), 8.0)
    assert median_delta_e(cv2.GaussianBlur(result, (0, 0), 8.0), blurred_reference) < 1.0
    assert median_delta_e(cv2.GaussianBlur(restored, (0, 0), 8.0), blurred_reference) > 5.0


def test_wavelet_color_transfer_keeps_the_detail_of_the_restored_image() -> None:
    reference = natural_scene(7, 128)
    restored = detail_and_cast(reference, 1)

    result = wavelet_color_transfer(restored, reference)

    correlation = np.corrcoef(highpass(result).ravel(), highpass(restored).ravel())[0, 1]
    assert correlation > 0.95


def test_wavelet_color_transfer_brings_the_sepia_back() -> None:
    sepia = sepia_print(128)
    neutral = neutral_gray(sepia)

    result = wavelet_color_transfer(neutral, sepia)

    assert np.median(chroma(neutral)) < 1.0
    assert np.median(chroma(result)) == pytest.approx(np.median(chroma(sepia)), abs=1.0)


def test_wavelet_color_transfer_of_an_image_onto_itself_is_the_identity() -> None:
    photo = natural_scene(8, 96)

    np.testing.assert_allclose(wavelet_color_transfer(photo, photo), photo, atol=1e-6)


def test_wavelet_lowpass_keeps_a_flat_image_and_the_mean() -> None:
    flat = np.full((40, 50, 3), 0.3, np.float32)
    photo = natural_scene(9, 96)

    np.testing.assert_allclose(wavelet_lowpass(flat), flat, atol=1e-6)
    assert wavelet_lowpass(photo).mean() == pytest.approx(photo.mean(), abs=5e-3)
    assert highpass(wavelet_lowpass(photo)).std() < highpass(photo).std() / 2.0


def test_wavelet_color_transfer_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        wavelet_color_transfer(natural_scene(1, 64), natural_scene(1, 32))


def test_wavelet_lowpass_rejects_zero_levels() -> None:
    with pytest.raises(ValueError):
        wavelet_lowpass(natural_scene(1, 32), levels=0)


def test_grain_keep_grain_remixes_the_denoise_residual() -> None:
    original = natural_scene(10, 64)
    denoised = cv2.GaussianBlur(original, (0, 0), 1.5)

    np.testing.assert_array_equal(keep_grain(original, denoised, 0.0), denoised)
    np.testing.assert_allclose(keep_grain(original, denoised, 1.0), original, atol=1e-6)
    np.testing.assert_allclose(
        keep_grain(original, denoised, 0.25) - denoised, (original - denoised) * 0.25, atol=1e-6
    )


def test_grain_keep_grain_rejects_bad_input() -> None:
    original = natural_scene(10, 32)
    with pytest.raises(ValueError):
        keep_grain(original, original, 1.5)
    with pytest.raises(ValueError):
        keep_grain(original, natural_scene(10, 16), 0.25)


def square_root_curve(rgb: np.ndarray) -> np.ndarray:
    return np.sqrt(rgb) * np.float32(0.9)


def test_bands_pointwise_processing_matches_the_whole_image() -> None:
    photo = natural_scene(11, 200)

    banded = process_in_bands(photo, square_root_curve, band_rows=64)

    np.testing.assert_array_equal(banded, square_root_curve(photo))


def test_bands_with_a_halo_match_the_whole_wavelet_lowpass() -> None:
    photo = natural_scene(12, 260)

    banded = process_in_bands(photo, wavelet_lowpass, band_rows=64, halo=atrous_halo(5))

    np.testing.assert_allclose(banded, wavelet_lowpass(photo), atol=1e-6)


def test_bands_without_the_halo_differ_from_the_whole_wavelet_lowpass() -> None:
    photo = natural_scene(12, 260)

    banded = process_in_bands(photo, wavelet_lowpass, band_rows=64)

    assert np.abs(banded - wavelet_lowpass(photo)).max() > 1e-3


def test_bands_can_change_the_channel_count() -> None:
    photo = natural_scene(13, 100)

    banded = process_in_bands(photo, lambda band: band.mean(axis=-1, keepdims=True), band_rows=30)

    np.testing.assert_allclose(banded, photo.mean(axis=-1, keepdims=True), atol=1e-7)


def test_bands_bound_the_peak_memory_to_the_output_plus_one_band() -> None:
    image = np.random.default_rng(0).random((8192, 128, 3), dtype=np.float32)

    def several_copies(band: np.ndarray) -> np.ndarray:
        return ((band * 2.0 + 1.0) * 0.5 - band) + band * np.float32(0.25)

    tracemalloc.start()
    try:
        process_in_bands(image, several_copies)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    band_nbytes = image[:BAND_ROWS].nbytes
    assert BAND_ROWS == 1024
    assert peak < image.nbytes + 6 * band_nbytes


def test_bands_reject_a_function_that_changes_the_row_count() -> None:
    with pytest.raises(ValueError):
        process_in_bands(natural_scene(1, 64), lambda band: band[:-1], band_rows=16)


@pytest.mark.parametrize(("band_rows", "halo"), [(0, 0), (16, -1)])
def test_bands_reject_bad_sizes(band_rows: int, halo: int) -> None:
    with pytest.raises(ValueError):
        process_in_bands(natural_scene(1, 32), lambda band: band, band_rows=band_rows, halo=halo)


def peaks_at(frequencies: list[tuple[float, float]]) -> PeriodicPeaks:
    points = np.array(frequencies, dtype=np.float64)
    return PeriodicPeaks(points, np.linspace(3.0, 2.0, len(points)), resolution=1.0 / 1024)


def test_descreen_peaks_drop_a_ray_of_non_harmonic_peaks_from_an_edge() -> None:
    ray = [(radius, 0.0) for radius in (0.042, 0.048, 0.051, 0.088, 0.091, 0.121)]

    assert len(harmonic_peaks(peaks_at(ray)).frequencies) == 0


def test_descreen_peaks_keep_harmonics_and_lone_peaks() -> None:
    diagonal = 0.2 / np.sqrt(2.0)
    screen = [(diagonal, diagonal), (2 * diagonal, 2 * diagonal), (diagonal, -diagonal), (0.0, 0.3)]

    kept = harmonic_peaks(peaks_at(screen))

    np.testing.assert_allclose(kept.frequencies, peaks_at(screen).frequencies)


def test_descreen_peaks_group_directions_across_the_half_plane_edge() -> None:
    nearly_horizontal = [(0.0, 0.1), (0.001, -0.117)]

    assert len(harmonic_peaks(peaks_at(nearly_horizontal)).frequencies) == 0


TONE_BAND_SETTINGS = {
    "keep_tone": ToneSettings(),
    "fix_faded": ToneSettings(strength=0.8, fix_faded=True),
    "fix_faded_gray_point": ToneSettings(strength=0.6, fix_faded=True, gray_point=(40, 70)),
    "neutral_gray": ToneSettings(neutral_gray=True),
    "local_contrast": ToneSettings(local_contrast=True),
    "fix_faded_local_contrast": ToneSettings(fix_faded=True, local_contrast=True),
    "identity": ToneSettings(keep_tone=False),
}


@pytest.mark.parametrize("settings", TONE_BAND_SETTINGS.values(), ids=TONE_BAND_SETTINGS.keys())
def test_bands_tone_matches_the_whole_image_tone(settings: ToneSettings) -> None:
    photo = density_fade(natural_scene(14, 150), np.random.default_rng(3))

    banded = apply_tone_in_bands(photo, settings, band_rows=37)

    np.testing.assert_allclose(banded, apply_tone(photo, settings), atol=2e-6)


def test_bands_tone_does_not_mutate_the_input() -> None:
    photo = natural_scene(15, 90)
    before = photo.copy()

    apply_tone_in_bands(photo, ToneSettings(fix_faded=True, local_contrast=True), band_rows=20)

    np.testing.assert_array_equal(photo, before)


def test_bands_tone_bounds_the_peak_memory() -> None:
    image = natural_scene(16, 256)
    tall = np.ascontiguousarray(np.tile(image, (32, 2, 1)))

    tracemalloc.start()
    try:
        apply_tone_in_bands(tall, ToneSettings(fix_faded=True))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < tall.nbytes + 8 * tall[:BAND_ROWS].nbytes
