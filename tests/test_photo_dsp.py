from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services.photo_dsp import (
    ToneSettings,
    apply_gray_point,
    apply_tone,
    fit_neutral_axis_curves,
    fix_faded_colors,
    keep_tone_levels,
    local_contrast,
    neutral_gray,
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
