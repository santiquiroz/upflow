from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services.inpaint_grain import match_grain


def high_freq(channel: np.ndarray) -> np.ndarray:
    plano = channel.astype(np.float32)
    return plano - cv2.GaussianBlur(plano, (0, 0), 1.2)


def test_match_grain_restores_the_sensor_noise_level() -> None:
    rng = np.random.default_rng(7)
    original = np.clip(np.rint(128 + rng.normal(0, 4, (200, 200, 3))), 0, 255).astype(np.uint8)
    edited = np.full((200, 200, 3), 128, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[60:140, 60:140] = 255

    out = match_grain(edited, original, mask, seed=3)

    dentro = mask >= 128
    want = float(high_freq(original[:, :, 0])[mask == 0].std())
    have = float(high_freq(out[:, :, 0])[dentro].std())
    assert have == pytest.approx(want, rel=0.2)


def test_match_grain_uses_the_local_ring_despite_distant_sharp_detail() -> None:
    rng = np.random.default_rng(7)
    original = np.clip(np.rint(128 + rng.normal(0, 3, (400, 400, 3))), 0, 255).astype(np.uint8)
    checkerboard = (np.indices((120, 120)).sum(axis=0) % 2 * 255).astype(np.uint8)
    original[:120, :120] = checkerboard[..., None]
    edited = np.full((400, 400, 3), 128, dtype=np.uint8)
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[150:250, 150:250] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
    ring = (cv2.dilate((mask > 0).astype(np.uint8), kernel) > 0) & (mask == 0)

    out = match_grain(edited, original, mask, seed=3)

    for channel in range(3):
        want = float(high_freq(original[:, :, channel])[ring].std())
        have = float(high_freq(out[:, :, channel])[mask >= 128].std())
        assert have == pytest.approx(want, rel=0.25)


def test_match_grain_keeps_the_average_color() -> None:
    rng = np.random.default_rng(7)
    original = np.clip(np.rint(128 + rng.normal(0, 4, (200, 200, 3))), 0, 255).astype(np.uint8)
    edited = np.full((200, 200, 3), 128, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[60:140, 60:140] = 255

    out = match_grain(edited, original, mask, seed=3)

    assert abs(float(out[mask >= 128].mean()) - 128) < 0.6


def test_match_grain_leaves_the_unmarked_pixels_identical() -> None:
    rng = np.random.default_rng(7)
    original = np.clip(np.rint(128 + rng.normal(0, 4, (200, 200, 3))), 0, 255).astype(np.uint8)
    edited = np.full((200, 200, 3), 128, dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[60:140, 60:140] = 255

    out = match_grain(edited, original, mask, seed=3)

    np.testing.assert_array_equal(out[mask == 0], edited[mask == 0])


def test_a_flat_original_leaves_the_edited_image_untouched() -> None:
    plano = np.full((64, 64, 3), 128, dtype=np.uint8)
    otro = np.full((64, 64, 3), 200, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255

    np.testing.assert_array_equal(match_grain(otro, plano, mask, seed=1), otro)


def test_an_empty_mark_returns_the_edited_unchanged() -> None:
    edited = np.full((32, 32, 3), 200, dtype=np.uint8)
    original = np.full((32, 32, 3), 128, dtype=np.uint8)

    np.testing.assert_array_equal(
        match_grain(edited, original, np.zeros((32, 32), dtype=np.uint8)), edited
    )


def test_a_fully_marked_image_with_no_background_returns_the_edited_unchanged() -> None:
    edited = np.full((32, 32, 3), 200, dtype=np.uint8)
    original = np.full((32, 32, 3), 128, dtype=np.uint8)

    np.testing.assert_array_equal(
        match_grain(edited, original, np.full((32, 32), 255, dtype=np.uint8)), edited
    )
