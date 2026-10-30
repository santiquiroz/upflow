from __future__ import annotations

import numpy as np

from app.services.inpaint_score import fill_score


def score_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.full((256, 256, 3), 180, dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=bool)
    mask[80:176, 80:176] = True
    return rgb, mask, np.zeros(mask.shape, dtype=bool)


def test_flat_fill_matching_the_ring_has_a_low_score() -> None:
    rgb, mask, protect = score_scene()

    result = fill_score(rgb, mask, protect)

    assert result["score"] == 0.0
    assert result["d_light"] == 0.0
    assert result["tex_ratio"] == 0.0


def test_dark_blob_in_a_light_ring_has_a_high_score() -> None:
    rgb, mask, protect = score_scene()
    rgb[mask] = 10

    result = fill_score(rgb, mask, protect)

    assert result["d_light"] > 50
    assert result["score"] > 3


def test_medium_scale_checkerboard_in_a_flat_ring_has_a_high_score() -> None:
    rgb, mask, protect = score_scene()
    rows, columns = np.indices(mask.shape)
    pattern = np.where((rows // 16 + columns // 16) % 2, 255, 105)
    rgb[mask] = pattern[mask, None]

    result = fill_score(rgb, mask, protect)

    assert result["tex_ratio"] > 1.2
    assert result["score"] > 3
