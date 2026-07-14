from __future__ import annotations

from fractions import Fraction

import pytest

from app.services.media_tools import compute_interpolated_fps

# ---------------------------------------------------------------------------
# Task 12 (4.4) - FPS math for RIFE interpolation: new_rate = source_fps * multiplier.
# Critical for playback duration / audio sync once frame count is multiplied.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_fps,multiplier,expected",
    [
        ("24000/1001", 2, Fraction(24000, 1001) * 2),
        ("24000/1001", 4, Fraction(24000, 1001) * 4),
        ("30/1", 2, Fraction(60, 1)),
        ("30/1", 3, Fraction(90, 1)),
        ("30", 2, Fraction(60, 1)),
        ("60/1", 1, Fraction(60, 1)),
    ],
)
def test_compute_interpolated_fps_multiplies_source_fps(
    source_fps: str, multiplier: int, expected: Fraction
) -> None:
    assert compute_interpolated_fps(source_fps, multiplier) == expected


def test_compute_interpolated_fps_rejects_zero_fps() -> None:
    with pytest.raises(ValueError, match="0/1"):
        compute_interpolated_fps("0/1", 2)


def test_compute_interpolated_fps_rejects_negative_fps() -> None:
    with pytest.raises(ValueError):
        compute_interpolated_fps("-30/1", 2)


def test_compute_interpolated_fps_rejects_malformed_fps() -> None:
    with pytest.raises(ValueError):
        compute_interpolated_fps("not-a-fraction", 2)


def test_compute_interpolated_fps_rejects_missing_fps() -> None:
    with pytest.raises(ValueError):
        compute_interpolated_fps(None, 2)  # type: ignore[arg-type]
