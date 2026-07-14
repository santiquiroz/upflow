from __future__ import annotations

from fractions import Fraction

import pytest

from app.services.media_tools import compute_interpolated_fps, compute_target_frame_count, format_fps_fraction

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


# ---------------------------------------------------------------------------
# Task 15 (6.6) - TARGET_FPS math: absolute target frame count for RIFE
# (target_frames = round(source_count * target_fps / source_fps)) plus fps
# string normalization used for the encode -framerate / outputFps metadata.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_frame_count,source_fps,target_fps",
    [
        (100, "24000/1001", "60"),
        (48, "24000/1001", "60000/1001"),
        (30, "30/1", "60/1"),
        (24, "24", "48"),
    ],
)
def test_compute_target_frame_count_matches_round_formula(
    source_frame_count: int, source_fps: str, target_fps: str
) -> None:
    expected = round(source_frame_count * Fraction(target_fps) / Fraction(source_fps))
    assert compute_target_frame_count(source_frame_count, source_fps, target_fps) == expected


def test_compute_target_frame_count_preserves_duration_within_one_target_frame() -> None:
    """Anime 23.976 -> 60 exact: rounding to an integer frame count means the
    resulting duration can only drift by less than one frame at the target rate."""
    source_frame_count = 480
    source_fps = "24000/1001"
    target_fps = "60"

    target_frame_count = compute_target_frame_count(source_frame_count, source_fps, target_fps)

    source_duration = source_frame_count / Fraction(source_fps)
    target_duration = target_frame_count / Fraction(target_fps)
    one_target_frame = Fraction(1) / Fraction(target_fps)

    assert abs(source_duration - target_duration) < one_target_frame


def test_compute_target_frame_count_rejects_invalid_source_fps() -> None:
    with pytest.raises(ValueError):
        compute_target_frame_count(100, "not-a-fraction", "60")


def test_compute_target_frame_count_rejects_invalid_target_fps() -> None:
    with pytest.raises(ValueError):
        compute_target_frame_count(100, "24000/1001", "abc")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("60", "60/1"),
        ("60000/1001", "60000/1001"),
        ("30/1", "30/1"),
    ],
)
def test_format_fps_fraction_normalizes_to_num_den(value: str, expected: str) -> None:
    assert format_fps_fraction(value) == expected


def test_format_fps_fraction_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        format_fps_fraction("abc")
