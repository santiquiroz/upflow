from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.photo_geometry import Geometry, apply_geometry


def gradient(height: int = 12, width: int = 20) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.random((height, width, 3), dtype=np.float32)


def test_no_geometry_returns_an_equal_copy() -> None:
    rgb = gradient()

    result = apply_geometry(rgb)

    np.testing.assert_array_equal(result, rgb)
    assert result is not rgb


def test_four_quarter_turns_are_the_identity() -> None:
    rgb = gradient()
    result = rgb
    for _ in range(4):
        result = apply_geometry(result, rotate90=1)

    np.testing.assert_array_equal(result, rgb)


def test_one_quarter_turn_is_clockwise() -> None:
    rgb = gradient(2, 3)

    result = apply_geometry(rgb, rotate90=1)

    assert result.shape == (3, 2, 3)
    np.testing.assert_array_equal(result[0, -1], rgb[0, 0])
    np.testing.assert_array_equal(result[0, 0], rgb[-1, 0])


def test_rotation_accepts_negative_and_large_turns() -> None:
    rgb = gradient()

    np.testing.assert_array_equal(apply_geometry(rgb, rotate90=-1), apply_geometry(rgb, rotate90=3))
    np.testing.assert_array_equal(apply_geometry(rgb, rotate90=5), apply_geometry(rgb, rotate90=1))


def test_crop_is_exact() -> None:
    rgb = gradient()

    result = apply_geometry(rgb, crop=(3, 2, 5, 4))

    np.testing.assert_array_equal(result, rgb[2:6, 3:8])
    assert result.flags["C_CONTIGUOUS"]


def test_full_width_crop_does_not_alias_the_input() -> None:
    rgb = gradient()

    result = apply_geometry(rgb, crop=(0, 2, 20, 4))
    result[:] = 0.0

    assert rgb[2:6].any()


def test_crop_applies_after_rotation() -> None:
    rgb = gradient()

    result = apply_geometry(rgb, rotate90=1, crop=(0, 0, 4, 6))

    np.testing.assert_array_equal(result, np.rot90(rgb, k=-1)[0:6, 0:4])


@pytest.mark.parametrize(
    "crop", [(-1, 0, 4, 4), (0, 0, 0, 4), (0, 0, 21, 4), (18, 0, 4, 4), (0, 10, 4, 4)]
)
def test_crop_outside_the_image_is_rejected(crop: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        apply_geometry(gradient(), crop=crop)


def test_straighten_zero_degrees_is_the_identity() -> None:
    rgb = gradient()

    np.testing.assert_array_equal(apply_geometry(rgb, angle_deg=0.0), rgb)


def test_straighten_keeps_size_range_and_dtype() -> None:
    rgb = gradient(40, 60)

    result = apply_geometry(rgb, angle_deg=3.5)

    assert result.shape == rgb.shape
    assert result.dtype == np.float32
    assert 0.0 <= result.min() and result.max() <= 1.0
    assert not np.array_equal(result, rgb)


def test_straighten_rotates_counterclockwise_for_positive_angles() -> None:
    rgb = np.zeros((101, 101, 3), dtype=np.float32)
    rgb[50, 60:100] = 1.0

    result = apply_geometry(rgb, angle_deg=10.0)

    rows, cols = np.nonzero(result[..., 0] > 0.5)
    right_side = cols > 80
    assert rows[right_side].mean() < 50


def test_straighten_matches_a_lanczos_affine_about_the_center() -> None:
    rgb = gradient(30, 40)
    matrix = cv2.getRotationMatrix2D((19.5, 14.5), 2.0, 1.0)
    expected = cv2.warpAffine(
        rgb, matrix, (40, 30), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
    )

    result = apply_geometry(rgb, angle_deg=2.0)

    np.testing.assert_allclose(result, np.clip(expected, 0.0, 1.0))


def test_straighten_beyond_the_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        apply_geometry(gradient(), angle_deg=46.0)


def test_input_array_and_original_file_are_untouched(tmp_path: Path) -> None:
    rgb = gradient()
    original = tmp_path / "original.png"
    ok = cv2.imwrite(str(original), (rgb * 255).astype(np.uint8))
    assert ok
    file_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    array_sha = hashlib.sha256(rgb.tobytes()).hexdigest()

    apply_geometry(rgb, rotate90=1, crop=(1, 1, 5, 5), angle_deg=4.0)

    assert hashlib.sha256(rgb.tobytes()).hexdigest() == array_sha
    assert hashlib.sha256(original.read_bytes()).hexdigest() == file_sha


def test_geometry_output_size_follows_rotation_and_crop() -> None:
    assert Geometry().output_size(12, 20) == (12, 20)
    assert Geometry(rotate90=1).output_size(12, 20) == (20, 12)
    assert Geometry(rotate90=3, crop=(1, 2, 5, 7)).output_size(12, 20) == (7, 5)
    assert Geometry(angle=10.0).output_size(12, 20) == (12, 20)


def test_geometry_output_size_matches_the_applied_image() -> None:
    geometry = Geometry(rotate90=1, crop=(2, 3, 8, 9), angle=-4.0)

    assert geometry.apply(gradient()).shape[:2] == geometry.output_size(12, 20)


def test_geometry_crop_outside_the_rotated_image_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        Geometry(rotate90=1, crop=(0, 0, 13, 5)).output_size(12, 20)


def test_geometry_from_mapping_round_trips() -> None:
    geometry = Geometry.from_mapping({"rotate90": 5, "crop": [1, 2, 3, 4], "angle": 1.5})

    assert geometry == Geometry(rotate90=1, crop=(1, 2, 3, 4), angle=1.5)
    assert Geometry.from_mapping(geometry.to_dict()) == geometry
    assert Geometry.from_mapping(None) == Geometry()


@pytest.mark.parametrize("angle", [46.0, float("nan")])
def test_geometry_rejects_an_angle_beyond_the_straighten_limit(angle: float) -> None:
    with pytest.raises(ValueError, match="Straighten"):
        Geometry(angle=angle)
