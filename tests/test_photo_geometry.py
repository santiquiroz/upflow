from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.photo_capture import CaptureSuggestions, capture_signs, suggest_capture
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


def test_perspective_with_rectangle_corners_equals_the_crop() -> None:
    rgb = gradient(30, 40)
    corners = ((5.0, 4.0), (25.0, 4.0), (25.0, 20.0), (5.0, 20.0))

    result = apply_geometry(rgb, corners=corners)

    np.testing.assert_allclose(result, rgb[4:20, 5:25], atol=1e-5)


def test_perspective_rectifies_a_keystoned_print() -> None:
    rgb = np.zeros((200, 200, 3), dtype=np.float32)
    corners = np.array([(60, 40), (140, 40), (180, 160), (20, 160)], dtype=np.int32)
    cv2.fillConvexPoly(rgb, corners, (1.0, 1.0, 1.0))

    result = apply_geometry(rgb, corners=((60, 40), (140, 40), (180, 160), (20, 160)))

    assert result.shape[:2] == (round(np.hypot(40, 120)), 160)
    assert result[4:-4, 4:-4].min() > 0.9


def test_perspective_orders_corners_given_in_any_order() -> None:
    shuffled = ((25.0, 20.0), (5.0, 4.0), (5.0, 20.0), (25.0, 4.0))

    geometry = Geometry(corners=shuffled)

    assert geometry.corners == ((5.0, 4.0), (25.0, 4.0), (25.0, 20.0), (5.0, 20.0))


def test_perspective_output_size_is_the_longest_opposite_sides() -> None:
    geometry = Geometry(corners=((10, 10), (90, 20), (80, 70), (20, 60)))

    height, width = geometry.output_size(100, 100)

    assert width == round(max(np.hypot(80, 10), np.hypot(60, 10)))
    assert height == round(max(np.hypot(10, 50), np.hypot(10, 50)))


def test_perspective_then_crop_measures_the_crop_in_the_rectified_image() -> None:
    corners = ((5.0, 4.0), (25.0, 4.0), (25.0, 20.0), (5.0, 20.0))
    rgb = gradient(30, 40)
    geometry = Geometry(crop=(2, 3, 10, 8), corners=corners)

    result = geometry.apply(rgb)

    assert geometry.output_size(30, 40) == (8, 10)
    np.testing.assert_allclose(result, rgb[7:15, 7:17], atol=1e-5)


def test_perspective_corners_follow_the_quarter_turn() -> None:
    corners = ((0, 0), (24, 0), (24, 40), (0, 40))

    assert Geometry(rotate90=1, corners=corners).output_size(24, 40) == (40, 24)
    with pytest.raises(ValueError, match="outside"):
        Geometry(corners=corners).output_size(24, 40)


@pytest.mark.parametrize(
    "corners",
    [
        ((0, 0), (10, 10), (20, 0), (10, 30)),
        ((0, 0), (5, 0), (5, 5), (0, 5), (1, 1)),
        ((0, 0), (float("nan"), 0), (5, 5), (0, 5)),
    ],
)
def test_perspective_rejects_bad_corners(corners: tuple) -> None:
    with pytest.raises(ValueError, match="Perspective"):
        Geometry(corners=corners)


def test_perspective_rejects_a_quad_smaller_than_the_minimum() -> None:
    with pytest.raises(ValueError, match="less than"):
        Geometry(corners=((0, 0), (10, 0), (10, 10), (0, 10))).output_size(40, 40)


def test_perspective_and_straighten_are_exclusive() -> None:
    with pytest.raises(ValueError, match="Perspective and straighten"):
        Geometry(angle=2.0, corners=((0, 0), (20, 0), (20, 20), (0, 20)))


def test_perspective_geometry_round_trips_through_a_mapping() -> None:
    geometry = Geometry(rotate90=2, crop=(1, 1, 16, 16), corners=((1.5, 2), (30, 1), (31, 28), (0, 27.25)))

    assert Geometry.from_mapping(geometry.to_dict()) == geometry
    assert Geometry.from_mapping({"rotate90": 1, "crop": None, "angle": 0.0}).corners is None


SHEET_WHITE = (0.96, 0.96, 0.95)


def scanner_sheet(height: int = 600, width: int = 800, base=SHEET_WHITE) -> np.ndarray:
    rng = np.random.default_rng(11)
    sheet = np.empty((height, width, 3), dtype=np.float32)
    sheet[:] = base
    return np.clip(sheet + rng.normal(0, 0.01, sheet.shape).astype(np.float32), 0.0, 1.0)


def photo_texture(height: int, width: int, tint=(0.35, 0.3, 0.25)) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    wave = 0.15 * np.sin(xs / 17.0) * np.cos(ys / 23.0)
    return np.clip(np.stack([tint[0] + wave, tint[1] + wave, tint[2] + wave], axis=2), 0.0, 1.0)


def place_photo(sheet: np.ndarray, corners: np.ndarray, tint=(0.35, 0.3, 0.25)) -> np.ndarray:
    mask = np.zeros(sheet.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 1)
    texture = photo_texture(*sheet.shape[:2], tint=tint)
    placed = sheet.copy()
    placed[mask.astype(bool)] = texture[mask.astype(bool)]
    return placed


def tilted_box(center: tuple[float, float], size: tuple[float, float], angle: float) -> np.ndarray:
    return cv2.boxPoints((center, size, angle))


def is_photo_only(rgb: np.ndarray) -> bool:
    return bool(rgb.max(axis=2).max() < 0.8)


def test_auto_crop_levels_and_crops_a_tilted_photo_on_a_scanner_sheet() -> None:
    sheet = place_photo(scanner_sheet(), tilted_box((420, 300), (360, 240), 5.0))

    geometry = suggest_capture(sheet, 0).auto_crop

    assert geometry is not None
    assert abs(abs(geometry.angle) - 5.0) < 0.6
    result = geometry.apply(sheet)
    assert is_photo_only(result)
    assert max(float(edge.max()) for edge in (result[0], result[-1], result[:, 0], result[:, -1])) < 0.6
    assert result.shape[0] * result.shape[1] > 0.85 * 360 * 240


def test_auto_crop_of_a_straight_photo_needs_no_straighten() -> None:
    sheet = scanner_sheet()
    sheet[100:400, 150:550] = photo_texture(300, 400)

    geometry = suggest_capture(sheet, 0).auto_crop

    assert geometry is not None
    assert geometry.angle == 0.0
    x, y, width, height = geometry.crop
    assert 150 <= x <= 156 and 100 <= y <= 106
    assert 390 <= width <= 400 and 290 <= height <= 300


def test_auto_crop_follows_the_quarter_turn() -> None:
    sheet = scanner_sheet()
    sheet[100:400, 150:550] = photo_texture(300, 400)

    geometry = suggest_capture(sheet, 1).auto_crop

    assert geometry is not None and geometry.rotate90 == 1
    result = geometry.apply(sheet)
    assert is_photo_only(result)
    assert result.shape[0] > result.shape[1]


def test_auto_crop_offers_nothing_when_the_photo_fills_the_frame() -> None:
    suggestions = suggest_capture(photo_texture(300, 400), 1)

    assert suggestions == CaptureSuggestions(frame=(400, 300))


def test_auto_crop_offers_nothing_on_a_blank_sheet() -> None:
    assert suggest_capture(scanner_sheet(), 0) == CaptureSuggestions(frame=(600, 800))


def test_split_finds_each_photo_on_the_sheet_in_reading_order() -> None:
    sheet = scanner_sheet(900, 1200)
    sheet = place_photo(sheet, tilted_box((300, 220), (380, 260), 3.0), tint=(0.6, 0.2, 0.2))
    sheet = place_photo(sheet, tilted_box((880, 240), (360, 280), -4.0), tint=(0.2, 0.55, 0.2))
    sheet = place_photo(sheet, tilted_box((560, 660), (420, 300), 0.0), tint=(0.2, 0.2, 0.6))

    suggestions = suggest_capture(sheet, 0)

    assert suggestions.auto_crop is None
    assert len(suggestions.photos) == 3
    dominant = [int(np.argmax(geometry.apply(sheet).mean(axis=(0, 1)))) for geometry in suggestions.photos]
    assert dominant == [0, 1, 2]
    assert all(is_photo_only(geometry.apply(sheet)) for geometry in suggestions.photos)


def test_split_ignores_dust_and_small_marks() -> None:
    sheet = scanner_sheet()
    sheet[100:400, 100:350] = photo_texture(300, 250)
    sheet[450:560, 450:750] = photo_texture(110, 300)
    sheet[20:26, 700:706] = 0.1

    suggestions = suggest_capture(sheet, 0)

    assert len(suggestions.photos) == 2


def test_split_handles_photos_pushed_into_the_scanner_corner() -> None:
    sheet = scanner_sheet()
    sheet[0:260, 0:340] = photo_texture(260, 340)
    sheet[300:560, 420:780] = photo_texture(260, 360)

    suggestions = suggest_capture(sheet, 0)

    assert len(suggestions.photos) == 2
    assert all(is_photo_only(geometry.apply(sheet)) for geometry in suggestions.photos)


def phone_photo_of_print(corners: np.ndarray) -> np.ndarray:
    table = scanner_sheet(600, 800, base=(0.42, 0.3, 0.2))
    mask = np.zeros(table.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 1)
    table[mask.astype(bool)] = photo_texture(600, 800, tint=(0.8, 0.78, 0.75))[mask.astype(bool)]
    return table


def test_perspective_is_suggested_for_a_phone_photo_of_a_print() -> None:
    corners = np.array([(220, 90), (600, 110), (700, 520), (120, 500)], dtype=np.float64)
    photo = phone_photo_of_print(corners)

    suggestions = suggest_capture(photo, 0)

    assert suggestions.auto_crop is None
    assert suggestions.perspective is not None
    found = np.asarray(suggestions.perspective.corners)
    assert np.abs(found - corners).max() < 6
    assert suggestions.perspective.output_size(600, 800)[1] > 500


def test_perspective_signs_flag_a_keystoned_print() -> None:
    corners = np.array([(220, 90), (600, 110), (700, 520), (120, 500)], dtype=np.float64)

    signs = capture_signs(phone_photo_of_print(corners))

    assert signs.perspective and signs.phone_capture


def test_perspective_signs_stay_quiet_for_a_tilted_scan() -> None:
    sheet = place_photo(scanner_sheet(), tilted_box((420, 300), (360, 240), 5.0))

    signs = capture_signs(sheet)

    assert not signs.perspective and not signs.phone_capture


def glare_spot(rgb: np.ndarray, center: tuple[int, int], radius: float) -> np.ndarray:
    ys, xs = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]].astype(np.float32)
    spot = 1.6 * np.exp(-((xs - center[0]) ** 2 + (ys - center[1]) ** 2) / (2 * radius**2))
    return np.clip(rgb + spot[..., None], 0.0, 1.0)


def test_capture_signs_find_glare_on_a_print() -> None:
    photo = glare_spot(photo_texture(400, 600), (300, 180), 30.0)

    signs = capture_signs(photo)

    assert signs.glare and signs.phone_capture
    assert signs.glare_fraction > 0.003


def test_capture_signs_do_not_take_a_sharp_white_object_for_glare() -> None:
    photo = photo_texture(400, 600, tint=(0.15, 0.15, 0.15))
    photo[150:250, 250:350] = 1.0

    assert not capture_signs(photo).glare


def test_capture_signs_ignore_clipped_white_touching_the_border() -> None:
    photo = photo_texture(400, 600)
    photo[:, :60] = 1.0

    assert not capture_signs(photo).glare
