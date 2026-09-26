from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from app.services.face_geometry import (
    ALIGN_BORDER_RGB,
    FACE_SIZE,
    MASK_EDGE_PX,
    SHARP_FACE_MIN,
    TEMPLATE_FFHQ_512,
    FacePolicy,
    align_face,
    align_matrix,
    default_face_policy,
    eye_distance,
    face_sharpness,
    inverse_paste_matrix,
    paste_box,
    template_mask,
    transform_points,
    upscaled_points,
    warp_to_box,
)

SCALES = (1, 2, 4)
LANDMARK_TOLERANCE_PX = 0.5


def tilted_landmarks(center=(300.0, 220.0), scale=0.35, angle_deg=12.0) -> np.ndarray:
    radians = math.radians(angle_deg)
    rotation = np.array([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]])
    template_centered = TEMPLATE_FFHQ_512 - FACE_SIZE / 2
    return (template_centered @ rotation.T) * scale + np.asarray(center)


def blob_image(points: np.ndarray, shape: tuple[int, int], sigma: float) -> np.ndarray:
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    image = np.zeros(shape, dtype=np.float32)
    for x, y in points:
        image += np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
    return image


def blob_centroid(image: np.ndarray, near: np.ndarray, radius: int) -> np.ndarray:
    x0, y0 = (int(round(v)) - radius for v in near)
    window = image[y0 : y0 + 2 * radius + 1, x0 : x0 + 2 * radius + 1].astype(np.float64)
    ys, xs = np.mgrid[0 : window.shape[0], 0 : window.shape[1]]
    total = window.sum()
    return np.array([(xs * window).sum() / total + x0, (ys * window).sum() / total + y0])


def upscale_pixels(image: np.ndarray, scale: int) -> np.ndarray:
    # Reescalado centrado: cada pixel de entrada ocupa un bloque s x s, como la salida de un SR.
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------- alinear y pegar


def test_template_is_the_facexlib_ffhq_512_template() -> None:
    assert TEMPLATE_FFHQ_512.shape == (5, 2)
    assert TEMPLATE_FFHQ_512[0].tolist() == pytest.approx([192.98138, 239.94708])
    assert TEMPLATE_FFHQ_512[4].tolist() == pytest.approx([313.08905, 371.15118])


def test_align_matrix_maps_the_landmarks_onto_the_template() -> None:
    landmarks = tilted_landmarks()

    matrix = align_matrix(landmarks)

    assert matrix.shape == (2, 3)
    assert np.abs(transform_points(landmarks, matrix) - TEMPLATE_FFHQ_512).max() < 1e-3


def test_align_matrix_is_a_similarity_even_with_noisy_landmarks() -> None:
    noisy = tilted_landmarks() + np.random.default_rng(3).normal(0.0, 1.5, (5, 2))

    linear = align_matrix(noisy)[:, :2]

    assert linear[0, 0] == pytest.approx(linear[1, 1], abs=1e-9)
    assert linear[0, 1] == pytest.approx(-linear[1, 0], abs=1e-9)


def test_align_matrix_rejects_degenerate_landmarks() -> None:
    with pytest.raises(ValueError, match="landmarks"):
        align_matrix(np.zeros((5, 2)))


def test_align_matrix_rejects_the_wrong_number_of_points() -> None:
    with pytest.raises(ValueError, match="5 landmarks"):
        align_matrix(np.zeros((4, 2)))


def test_align_face_puts_the_landmark_pixels_on_the_template() -> None:
    landmarks = tilted_landmarks()
    image = blob_image(landmarks, (480, 640), sigma=3.0)

    crop = align_face(image, align_matrix(landmarks))

    assert crop.shape == (FACE_SIZE, FACE_SIZE)
    for expected in TEMPLATE_FFHQ_512:
        assert np.abs(blob_centroid(crop, expected, radius=4) - expected).max() < LANDMARK_TOLERANCE_PX


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.uint8, [135, 133, 132]),
        (np.uint16, [135 * 257, 133 * 257, 132 * 257]),
        (np.float32, [135 / 255, 133 / 255, 132 / 255]),
    ],
)
def test_align_face_fills_outside_the_photo_with_the_gray_border(dtype, expected) -> None:
    image = np.zeros((120, 160, 3), dtype=dtype)
    landmarks = tilted_landmarks(center=(20.0, 20.0), scale=0.2)

    crop = align_face(image, align_matrix(landmarks))

    assert crop.dtype == dtype
    assert crop[0, 0].tolist() == pytest.approx(expected, rel=1e-6)
    assert ALIGN_BORDER_RGB == (135, 133, 132)


@pytest.mark.parametrize("scale", SCALES)
def test_inverse_paste_matrix_returns_the_template_to_the_upscaled_landmarks(scale: int) -> None:
    landmarks = tilted_landmarks()
    inverse = inverse_paste_matrix(align_matrix(landmarks), scale)

    pasted = transform_points(TEMPLATE_FFHQ_512, inverse)

    assert np.abs(pasted - upscaled_points(landmarks, scale)).max() < LANDMARK_TOLERANCE_PX


@pytest.mark.parametrize("scale", SCALES)
def test_align_and_paste_round_trip_keeps_the_landmark_pixels_within_half_a_pixel(scale: int) -> None:
    landmarks = tilted_landmarks()
    photo = blob_image(landmarks, (480, 640), sigma=3.0)
    matrix = align_matrix(landmarks)
    crop = align_face(photo, matrix)
    upscaled = upscale_pixels(photo, scale)
    inverse = inverse_paste_matrix(matrix, scale)
    box = paste_box(inverse, upscaled.shape[:2])

    pasted = np.zeros_like(upscaled)
    x0, y0, x1, y1 = box
    pasted[y0:y1, x0:x1] = warp_to_box(crop, inverse, box)

    for point in landmarks:
        expected = blob_centroid(upscaled, upscaled_points(point[None], scale)[0], radius=4 * scale)
        found = blob_centroid(pasted, expected, radius=4 * scale)
        assert np.abs(found - expected).max() < LANDMARK_TOLERANCE_PX


def test_facexlib_offset_would_be_half_a_pixel_off_at_2x() -> None:
    landmarks = tilted_landmarks()
    inverse = inverse_paste_matrix(align_matrix(landmarks), 2)
    facexlib = cv2.invertAffineTransform(align_matrix(landmarks)) * 2
    facexlib[:, 2] += 0.5 * 2

    ours = transform_points(TEMPLATE_FFHQ_512, inverse)
    theirs = transform_points(TEMPLATE_FFHQ_512, facexlib)

    assert np.abs(theirs - ours).max() == pytest.approx(0.5, abs=1e-6)


def test_inverse_paste_matrix_rejects_a_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        inverse_paste_matrix(align_matrix(tilted_landmarks()), 0)


@pytest.mark.parametrize("scale", SCALES)
def test_warp_to_box_equals_the_full_image_warp_inside_the_box(scale: int) -> None:
    crop = np.random.default_rng(1).random((FACE_SIZE, FACE_SIZE, 3)).astype(np.float32)
    crop = cv2.GaussianBlur(crop, (0, 0), 3.0)
    inverse = inverse_paste_matrix(align_matrix(tilted_landmarks()), scale)
    shape = (480 * scale, 640 * scale)
    box = paste_box(inverse, shape)

    full = cv2.warpAffine(crop, inverse, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)
    reach = cv2.warpAffine(np.ones((FACE_SIZE, FACE_SIZE), np.float32), inverse, (shape[1], shape[0]))
    x0, y0, x1, y1 = box
    # En el borde del recorte OpenCV cuantiza a 1/32 px y puede tomar o no el vecino negro.
    interior = cv2.erode((reach[y0:y1, x0:x1] > 0.999).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    difference = np.abs(warp_to_box(crop, inverse, box) - full[y0:y1, x0:x1])

    assert difference[interior].max() < 1e-3
    assert difference.max() < 1 / 32


@pytest.mark.parametrize("scale", SCALES)
def test_paste_box_holds_every_pixel_the_crop_reaches(scale: int) -> None:
    inverse = inverse_paste_matrix(align_matrix(tilted_landmarks()), scale)
    shape = (480 * scale, 640 * scale)
    reach = cv2.warpAffine(np.ones((FACE_SIZE, FACE_SIZE), np.float32), inverse, (shape[1], shape[0]))
    x0, y0, x1, y1 = paste_box(inverse, shape)

    outside = reach.copy()
    outside[y0:y1, x0:x1] = 0.0

    assert outside.max() == 0.0
    assert (x1 - x0) * (y1 - y0) < shape[0] * shape[1] / 4


def test_paste_box_is_clipped_to_the_image() -> None:
    landmarks = tilted_landmarks(center=(10.0, 10.0))
    inverse = inverse_paste_matrix(align_matrix(landmarks), 1)

    x0, y0, x1, y1 = paste_box(inverse, (200, 300))

    assert (x0, y0) == (0, 0)
    assert 0 < x1 <= 300
    assert 0 < y1 <= 200


def test_paste_box_is_empty_when_the_face_is_outside_the_image() -> None:
    inverse = inverse_paste_matrix(align_matrix(tilted_landmarks(center=(2000.0, 2000.0))), 1)

    x0, y0, x1, y1 = paste_box(inverse, (200, 300))

    assert x1 <= x0 or y1 <= y0


# ---------------------------------------------------------------- mascara


def test_template_mask_is_opaque_in_the_center_and_clear_at_the_edges() -> None:
    mask = template_mask()

    assert mask.shape == (FACE_SIZE, FACE_SIZE)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0
    assert mask[FACE_SIZE // 2, FACE_SIZE // 2] == pytest.approx(1.0)
    assert mask[0, 0] < 0.01
    assert mask[FACE_SIZE // 2, 0] < 0.05


def test_template_mask_fades_over_the_edge_band() -> None:
    row = template_mask()[FACE_SIZE // 2]

    assert MASK_EDGE_PX == FACE_SIZE // 20
    assert row[2 * MASK_EDGE_PX] > 0.99
    assert 0.2 < row[MASK_EDGE_PX] < 0.8
    assert np.all(np.diff(row[: FACE_SIZE // 2]) >= -1e-6)


def test_template_mask_is_symmetric() -> None:
    mask = template_mask()

    assert np.allclose(mask, mask[:, ::-1], atol=1e-6)
    assert np.allclose(mask, mask.T, atol=1e-6)


# ---------------------------------------------------------------- ojos y nitidez


def test_eye_distance_is_the_distance_between_the_first_two_landmarks() -> None:
    landmarks = np.array([[10.0, 20.0], [13.0, 24.0], [0, 0], [0, 0], [0, 0]])

    assert eye_distance(landmarks) == pytest.approx(5.0)


def textured_face(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.random((FACE_SIZE, FACE_SIZE)).astype(np.float32)
    detail = cv2.GaussianBlur(noise, (0, 0), 1.0)
    shading = np.linspace(0.3, 0.7, FACE_SIZE, dtype=np.float32)[None, :]
    return np.clip(shading + (detail - detail.mean()) * 1.5, 0.0, 1.0)


def test_face_sharpness_ranks_a_sharp_crop_above_its_blurred_copies() -> None:
    sharp = textured_face()
    soft = cv2.GaussianBlur(sharp, (0, 0), 2.0)
    softer = cv2.GaussianBlur(sharp, (0, 0), 5.0)

    assert face_sharpness(sharp) > face_sharpness(soft) > face_sharpness(softer)


def test_face_sharpness_ranks_a_downscaled_and_upscaled_face_as_soft() -> None:
    sharp = textured_face(1)
    small = cv2.resize(sharp, (64, 64), interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(small, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_CUBIC)

    assert face_sharpness(upscaled) < SHARP_FACE_MIN <= face_sharpness(sharp)


def test_face_sharpness_does_not_depend_on_contrast() -> None:
    face = textured_face(2)
    flat = 0.5 + (face - 0.5) * 0.4

    assert face_sharpness(flat) == pytest.approx(face_sharpness(face), rel=0.02)


def test_face_sharpness_reads_rgb_and_integer_crops_alike() -> None:
    gray = textured_face(3)
    rgb_u8 = np.rint(np.repeat(gray[:, :, None], 3, axis=2) * 255).astype(np.uint8)

    assert face_sharpness(rgb_u8) == pytest.approx(face_sharpness(gray), rel=0.05)


def test_a_flat_crop_has_zero_sharpness() -> None:
    assert face_sharpness(np.full((FACE_SIZE, FACE_SIZE, 3), 0.4, np.float32)) == 0.0


def test_face_sharpness_ignores_the_gray_border_outside_the_face() -> None:
    face = textured_face(4)
    bordered = face.copy()
    bordered[:, :100] = 133 / 255

    assert face_sharpness(bordered) == pytest.approx(face_sharpness(face), rel=1e-6)


# ---------------------------------------------------------------- politica por cara (§2.4)


@pytest.mark.parametrize(
    ("eye_px", "sharpness", "tier", "enabled", "blend", "selectable", "confirm"),
    [
        (80.0, 0.0, "restore", True, 0.6, True, False),
        (32.0, SHARP_FACE_MIN - 1e-3, "restore", True, 0.6, True, False),
        (80.0, SHARP_FACE_MIN, "alreadyClear", False, 0.4, True, False),
        (31.9, 0.0, "small", False, 0.5, True, False),
        (16.0, 5.0, "small", False, 0.5, True, False),
        (15.9, 0.0, "tooSmallFaithful", False, 0.4, True, True),
        (8.0, 0.0, "tooSmallFaithful", False, 0.4, True, True),
        (7.9, 0.0, "tooSmall", False, None, False, False),
        (0.0, 0.0, "tooSmall", False, None, False, False),
    ],
)
def test_default_face_policy_follows_the_section_2_4_table(
    eye_px, sharpness, tier, enabled, blend, selectable, confirm
) -> None:
    policy = default_face_policy(eye_px, sharpness)

    assert policy == FacePolicy(
        tier=tier,
        enabled=enabled,
        blend=blend,
        selectable=selectable,
        needs_confirmation=confirm,
        label_key=f"restore.face.{tier}",
    )


def test_every_policy_label_has_english_text() -> None:
    from app.services.face_geometry import FACE_POLICY_TEXTS

    assert FACE_POLICY_TEXTS["restore.face.alreadyClear"] == "Already clear — restoring may change it"
    assert FACE_POLICY_TEXTS["restore.face.tooSmall"] == "Too small"
    for eye_px in (80.0, 20.0, 10.0, 4.0):
        for sharpness in (0.0, SHARP_FACE_MIN):
            assert default_face_policy(eye_px, sharpness).label_key in FACE_POLICY_TEXTS


@pytest.mark.parametrize(("eye_px", "sharpness"), [(-1.0, 0.0), (math.nan, 0.0), (40.0, math.nan), (40.0, -0.5)])
def test_default_face_policy_rejects_invalid_measures(eye_px: float, sharpness: float) -> None:
    with pytest.raises(ValueError):
        default_face_policy(eye_px, sharpness)
