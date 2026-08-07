from __future__ import annotations

import numpy as np
import pytest

from app.services.inpaint_mask import (
    compute_crop_box,
    dilate_mask,
    feather_mask,
    mask_bbox,
    soft_composite,
)


def square_mask(size: int = 64, box: tuple[int, int, int, int] = (20, 20, 40, 40)) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = 255
    return mask


# --- dilatación -------------------------------------------------------------


def test_dilate_grows_the_marked_area_outward() -> None:
    mask = square_mask()

    grown = dilate_mask(mask, 4)

    assert grown[20, 20] == 255
    # 4 px hacia afuera del borde superior/izquierdo quedan dentro
    assert grown[16, 30] == 255
    assert grown[30, 16] == 255
    # 6 px afuera sigue fuera
    assert grown[14, 30] == 0


def test_dilate_with_zero_pixels_is_a_no_op() -> None:
    mask = square_mask()
    np.testing.assert_array_equal(dilate_mask(mask, 0), mask)


def test_dilate_never_leaves_the_canvas() -> None:
    mask = square_mask(box=(0, 0, 10, 10))
    grown = dilate_mask(mask, 8)
    assert grown.shape == mask.shape
    assert grown[0, 0] == 255


# --- feather ----------------------------------------------------------------


def test_feather_creates_a_gradient_at_the_edge_only() -> None:
    mask = square_mask()

    soft = feather_mask(mask, 6)

    assert soft[30, 30] == 255  # centro intacto
    assert soft[0, 0] == 0  # lejos del borde sigue negro
    edge_values = soft[20, 14:26]
    assert 0 < int(edge_values.min()) or int(edge_values.max()) == 255
    assert len(set(edge_values.tolist())) > 2  # hay transición, no un escalón


def test_feather_with_zero_radius_keeps_hard_edges() -> None:
    mask = square_mask()
    np.testing.assert_array_equal(feather_mask(mask, 0), mask)


# --- bbox y caja de recorte -------------------------------------------------


def test_mask_bbox_finds_the_marked_region() -> None:
    assert mask_bbox(square_mask(box=(10, 12, 30, 34))) == (10, 12, 30, 34)


def test_mask_bbox_returns_none_for_an_empty_mask() -> None:
    assert mask_bbox(np.zeros((32, 32), dtype=np.uint8)) is None


def test_crop_box_adds_padding_and_stays_inside_the_image() -> None:
    box = compute_crop_box((20, 20, 40, 40), padding=32, image_size=(64, 64))

    assert box == (0, 0, 64, 64)


def test_crop_box_keeps_a_square_aspect_for_the_model() -> None:
    left, top, right, bottom = compute_crop_box((100, 100, 140, 110), padding=16, image_size=(512, 512))

    assert right - left == bottom - top
    # la región marcada sigue adentro
    assert left <= 100 and top <= 100 and right >= 140 and bottom >= 110


def test_crop_box_of_a_wide_mask_is_clamped_to_the_image() -> None:
    left, top, right, bottom = compute_crop_box((0, 200, 512, 260), padding=64, image_size=(512, 512))

    assert (left, right) == (0, 512)
    assert top >= 0 and bottom <= 512


def test_crop_box_never_returns_a_degenerate_region() -> None:
    left, top, right, bottom = compute_crop_box((10, 10, 11, 11), padding=0, image_size=(256, 256))
    assert right > left and bottom > top


# --- marcas mas largas que la dimension menor de la imagen (bug 2026-08-07) --
# El cuadrado clampado a min(lado, ancho, alto) dejaba la marca afuera del crop
# y solo se editaba la franja central, con costura dura a los costados.


def test_crop_box_covers_a_mark_wider_than_the_image_height() -> None:
    bbox = (200, 400, 1700, 700)  # 1500x300 en una foto 1920x1080
    left, top, right, bottom = compute_crop_box(bbox, padding=0, image_size=(1920, 1080))

    assert left <= 200 and right >= 1700, "la marca tiene que entrar entera"
    assert top >= 0 and bottom <= 1080


def test_crop_box_covers_a_mark_taller_than_the_image_width() -> None:
    bbox = (400, 200, 700, 1700)  # 300x1500 en una foto vertical 1080x1920
    left, top, right, bottom = compute_crop_box(bbox, padding=0, image_size=(1080, 1920))

    assert top <= 200 and bottom >= 1700, "la marca tiene que entrar entera"
    assert left >= 0 and right <= 1080


def test_crop_box_stays_square_when_the_mark_fits() -> None:
    # Comportamiento historico intacto: una marca normal sigue dando cuadrado.
    left, top, right, bottom = compute_crop_box(
        (100, 100, 400, 200), padding=50, image_size=(2000, 2000)
    )

    assert right - left == bottom - top


# --- composición suave ------------------------------------------------------


def test_soft_composite_keeps_the_original_where_the_mask_is_black() -> None:
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    generated = np.full((8, 8, 3), 255, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    result = soft_composite(generated, original, mask)

    assert result[0, 0].tolist() == [0, 0, 0]
    assert result[3, 3].tolist() == [255, 255, 255]


def test_soft_composite_blends_proportionally_on_the_gradient() -> None:
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    generated = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.full((4, 4), 128, dtype=np.uint8)

    result = soft_composite(generated, original, mask)

    assert result[0, 0, 0] == pytest.approx(100, abs=2)


def test_soft_composite_rejects_a_mismatched_mask() -> None:
    with pytest.raises(ValueError):
        soft_composite(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((5, 5), dtype=np.uint8),
        )
