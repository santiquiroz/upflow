from __future__ import annotations

import numpy as np
import pytest

from app.services.inpaint_mask import (
    MAX_COMPONENTS,
    clean_protect_mask,
    compute_crop_box,
    dilate_mask,
    dilation_for,
    feather_for,
    feather_mask,
    grow_per_component,
    grow_small_hulls,
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


# --- dilatación y feather proporcionales ------------------------------------
# 8 px fijos no tapan el halo de una marca grande (medido 2026-10: persona de
# 1600x2700 con halo, limpia agrandando ~1.5% del lado mayor).


def test_dilation_scales_with_the_marked_size() -> None:
    assert dilation_for((0, 0, 2000, 2000)) == 30


def test_dilation_never_goes_below_the_minimum() -> None:
    assert dilation_for((0, 0, 100, 100)) == 8


def test_feather_scales_with_the_marked_size() -> None:
    assert feather_for((0, 0, 2000, 2000)) == 10


def test_feather_never_goes_below_the_minimum() -> None:
    assert feather_for((0, 0, 100, 100)) == 8


def test_distant_small_components_keep_their_own_growth_scale() -> None:
    mask = np.zeros((3000, 3000), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    mask[2800:2900, 2800:2900] = 255

    grown = grow_per_component(mask, None, None)

    assert grown.dtype == np.uint8
    assert grown.shape == mask.shape
    for start in (100, 2800):
        region = grown[start - 60:start + 160, start - 60:start + 160]
        assert np.count_nonzero(region > 127) <= (100 + 2 * 8 + 2 * 8) ** 2
        assert grown[start + 50, start - 8] == 255
        assert grown[start + 50, start - 45] == 0
    assert np.all(grown >= mask)


def test_large_component_keeps_proportional_thirty_pixel_dilation() -> None:
    mask = square_mask(size=2200, box=(100, 100, 2100, 2100))

    grown = grow_per_component(mask, None, None)

    assert grown[1100, 70] == 255
    assert grown[1100, 69] < 255
    assert grown[1100, 30] == 0


def test_zero_growth_preserves_a_prepared_gradient_mask() -> None:
    mask = np.tile(np.arange(256, dtype=np.uint8), (256, 1))

    grown = grow_per_component(mask, 0, 0)

    assert np.array_equal(grown, mask)


def test_component_growth_preserves_values_below_the_component_threshold() -> None:
    mask = square_mask()
    mask[0, 0] = 80

    grown = grow_per_component(mask, 2, 0)

    assert grown[0, 0] == 80
    assert np.all(grown >= mask)


@pytest.mark.parametrize("dilation, feather", [(3, 0), (0, 2), (3, 2), (None, 0), (0, None)])
def test_component_growth_respects_explicit_radii_and_mask_values(
    dilation: int | None, feather: int | None
) -> None:
    mask = square_mask(size=128, box=(50, 50, 70, 70))
    mask[50:70, 50:70] = 180
    resolved_dilation = dilation if dilation is not None else 8
    resolved_feather = feather if feather is not None else 8
    expected = feather_mask(dilate_mask(mask, resolved_dilation), resolved_feather)

    grown = grow_per_component(mask, dilation, feather)

    np.testing.assert_allclose(grown, expected, atol=1, rtol=0)
    assert grown[60, 60] == 180


@pytest.mark.parametrize("dilation, feather", [(None, None), (3, 2), (None, 0), (0, None)])
def test_many_components_use_whole_mask_growth_without_component_windows(
    monkeypatch: pytest.MonkeyPatch, dilation: int | None, feather: int | None
) -> None:
    mask = np.zeros((2000, 2000), dtype=np.uint8)
    for row in range(10):
        for column in range(20):
            top, left = 50 + row * 190, 50 + column * 95
            mask[top:top + 3, left:left + 3] = 255
    whole_bbox = mask_bbox(mask)
    assert whole_bbox is not None
    resolved_dilation = dilation if dilation is not None else dilation_for(whole_bbox)
    resolved_feather = feather if feather is not None else feather_for(whole_bbox)
    expected = feather_mask(dilate_mask(mask, resolved_dilation), resolved_feather)
    monkeypatch.setattr(
        "app.services.inpaint_mask._grow_component_window",
        lambda *_args: pytest.fail("No debe recorrer los componentes de una máscara ruidosa"),
    )

    grown = grow_per_component(mask, dilation, feather)

    np.testing.assert_array_equal(grown, expected)


def test_small_l_shape_fills_its_hull_and_grows_by_eight_percent() -> None:
    import cv2

    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[50:250, 50:70] = 255
    mask[230:250, 50:250] = 255
    mask[0, 0] = 80
    original = mask.copy()
    hull = np.zeros_like(mask)
    vertices = np.array([(50, 50), (69, 50), (249, 230), (249, 249), (50, 249)])
    cv2.fillConvexPoly(hull, vertices, 255)
    expected = np.maximum(mask, dilate_mask(hull, 16))

    grown = grow_small_hulls(mask)

    np.testing.assert_array_equal(grown, expected)
    np.testing.assert_array_equal(mask, original)
    assert grown.dtype == np.uint8
    assert mask[150, 100] == 0 and grown[150, 100] == 255
    assert grown[150, 34] == 255
    assert grown[150, 33] == 0
    assert grown[0, 0] == 80


def test_hull_growth_preserves_components_larger_than_small_side() -> None:
    mask = np.zeros((800, 800), dtype=np.uint8)
    mask[30:731, 30:50] = 255
    mask[711:731, 30:731] = 255
    original = mask.copy()

    grown = grow_small_hulls(mask)

    np.testing.assert_array_equal(grown, original)
    np.testing.assert_array_equal(mask, original)


def test_hull_growth_preserves_masks_with_more_than_max_components() -> None:
    mask = np.zeros((3, 3 * (MAX_COMPONENTS + 1)), dtype=np.uint8)
    mask[1, 1::3] = 255
    original = mask.copy()

    grown = grow_small_hulls(mask)

    np.testing.assert_array_equal(grown, original)
    np.testing.assert_array_equal(mask, original)


def test_hull_growth_preserves_an_empty_mask() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)

    grown = grow_small_hulls(mask)

    np.testing.assert_array_equal(grown, mask)
    assert grown.dtype == np.uint8


def test_clean_protect_removes_small_islands_and_keeps_large_subjects() -> None:
    protect = np.zeros((400, 500), dtype=bool)
    protect[30:130, 30:130] = True
    protect[180:230, 180:230] = True
    protect[300:310, 300:310] = True
    original = protect.copy()

    cleaned = clean_protect_mask(protect)

    assert cleaned[80, 80]
    assert cleaned[200, 200]
    assert not cleaned[300:310, 300:310].any()
    assert cleaned.dtype == np.bool_
    assert cleaned.shape == protect.shape
    np.testing.assert_array_equal(protect, original)


def test_clean_protect_keeps_components_exactly_at_the_area_threshold() -> None:
    protect = np.zeros((400, 500), dtype=bool)
    protect[30:130, 30:130] = True
    protect[200:210, 200:220] = True
    protect[300, 200:399] = True

    cleaned = clean_protect_mask(protect)

    assert cleaned[205, 210]
    assert not cleaned[300, 200:399].any()


@pytest.mark.parametrize("shape,radius", [((200, 300), 6), ((100, 3000), 9), ((3000, 100), 9)])
def test_clean_protect_grows_with_an_ellipse_scaled_to_the_long_side(shape, radius) -> None:
    protect = np.zeros(shape, dtype=bool)
    top, left = shape[0] // 2, shape[1] // 2
    protect[top, left] = True

    cleaned = clean_protect_mask(protect)

    assert cleaned[top - radius, left]
    assert cleaned[top + radius, left]
    assert cleaned[top, left - radius]
    assert cleaned[top, left + radius]
    assert not cleaned[top - radius - 1, left]
    assert not cleaned[top, left + radius + 1]
    assert not cleaned[top - radius, left - radius]


def test_clean_protect_respects_custom_area_and_growth_ratios() -> None:
    protect = np.zeros((400, 500), dtype=bool)
    protect[30:130, 30:130] = True
    protect[250:270, 250:270] = True

    cleaned = clean_protect_mask(protect, island_fraction=0.05, grow_ratio=0.02)

    assert not cleaned[260, 260]
    assert cleaned[80, 20]
    assert not cleaned[80, 19]


def test_clean_protect_preserves_an_empty_mask() -> None:
    protect = np.zeros((32, 48), dtype=bool)

    cleaned = clean_protect_mask(protect)

    np.testing.assert_array_equal(cleaned, protect)
    assert cleaned.dtype == np.bool_
