from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.services.object_transfer import (
    PasteSpec,
    crop_object,
    match_color_mk,
    transfer_object,
)

OBJECT_COLOR = (10, 200, 30)
TARGET_COLOR = (0, 0, 128)


def solid(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def rect_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = 255
    return Image.fromarray(mask, mode="L")


def noisy(size: tuple[int, int], mean: tuple[float, float, float], spread: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.normal(mean, spread, (size[1], size[0], 3)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


# --- (a) paste opaco --------------------------------------------------------


def test_opaque_paste_keeps_object_and_target_intact() -> None:
    source = solid((60, 60), OBJECT_COLOR)
    mask = rect_mask((60, 60), (10, 10, 50, 50))
    target = solid((100, 100), TARGET_COLOR)
    spec = PasteSpec(x=30, y=30, width=40, height=40, feather_px=0, match_color=False)

    composed, _ = transfer_object(source, mask, target, spec)

    arr = np.asarray(composed)
    # el rectángulo pegado es el objeto, esquina a esquina
    assert tuple(arr[30, 30]) == OBJECT_COLOR
    assert tuple(arr[50, 50]) == OBJECT_COLOR
    assert tuple(arr[69, 69]) == OBJECT_COLOR
    # fuera del pegado, el destino queda bit a bit
    assert tuple(arr[10, 10]) == TARGET_COLOR
    assert tuple(arr[29, 50]) == TARGET_COLOR
    assert tuple(arr[80, 80]) == TARGET_COLOR


# --- (b) feather ------------------------------------------------------------


def test_feather_transition_is_monotonic_without_hard_step() -> None:
    source = solid((60, 60), (255, 255, 255))
    mask = rect_mask((60, 60), (0, 0, 60, 60))
    target = solid((100, 100), (0, 0, 0))
    spec = PasteSpec(x=20, y=20, width=40, height=40, feather_px=6, match_color=False)

    composed, _ = transfer_object(source, mask, target, spec)

    # fila por el centro del pegado, entrando desde afuera hasta el centro
    row = np.asarray(composed)[40, 10:41, 0].astype(int)
    assert row[0] == 0
    assert row[-1] == 255
    assert (np.diff(row) >= 0).all()
    intermediate = set(row[(row > 0) & (row < 255)])
    assert len(intermediate) >= 3


# --- (c) fuera de límites ---------------------------------------------------


def test_partial_out_of_bounds_paste_composites_only_visible_part() -> None:
    source = solid((40, 40), OBJECT_COLOR)
    mask = rect_mask((40, 40), (0, 0, 40, 40))
    target = solid((100, 100), TARGET_COLOR)
    spec = PasteSpec(x=-20, y=-20, width=50, height=50, feather_px=0, match_color=False)

    composed, pasted_mask = transfer_object(source, mask, target, spec)

    arr = np.asarray(composed)
    assert tuple(arr[0, 0]) == OBJECT_COLOR
    assert tuple(arr[29, 29]) == OBJECT_COLOR
    assert tuple(arr[35, 35]) == TARGET_COLOR
    assert pasted_mask.size == (100, 100)


def test_fully_out_of_bounds_paste_returns_target_and_empty_mask() -> None:
    source = solid((40, 40), OBJECT_COLOR)
    mask = rect_mask((40, 40), (0, 0, 40, 40))
    target = solid((100, 100), TARGET_COLOR)
    spec = PasteSpec(x=200, y=200, width=30, height=30)

    composed, pasted_mask = transfer_object(source, mask, target, spec)

    np.testing.assert_array_equal(np.asarray(composed), np.asarray(target))
    assert np.asarray(pasted_mask).max() == 0


# --- (d) match_color_mk -----------------------------------------------------


def test_match_color_mk_warms_gray_object_toward_reddish_region() -> None:
    rng = np.random.default_rng(7)
    object_rgb = rng.normal(128, 10, (20, 20, 3)).clip(0, 255).astype(np.float32)
    alpha = np.ones((20, 20), dtype=np.float32)
    reddish = rng.normal((200, 80, 60), (12, 10, 8), (400, 3)).clip(0, 255).astype(np.float32)

    mapped = match_color_mk(object_rgb, alpha, reddish)

    assert mapped.shape == object_rgb.shape
    assert mapped[..., 0].mean() > object_rgb[..., 0].mean() + 20
    assert mapped[..., 1].mean() < object_rgb[..., 1].mean()


def test_match_color_mk_degenerate_covariance_does_not_crash() -> None:
    flat_object = np.full((10, 10, 3), 128.0, dtype=np.float32)
    alpha = np.ones((10, 10), dtype=np.float32)
    rng = np.random.default_rng(3)
    reddish = rng.normal((200, 80, 60), (12, 10, 8), (200, 3)).clip(0, 255).astype(np.float32)

    mapped = match_color_mk(flat_object, alpha, reddish)

    assert np.isfinite(mapped).all()
    assert mapped.min() >= 0.0 and mapped.max() <= 255.0
    assert mapped[..., 0].mean() > 128.0


def test_transfer_with_match_color_warms_pasted_area() -> None:
    source = noisy((60, 60), (128, 128, 128), 10, seed=11)
    mask = rect_mask((60, 60), (0, 0, 60, 60))
    target = noisy((100, 100), (200, 80, 60), 10, seed=13)
    base_spec = dict(x=30, y=30, width=40, height=40, feather_px=0)

    with_color, _ = transfer_object(source, mask, target, PasteSpec(**base_spec, match_color=True))
    without_color, _ = transfer_object(source, mask, target, PasteSpec(**base_spec, match_color=False))

    red_with = np.asarray(with_color)[30:70, 30:70, 0].mean()
    red_without = np.asarray(without_color)[30:70, 30:70, 0].mean()
    assert red_with > red_without + 20


# --- (e) máscara devuelta ---------------------------------------------------


def test_returned_mask_covers_dilated_paste_and_is_zero_far_away() -> None:
    source = solid((40, 40), OBJECT_COLOR)
    mask = rect_mask((40, 40), (0, 0, 40, 40))
    target = solid((100, 100), TARGET_COLOR)
    spec = PasteSpec(x=30, y=30, width=40, height=40, feather_px=0, match_color=False)

    _, pasted_mask = transfer_object(source, mask, target, spec)

    m = np.asarray(pasted_mask)
    assert pasted_mask.mode == "L"
    # el bbox del pegado entero queda a 255
    assert (m[30:70, 30:70] == 255).all()
    # la dilatación (max(8, 10% de 40) = 8 px) también está cubierta
    assert m[25, 50] == 255
    assert m[50, 75] == 255
    # lejos del objeto, cero
    assert m[0, 0] == 0
    assert m[99, 99] == 0


# --- (f) match_color=False --------------------------------------------------


def test_match_color_false_keeps_object_colors_exact() -> None:
    source = solid((40, 40), OBJECT_COLOR)
    mask = rect_mask((40, 40), (0, 0, 40, 40))
    target = noisy((100, 100), (200, 80, 60), 10, seed=17)
    spec = PasteSpec(x=10, y=10, width=40, height=40, feather_px=0, match_color=False)

    composed, _ = transfer_object(source, mask, target, spec)

    pasted = np.asarray(composed)[10:50, 10:50]
    assert (pasted == OBJECT_COLOR).all()


# --- (g) invariante de tamaño -----------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        PasteSpec(x=30, y=30, width=40, height=40),
        PasteSpec(x=5, y=80, width=13, height=27),  # escala no uniforme
        PasteSpec(x=-15, y=-15, width=50, height=50),
        PasteSpec(x=90, y=90, width=30, height=30),
        PasteSpec(x=300, y=300, width=20, height=20),  # completamente afuera
    ],
)
def test_output_sizes_always_match_target(spec: PasteSpec) -> None:
    source = solid((60, 60), OBJECT_COLOR)
    mask = rect_mask((60, 60), (10, 10, 50, 50))
    target = solid((100, 100), TARGET_COLOR)

    composed, pasted_mask = transfer_object(source, mask, target, spec)

    assert composed.size == target.size
    assert pasted_mask.size == target.size
    assert composed.mode == "RGB"
    assert pasted_mask.mode == "L"


# --- bordes -----------------------------------------------------------------


def test_crop_object_with_empty_mask_raises() -> None:
    source = solid((30, 30), OBJECT_COLOR)
    empty = rect_mask((30, 30), (0, 0, 0, 0))
    with pytest.raises(ValueError):
        crop_object(source, empty)


def test_mask_size_mismatch_raises() -> None:
    source = solid((30, 30), OBJECT_COLOR)
    mask = rect_mask((40, 40), (0, 0, 40, 40))
    target = solid((100, 100), TARGET_COLOR)
    with pytest.raises(ValueError):
        transfer_object(source, mask, target, PasteSpec(x=0, y=0, width=10, height=10))
