from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.services.object_transfer import (
    DEFAULT_HARMONIZE_BLEND,
    HARMONIZE_SEAM_MIN_PX,
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
    # el borde del pegado va al máximo (con el perfil continuo de la fase 2 el
    # centro ya no: ver test_harmonization_mask_peaks_at_the_seam_...)
    assert m[30, 50] > 200 and m[69, 50] > 200
    assert m[50, 30] > 200 and m[50, 69] > 200
    # la dilatación (max(8, 10% de 40) = 8 px) también está cubierta
    assert m[25, 50] == 255
    assert m[50, 75] == 255
    # lejos del objeto, cero
    assert m[0, 0] == 0
    assert m[99, 99] == 0


# --- (e2) F2: perfil continuo de armonizacion -------------------------------
#
# El objetivo de la fase 2: la mascara ya no dice "regenera todo esto" sino
# "cuanto regenerar en cada pixel". Maxima en la costura (que es lo unico que
# delata el pegado) y casi nula en el centro del objeto (que no tiene nada roto
# que arreglar y es justo donde se perderia la identidad de lo pegado).

PASTE_AT = 90
OBJECT_SIDE = 120
CANVAS = (300, 300)


def harmonize_mask_for(blend: float) -> np.ndarray:
    """Mascara de armonizacion de un cuadrado opaco pegado en el centro."""
    source = solid((OBJECT_SIDE, OBJECT_SIDE), OBJECT_COLOR)
    mask = rect_mask((OBJECT_SIDE, OBJECT_SIDE), (0, 0, OBJECT_SIDE, OBJECT_SIDE))
    target = solid(CANVAS, TARGET_COLOR)
    spec = PasteSpec(
        x=PASTE_AT, y=PASTE_AT, width=OBJECT_SIDE, height=OBJECT_SIDE,
        feather_px=0, match_color=False, harmonize_blend=blend,
    )
    _, harmonize = transfer_object(source, mask, target, spec)
    return np.asarray(harmonize)


def test_harmonization_mask_peaks_at_the_seam_and_vanishes_at_the_center() -> None:
    m = harmonize_mask_for(DEFAULT_HARMONIZE_BLEND)

    center = PASTE_AT + OBJECT_SIDE // 2
    # el centro del objeto se preserva: la difusion diferencial re-inyecta el
    # original ahi en cada paso
    assert m[center, center] == 0
    # la costura (el borde, por afuera y por adentro) es lo que se regenera
    assert m[center, PASTE_AT - 4] == 255
    assert m[center, PASTE_AT + 2] > 200
    assert m.max() == 255


def test_harmonization_mask_decays_monotonically_from_the_seam_inward() -> None:
    m = harmonize_mask_for(DEFAULT_HARMONIZE_BLEND)

    center = PASTE_AT + OBJECT_SIDE // 2
    inward = m[center, PASTE_AT : center + 1].astype(int)
    assert (np.diff(inward) <= 0).all()
    assert inward[0] > 200 and inward[-1] == 0
    # una rampa de verdad, no un escalon con dos valores
    assert len({int(v) for v in inward if 0 < v < 255}) >= 5


def test_harmonization_blend_one_reproduces_the_uniform_legacy_mask() -> None:
    m = harmonize_mask_for(1.0)

    # el contrato viejo: todo lo pegado a 255, mas la dilatacion, y cero lejos
    assert (m[PASTE_AT : PASTE_AT + OBJECT_SIDE, PASTE_AT : PASTE_AT + OBJECT_SIDE] == 255).all()
    assert m[PASTE_AT - 4, PASTE_AT + 60] == 255
    assert m[0, 0] == 0


def test_lower_blend_preserves_more_of_the_object() -> None:
    interior = (slice(PASTE_AT, PASTE_AT + OBJECT_SIDE), slice(PASTE_AT, PASTE_AT + OBJECT_SIDE))
    coverage = [harmonize_mask_for(blend)[interior].mean() for blend in (0.0, 0.2, 0.5, 1.0)]

    assert coverage == sorted(coverage)
    assert coverage[0] < coverage[-1]


def test_blend_zero_still_regenerates_the_minimum_seam_band() -> None:
    m = harmonize_mask_for(0.0)

    center = PASTE_AT + OBJECT_SIDE // 2
    # el piso en px existe para que un parametro en 0 no deje al modelo sin
    # banda con la que fundir
    assert m[center, PASTE_AT + 2] > 200
    assert m[center, PASTE_AT + HARMONIZE_SEAM_MIN_PX + 2] == 0


def test_edge_touching_the_target_border_is_not_treated_as_a_seam() -> None:
    """Un objeto cortado por el borde del destino no tiene costura ahi.

    Del otro lado no hay nada con que fundir: esos pixeles cuentan como
    profundos y se preservan, igual que el centro.
    """
    source = solid((120, 300), OBJECT_COLOR)
    mask = rect_mask((120, 300), (0, 0, 120, 300))
    target = solid(CANVAS, TARGET_COLOR)
    spec = PasteSpec(
        x=0, y=0, width=120, height=300,
        feather_px=0, match_color=False, harmonize_blend=DEFAULT_HARMONIZE_BLEND,
    )

    _, harmonize = transfer_object(source, mask, target, spec)

    m = np.asarray(harmonize)
    # pegado al borde izquierdo de la foto: preservado
    assert m[150, 2] == 0
    # el borde derecho, que si limita con el destino, es costura
    assert m[150, 118] > 200


def test_harmonize_blend_does_not_change_the_composite() -> None:
    source = solid((OBJECT_SIDE, OBJECT_SIDE), OBJECT_COLOR)
    mask = rect_mask((OBJECT_SIDE, OBJECT_SIDE), (0, 0, OBJECT_SIDE, OBJECT_SIDE))
    target = noisy(CANVAS, (200, 80, 60), 10, seed=23)
    base = dict(x=PASTE_AT, y=PASTE_AT, width=OBJECT_SIDE, height=OBJECT_SIDE, feather_px=4)

    soft, _ = transfer_object(source, mask, target, PasteSpec(**base, harmonize_blend=0.2))
    legacy, _ = transfer_object(source, mask, target, PasteSpec(**base, harmonize_blend=1.0))

    np.testing.assert_array_equal(np.asarray(soft), np.asarray(legacy))


@pytest.mark.parametrize("blend", [-0.1, 1.5])
def test_out_of_range_blend_raises(blend: float) -> None:
    source = solid((40, 40), OBJECT_COLOR)
    mask = rect_mask((40, 40), (0, 0, 40, 40))
    target = solid((100, 100), TARGET_COLOR)

    with pytest.raises(ValueError):
        transfer_object(
            source, mask, target,
            PasteSpec(x=10, y=10, width=40, height=40, harmonize_blend=blend),
        )


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
