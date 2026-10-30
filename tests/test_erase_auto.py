from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from app.services.erase_auto import auto_erase
from app.services.inpaint_mask import (
    clean_protect_mask,
    grow_per_component,
    grow_small_hulls,
    matte_core,
    matte_radius,
    subject_matte,
)


def scene(box=(120, 120, 200, 200)):
    rgb = np.full((320, 320, 3), 180, dtype=np.uint8)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = 255
    return Image.fromarray(rgb), mask


def fill_with(canvas, mask, value):
    result = canvas.copy()
    result[mask > 0] = value
    return result


def test_small_group_calls_only_lama_with_the_prepared_mask() -> None:
    image, mask = scene()
    lama = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 90))
    sdxl = Mock(side_effect=AssertionError("SDXL no debe editar grupos pequeños"))

    result, choices = auto_erase(image, mask, None, lama, sdxl)

    lama.assert_called_once()
    sdxl.assert_not_called()
    expected_mask = grow_per_component(grow_small_hulls(mask), None, None)
    np.testing.assert_array_equal(lama.call_args.args[1], expected_mask)
    assert not lama.call_args.args[2].any()
    assert choices == [{"engine": "lama"}]
    assert np.asarray(result)[160, 160].tolist() == [90, 90, 90]


@pytest.mark.parametrize("winning_strength,engine", [(0.5, "hibrido"), (1.0, "sdxl")])
def test_big_group_keeps_the_candidate_with_the_lower_measured_score(
    winning_strength, engine
) -> None:
    image, mask = scene((100, 100, 220, 220))
    lama = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 90))
    calls = []

    def sdxl(base, group, strength):
        calls.append((np.asarray(base).copy(), strength))
        if strength == winning_strength:
            return Image.fromarray(fill_with(np.asarray(base), group, 180))
        rows, columns = np.indices(group.shape)
        pattern = np.where((rows // 16 + columns // 16) % 2, 255, 105)
        return Image.fromarray(fill_with(np.asarray(base), group, pattern[group > 0, None]))

    result, choices = auto_erase(image, mask, None, lama, sdxl, small_side=100)

    lama.assert_called_once()
    assert [strength for _, strength in calls] == [0.5, 1.0]
    assert calls[0][0][160, 160].tolist() == [90, 90, 90]
    np.testing.assert_array_equal(calls[1][0], np.asarray(image))
    np.testing.assert_array_equal(np.asarray(result), np.asarray(image))
    assert choices[0]["engine"] == engine
    scores = choices[0]["scores"]
    other_engine = "sdxl" if engine == "hibrido" else "hibrido"
    assert scores[engine]["score"] < scores[other_engine]["score"]


@pytest.mark.parametrize("small_side", [700, 30])
def test_protected_pixels_survive_even_when_a_provider_overwrites_them(small_side) -> None:
    image, mask = scene((100, 100, 220, 220))
    protect = np.zeros(mask.shape, dtype=bool)
    protect[130:190, 130:190] = True
    cleaned = clean_protect_mask(protect)
    core = matte_core(cleaned, matte_radius(mask.shape))
    alpha = subject_matte(np.asarray(image), cleaned)
    calls = []

    def lama(canvas, group, guarded):
        calls.append((group.copy(), guarded.copy()))
        return np.full_like(canvas, 90)

    def sdxl(base, group, strength):
        assert not group[core].any()
        return Image.new("RGB", base.size, (90, 90, 90))

    result, _ = auto_erase(image, mask, protect, lama, sdxl, small_side=small_side)

    assert len(calls) == 1
    assert not calls[0][0][core].any()
    np.testing.assert_array_equal(calls[0][1], core)
    assert (alpha == 1.0).any()
    np.testing.assert_array_equal(np.asarray(result)[alpha == 1.0], np.asarray(image)[alpha == 1.0])


@pytest.mark.parametrize("small_side", [700, 30])
def test_overlong_protect_border_is_replaced_by_the_fill(small_side) -> None:
    original = np.full((320, 320, 3), 220, dtype=np.uint8)
    original[:, :160] = 30
    protect = np.zeros((320, 320), dtype=bool)
    protect[:, :157] = True
    cleaned = clean_protect_mask(protect)
    mask = np.zeros(protect.shape, dtype=np.uint8)
    mask[100:220, 140:210] = 255
    lama = Mock(side_effect=lambda canvas, group, guarded: fill_with(canvas, group, 90))
    sdxl = Mock(side_effect=lambda base, group, strength: Image.fromarray(
        fill_with(np.asarray(base), group, 90)
    ))

    result, _ = auto_erase(Image.fromarray(original), mask, protect, lama, sdxl, small_side=small_side)

    assert cleaned[160, 160:163].all()
    assert not cleaned[160, 163]
    assert np.all(lama.call_args.args[1][160, 160:163] > 0)
    border = np.asarray(result)[160, 160:163].astype(np.int16)
    assert np.all(border < original[160, 160:163])
    assert np.all(border >= 90)
    assert np.all(np.abs(border[-1] - 90) < np.abs(border[-1] - 220))
    np.testing.assert_array_equal(np.asarray(result)[160, 140], original[160, 140])


def test_empty_mask_returns_the_image_without_calling_providers() -> None:
    image, mask = scene()
    mask[:] = 0
    lama, sdxl = Mock(), Mock()

    result, choices = auto_erase(image, mask, None, lama, sdxl)

    np.testing.assert_array_equal(np.asarray(result), np.asarray(image))
    assert choices == []
    lama.assert_not_called()
    sdxl.assert_not_called()


def test_separate_groups_are_applied_sequentially_to_the_current_image() -> None:
    image = Image.new("RGB", (2400, 320), (180, 180, 180))
    mask = np.zeros((320, 2400), dtype=np.uint8)
    mask[120:180, 100:160] = 255
    mask[120:180, 2200:2260] = 255
    canvases = []

    def lama(canvas, group, protect):
        canvases.append(canvas.copy())
        return fill_with(canvas, group, 90)

    result, choices = auto_erase(image, mask, None, lama, Mock())

    assert choices == [{"engine": "lama"}, {"engine": "lama"}]
    assert len(canvases) == 2
    assert canvases[0][150, 130].tolist() == [180, 180, 180]
    assert canvases[1][150, 130].tolist() == [90, 90, 90]
    assert np.asarray(result)[150, 2230].tolist() == [90, 90, 90]


def test_large_group_uses_osor_without_calling_sdxl() -> None:
    image, mask = scene((100, 100, 220, 220))
    lama = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 90))
    osor = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 120))
    sdxl = Mock(side_effect=AssertionError("OSOR debe reemplazar SDXL"))

    result, choices = auto_erase(
        image, mask, None, lama, sdxl, small_side=100, osor_fill=osor,
        find_faces=lambda _rgb, _mask: [],
    )

    osor.assert_called_once()
    sdxl.assert_not_called()
    assert choices == [{"engine": "osor"}]
    assert np.asarray(result)[160, 160].tolist() == [120, 120, 120]


def test_large_group_face_guard_runs_lama_before_osor_again() -> None:
    image, mask = scene((100, 100, 220, 220))
    lama_outputs = []

    def lama_fill(canvas, group, protect):
        output = fill_with(canvas, group, 90)
        lama_outputs.append(output)
        return output

    lama = Mock(side_effect=lama_fill)
    osor_inputs = []

    def osor(canvas, group, protect):
        osor_inputs.append(canvas.copy())
        return fill_with(canvas, group, 120)

    result, choices = auto_erase(
        image, mask, None, lama, Mock(), small_side=100, osor_fill=osor,
        find_faces=lambda _rgb, _mask: [(1, 1, 10, 10)],
    )

    lama.assert_called_once()
    assert len(osor_inputs) == 2
    np.testing.assert_array_equal(osor_inputs[1], lama_outputs[0])
    assert osor_inputs[1][160, 160].tolist() == [90, 90, 90]
    assert choices == [{"engine": "lama+osor", "face_guard": True}]
    assert np.asarray(result)[160, 160].tolist() == [120, 120, 120]


def test_large_group_without_osor_keeps_existing_candidate_scoring() -> None:
    image, mask = scene((100, 100, 220, 220))
    lama = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 90))
    sdxl = Mock(side_effect=lambda base, group, strength: Image.fromarray(
        fill_with(np.asarray(base), group, 150)
    ))

    _, choices = auto_erase(image, mask, None, lama, sdxl, small_side=100)

    assert choices[0]["engine"] in {"hibrido", "sdxl"}
    assert "scores" in choices[0]
    assert sdxl.call_count == 2
