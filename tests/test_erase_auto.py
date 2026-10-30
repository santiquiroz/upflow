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
    assert choices[0]["engine"] == "lama"
    assert choices[0]["index"] == 0
    assert len(choices[0]["box"]) == 4
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

    assert [choice["engine"] for choice in choices] == ["lama", "lama"]
    assert [choice["index"] for choice in choices] == [0, 1]
    assert all(len(choice["box"]) == 4 for choice in choices)
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
    assert choices[0]["engine"] == "osor"
    assert choices[0]["guard"] is None
    assert choices[0]["residualPerson"] is False
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
        find_people=lambda _rgb, _mask: [],
    )

    lama.assert_called_once()
    assert len(osor_inputs) == 2
    np.testing.assert_array_equal(osor_inputs[1], lama_outputs[0])
    assert osor_inputs[1][160, 160].tolist() == [90, 90, 90]
    assert choices[0]["engine"] == "lama+osor"
    assert choices[0]["guard"] == "face"
    assert choices[0]["residualPerson"] is True
    assert np.asarray(result)[160, 160].tolist() == [120, 120, 120]


def test_large_group_person_guard_retries_with_lama_and_flags_residual() -> None:
    image, mask = scene((100, 100, 220, 220))
    lama = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 90))
    osor = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 120))
    find_people = Mock(return_value=[(110, 110, 200, 200)])

    _, choices = auto_erase(
        image, mask, None, lama, Mock(), small_side=100, osor_fill=osor,
        find_faces=lambda _rgb, _mask: [], find_people=find_people,
    )

    assert lama.call_count == 1
    assert osor.call_count == 2
    assert find_people.call_count == 2
    assert choices[0]["engine"] == "lama+osor"
    assert choices[0]["guard"] == "person"
    assert choices[0]["residualPerson"] is True


def test_large_group_without_face_or_person_keeps_osor_without_residual() -> None:
    image, mask = scene((100, 100, 220, 220))
    lama = Mock()
    osor = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 120))

    _, choices = auto_erase(
        image, mask, None, lama, Mock(), small_side=100, osor_fill=osor,
        find_faces=lambda _rgb, _mask: [],
        find_people=lambda _rgb, _mask: [],
    )

    lama.assert_not_called()
    osor.assert_called_once()
    assert choices[0]["engine"] == "osor"
    assert choices[0]["guard"] is None
    assert choices[0]["residualPerson"] is False


def test_face_guard_takes_precedence_when_faces_and_people_are_found() -> None:
    image, mask = scene((100, 100, 220, 220))
    faces = Mock(return_value=[(110, 110, 20, 20)])
    people = Mock(return_value=[(110, 110, 100, 100)])
    osor = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 120))

    _, choices = auto_erase(
        image, mask, None, Mock(side_effect=lambda canvas, group, protect: fill_with(
            canvas, group, 90
        )), Mock(), small_side=100, osor_fill=osor,
        find_faces=faces, find_people=people,
    )

    assert choices[0]["guard"] == "face"
    assert choices[0]["residualPerson"] is True
    assert faces.call_count == 2
    assert people.call_count == 2


def test_choices_carry_group_index_and_json_serializable_crop_box() -> None:
    image = Image.new("RGB", (2400, 320), (180, 180, 180))
    mask = np.zeros((320, 2400), dtype=np.uint8)
    mask[120:180, 100:160] = 255
    mask[120:180, 2200:2260] = 255
    lama = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 90))

    _, choices = auto_erase(image, mask, None, lama, Mock())

    assert [choice["index"] for choice in choices] == [0, 1]
    for choice in choices:
        box = choice["box"]
        assert isinstance(box, list)
        assert len(box) == 4
        assert all(type(coordinate) is int for coordinate in box)
        assert box[0] < box[2]
        assert box[1] < box[3]


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


def test_forced_lama_applies_to_large_groups() -> None:
    image, mask = scene((100, 100, 220, 220))
    lama = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 90))
    osor = Mock(side_effect=AssertionError("lama forzado no debe llamar a OSOR"))
    sdxl = Mock(side_effect=AssertionError("lama forzado no debe llamar a SDXL"))

    result, choices = auto_erase(
        image, mask, None, lama, sdxl, small_side=100, osor_fill=osor, engine="lama"
    )

    lama.assert_called_once()
    osor.assert_not_called()
    sdxl.assert_not_called()
    assert choices[0]["engine"] == "lama"
    assert np.asarray(result)[160, 160].tolist() == [90, 90, 90]


def test_forced_engine_never_switches_but_flags_residual() -> None:
    image, mask = scene((100, 100, 220, 220))
    osor = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 120))

    _, choices = auto_erase(
        image, mask, None, Mock(), Mock(), small_side=100, osor_fill=osor,
        find_faces=lambda _rgb, _mask: [(110, 110, 20, 20)],
        find_people=lambda _rgb, _mask: [(110, 110, 100, 100)],
        engine="osor",
    )

    osor.assert_called_once()
    assert choices[0]["engine"] == "osor"
    assert choices[0]["guard"] is None
    assert choices[0]["residualPerson"] is True


def test_forced_engine_composites_protected_subject_before_detection() -> None:
    image, mask = scene((100, 100, 220, 220))
    protect = np.zeros(mask.shape, dtype=bool)
    protect[140:180, 140:180] = True
    detected_images = []

    def detect(rgb, _mask):
        detected_images.append(rgb.copy())
        return []

    lama = Mock(side_effect=lambda canvas, group, core: np.full_like(canvas, 90))
    result, _ = auto_erase(
        image, mask, protect, lama, Mock(), find_faces=detect,
        find_people=detect, engine="lama",
    )

    original = np.asarray(image)
    output = np.asarray(result)
    cleaned = clean_protect_mask(protect)
    core = matte_core(cleaned, matte_radius(mask.shape))
    np.testing.assert_array_equal(output[core], original[core])
    assert len(detected_images) == 2
    for detected in detected_images:
        np.testing.assert_array_equal(detected, output)
        np.testing.assert_array_equal(detected[core], original[core])


def test_forced_osor_applies_to_small_groups() -> None:
    image, mask = scene()
    osor = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 120))

    _, choices = auto_erase(
        image, mask, None, Mock(), Mock(), osor_fill=osor, engine="osor"
    )

    osor.assert_called_once()
    assert choices[0]["engine"] == "osor"


def test_forced_osor_without_fill_raises() -> None:
    image, mask = scene()

    with pytest.raises(ValueError, match="OSOR"):
        auto_erase(image, mask, None, Mock(), Mock(), engine="osor")


def test_forced_sdxl_without_installed_edit_raises() -> None:
    image, mask = scene((100, 100, 220, 220))

    with pytest.raises(ValueError, match="SDXL"):
        auto_erase(
            image,
            mask,
            None,
            Mock(),
            None,
            engine="sdxl",
            sdxl_edit_factory=lambda: None,
        )


def test_forced_lama_osor_runs_both_and_sdxl_uses_full_strength() -> None:
    image, mask = scene()
    lama = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 90))
    osor = Mock(side_effect=lambda canvas, group, protect: fill_with(canvas, group, 120))

    _, choices = auto_erase(
        image, mask, None, lama, Mock(), osor_fill=osor, engine="lama+osor"
    )

    lama.assert_called_once()
    osor.assert_called_once()
    assert choices[0]["engine"] == "lama+osor"

    sdxl = Mock(return_value=Image.new("RGB", image.size, (150, 150, 150)))
    _, choices = auto_erase(image, mask, None, Mock(), sdxl, engine="sdxl")

    sdxl.assert_called_once()
    assert sdxl.call_args.args[2] == 1.0
    assert choices[0]["engine"] == "sdxl"
