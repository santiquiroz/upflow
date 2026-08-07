from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.services.inpaint_pipeline import (
    MaskedEditSettings,
    resolve_model_side,
    run_masked_edit,
)


def base_photo(size: tuple[int, int] = (900, 600)) -> Image.Image:
    width, height = size
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    gradient[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    gradient[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    gradient[:, :, 2] = 90
    return Image.fromarray(gradient)


def mask_with_box(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = 255
    return Image.fromarray(mask, mode="L")


class RecordingModel:
    """Sustituto del pipeline de difusión: devuelve rojo puro y registra qué recibió."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, image: Image.Image, mask: Image.Image, width: int, height: int) -> Image.Image:
        self.calls.append(
            {"image_size": image.size, "mask_size": mask.size, "width": width, "height": height}
        )
        return Image.new("RGB", (width, height), (255, 0, 0))


SETTINGS = MaskedEditSettings(
    dilate_px=8, feather_px=8, padding_px=48, target_side=512, native_side=512
)


def test_output_keeps_the_original_resolution() -> None:
    photo = base_photo((900, 600))
    mask = mask_with_box((900, 600), (400, 300, 460, 340))

    result = run_masked_edit(photo, mask, RecordingModel(), SETTINGS)

    assert result.size == (900, 600)


def test_the_model_only_sees_the_marked_region_at_its_native_resolution() -> None:
    photo = base_photo((900, 600))
    mask = mask_with_box((900, 600), (400, 300, 460, 340))
    model = RecordingModel()

    run_masked_edit(photo, mask, model, SETTINGS)

    assert len(model.calls) == 1
    call = model.calls[0]
    # El recorte chico se lleva al piso nativo del modelo, no la imagen entera.
    assert call["width"] == 512 and call["height"] == 512
    assert call["image_size"] == (512, 512)
    assert call["mask_size"] == (512, 512)


def test_pixels_far_from_the_mask_are_untouched() -> None:
    photo = base_photo((900, 600))
    mask = mask_with_box((900, 600), (400, 300, 460, 340))
    before = np.asarray(photo)

    after = np.asarray(run_masked_edit(photo, mask, RecordingModel(), SETTINGS))

    np.testing.assert_array_equal(after[0:100, 0:100], before[0:100, 0:100])
    np.testing.assert_array_equal(after[500:600, 800:900], before[500:600, 800:900])


def test_the_marked_region_actually_changes() -> None:
    photo = base_photo((900, 600))
    mask = mask_with_box((900, 600), (400, 300, 460, 340))

    after = np.asarray(run_masked_edit(photo, mask, RecordingModel(), SETTINGS))

    center = after[320, 430]
    assert center[0] > 200 and center[1] < 60


def test_the_seam_is_a_gradient_and_not_a_step() -> None:
    photo = base_photo((900, 600))
    mask = mask_with_box((900, 600), (400, 300, 460, 340))

    after = np.asarray(run_masked_edit(photo, mask, RecordingModel(), SETTINGS))

    # Franja horizontal que cruza el borde derecho del área marcada.
    strip = after[320, 455:485, 0].astype(int)
    steps = np.abs(np.diff(strip))
    assert steps.max() < 120, "un salto grande significa costura dura"
    assert len(set(strip.tolist())) > 3, "debería haber una transición progresiva"


def test_a_mask_touching_the_border_still_works() -> None:
    photo = base_photo((512, 512))
    mask = mask_with_box((512, 512), (0, 0, 60, 60))

    result = run_masked_edit(photo, mask, RecordingModel(), SETTINGS)

    assert result.size == (512, 512)
    assert np.asarray(result)[10, 10][0] > 200


def test_a_mask_covering_everything_falls_back_to_the_whole_image() -> None:
    photo = base_photo((640, 640))
    mask = mask_with_box((640, 640), (0, 0, 640, 640))
    model = RecordingModel()

    result = run_masked_edit(photo, mask, model, SETTINGS)

    assert result.size == (640, 640)
    assert model.calls[0]["width"] == 512


def test_an_empty_mask_returns_the_original_untouched() -> None:
    photo = base_photo((320, 240))
    empty = mask_with_box((320, 240), (0, 0, 0, 0))
    model = RecordingModel()

    result = run_masked_edit(photo, empty, model, SETTINGS)

    np.testing.assert_array_equal(np.asarray(result), np.asarray(photo))
    assert model.calls == []


def test_dilation_widens_what_gets_replaced() -> None:
    photo = base_photo((512, 512))
    box = (200, 200, 240, 240)
    mask = mask_with_box((512, 512), box)

    without = np.asarray(
        run_masked_edit(photo, mask, RecordingModel(), MaskedEditSettings(0, 0, 48, 512))
    )
    with_dilation = np.asarray(
        run_masked_edit(photo, mask, RecordingModel(), MaskedEditSettings(12, 0, 48, 512))
    )

    # Justo afuera del trazo: sin dilatar sigue original, dilatado ya fue reemplazado.
    assert int(without[220, 245, 0]) < 200
    assert int(with_dilation[220, 245, 0]) > 200


def test_target_side_is_snapped_to_a_multiple_of_64() -> None:
    photo = base_photo((900, 600))
    mask = mask_with_box((900, 600), (400, 300, 460, 340))
    model = RecordingModel()

    run_masked_edit(photo, mask, model, MaskedEditSettings(8, 8, 48, 700))

    assert model.calls[0]["width"] % 64 == 0


class TestResolveModelSide:
    """(trabajo, canvas) puros: piso nativo, techo de edición y buckets de 128."""

    def test_a_small_crop_is_floored_to_the_native_side(self) -> None:
        assert resolve_model_side(300, 1024, 512) == (512, 512)

    def test_a_crop_at_or_above_native_keeps_its_own_size(self) -> None:
        assert resolve_model_side(700, 1024, 512) == (700, 768)

    def test_without_native_side_small_crops_are_not_upscaled(self) -> None:
        assert resolve_model_side(300, 1024, None) == (300, 384)

    def test_the_target_ceiling_still_caps_big_crops(self) -> None:
        assert resolve_model_side(900, 512, None) == (512, 512)

    def test_the_native_floor_wins_over_the_target_ceiling(self) -> None:
        # El piso dice dónde rinde el modelo; un techo menor no lo anula.
        assert resolve_model_side(300, 512, 1024) == (1024, 1024)

    def test_the_canvas_never_goes_below_the_minimum_bucket(self) -> None:
        assert resolve_model_side(10, 100, None) == (10, 128)


def test_a_small_crop_is_diffused_at_the_native_side_and_stitched_back() -> None:
    photo = base_photo((900, 600))
    mask = mask_with_box((900, 600), (400, 300, 460, 340))
    model = RecordingModel()
    settings = MaskedEditSettings(8, 8, 48, 1024, native_side=512)

    result = run_masked_edit(photo, mask, model, settings)

    call = model.calls[0]
    # El recorte mide ~190 px: sin piso se difundiría ahí y saldría blando.
    assert call["width"] == 512 and call["height"] == 512
    assert call["image_size"] == (512, 512) and call["mask_size"] == (512, 512)
    assert result.size == (900, 600)
    after = np.asarray(result)
    assert after[320, 430][0] > 200, "la zona marcada tiene que quedar editada"
    np.testing.assert_array_equal(after[0:100, 0:100], np.asarray(photo)[0:100, 0:100])


def test_a_crop_already_larger_than_native_is_left_alone() -> None:
    photo = base_photo((1400, 1000))
    mask = mask_with_box((1400, 1000), (200, 200, 800, 800))
    with_floor, without_floor = RecordingModel(), RecordingModel()

    result_with = run_masked_edit(
        photo, mask, with_floor, MaskedEditSettings(8, 8, 48, 1024, native_side=512)
    )
    result_without = run_masked_edit(
        photo, mask, without_floor, MaskedEditSettings(8, 8, 48, 1024)
    )

    assert with_floor.calls == without_floor.calls
    assert with_floor.calls[0]["width"] > 512, "no debería clavarse al piso"
    np.testing.assert_array_equal(np.asarray(result_with), np.asarray(result_without))


def test_native_side_none_behaves_exactly_like_before() -> None:
    photo = base_photo((900, 600))
    mask = mask_with_box((900, 600), (400, 300, 460, 340))
    by_default, explicit_none, floor_below_crop = (
        RecordingModel(),
        RecordingModel(),
        RecordingModel(),
    )

    result_default = run_masked_edit(
        photo, mask, by_default, MaskedEditSettings(8, 8, 48, 512)
    )
    result_none = run_masked_edit(
        photo, mask, explicit_none, MaskedEditSettings(8, 8, 48, 512, native_side=None)
    )
    result_low = run_masked_edit(
        photo, mask, floor_below_crop, MaskedEditSettings(8, 8, 48, 512, native_side=128)
    )

    assert by_default.calls == explicit_none.calls == floor_below_crop.calls
    np.testing.assert_array_equal(np.asarray(result_default), np.asarray(result_none))
    np.testing.assert_array_equal(np.asarray(result_default), np.asarray(result_low))


@pytest.mark.parametrize(
    "box", [(400, 300, 460, 340), (300, 200, 600, 500), (100, 100, 700, 560)]
)
def test_model_dimensions_land_on_buckets_of_128(box: tuple[int, int, int, int]) -> None:
    photo = base_photo((900, 600))
    mask = mask_with_box((900, 600), box)
    model = RecordingModel()

    run_masked_edit(photo, mask, model, MaskedEditSettings(8, 8, 48, 1024))

    call = model.calls[0]
    assert call["width"] % 128 == 0 and call["height"] % 128 == 0
    assert call["image_size"] == (call["width"], call["height"])
    assert call["mask_size"] == (call["width"], call["height"])


def test_the_stitch_is_cut_back_to_the_exact_crop_despite_the_bigger_canvas() -> None:
    photo = base_photo((1400, 1000))
    mask = mask_with_box((1400, 1000), (200, 200, 800, 800))
    model = RecordingModel()

    result = run_masked_edit(
        photo, mask, model, MaskedEditSettings(8, 8, 48, 1024, native_side=512)
    )

    # El recorte mide ~726 px; el canvas bucketea a 768 y el sobrante se tira.
    assert model.calls[0]["width"] == 768
    assert result.size == (1400, 1000)
    before, after = np.asarray(photo), np.asarray(result)
    np.testing.assert_array_equal(after[0:80, 0:80], before[0:80, 0:80])
    np.testing.assert_array_equal(after[920:1000, 1320:1400], before[920:1000, 1320:1400])
    assert after[500, 500][0] > 200, "el centro de lo marcado sí cambió"
