from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.services.inpaint_pipeline import MaskedEditSettings, run_masked_edit


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


SETTINGS = MaskedEditSettings(dilate_px=8, feather_px=8, padding_px=48, target_side=512)


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
    # El recorte se lleva a la resolución nativa del modelo, no la imagen entera.
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
