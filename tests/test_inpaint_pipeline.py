from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.services.inpaint_pipeline import (
    MaskedEditSettings,
    resolve_model_dims,
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


def test_a_prepared_gradient_mask_reaches_the_model_intact_without_dilation() -> None:
    """La máscara de armonización de objetos ya viene con su perfil.

    El motor la recibe con dilate/feather en 0 justamente para no pisarlo: la
    difusión diferencial necesita el degradado tal cual (máximo en la costura,
    casi nulo en el centro) para saber cuánto re-generar en cada píxel.
    """
    from app.services.object_transfer import PasteSpec, harmonization_mask

    photo = base_photo((512, 512))
    alpha = np.zeros((200, 200), dtype=np.float32)
    alpha[:, :] = 1.0
    prepared = harmonization_mask(
        (512, 512), alpha, PasteSpec(x=150, y=150, width=200, height=200, harmonize_blend=0.35)
    )
    seen: list[Image.Image] = []

    run_masked_edit(
        photo, prepared, RecordingModel(),
        MaskedEditSettings(dilate_px=0, feather_px=0, padding_px=48, target_side=512),
        on_prepared=lambda _image, mask: seen.append(mask),
    )

    received = np.asarray(seen[0].convert("L"))
    # el degradado llega vivo: hay muchos niveles intermedios, no un binario
    assert len(np.unique(received)) > 20
    assert received.max() == 255
    assert received.min() == 0


def test_engine_dilation_would_flatten_a_prepared_gradient_mask() -> None:
    """Por qué la armonización manda 0/0 y no los defaults del motor."""
    from app.services.object_transfer import PasteSpec, harmonization_mask

    photo = base_photo((512, 512))
    alpha = np.ones((200, 200), dtype=np.float32)
    prepared = harmonization_mask(
        (512, 512), alpha, PasteSpec(x=150, y=150, width=200, height=200, harmonize_blend=0.35)
    )
    masks: dict[str, np.ndarray] = {}

    for name, settings in (
        ("prepared", MaskedEditSettings(0, 0, 48, 512)),
        ("dilated", MaskedEditSettings(8, 8, 48, 512)),
    ):
        run_masked_edit(
            photo, prepared, RecordingModel(), settings,
            on_prepared=lambda _i, m, key=name: masks.__setitem__(key, np.asarray(m.convert("L"))),
        )

    # el MaxFilter del motor empuja el 255 de la costura hacia adentro y sube el
    # centro preservado: el perfil continuo se aplana
    assert masks["dilated"].mean() > masks["prepared"].mean()


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


class TestResolveModelDims:
    """Generalización rectangular: piso y techo sobre el LADO MAYOR, aspecto
    preservado con escala uniforme, buckets de 128 por eje."""

    def test_a_square_crop_matches_the_scalar_path(self) -> None:
        for crop_side, target, native in [(300, 1024, 512), (700, 1024, 512), (900, 512, None)]:
            work, canvas = resolve_model_side(crop_side, target, native)
            assert resolve_model_dims((crop_side, crop_side), target, native) == (
                (work, work),
                (canvas, canvas),
            )

    def test_the_native_floor_scales_uniformly_preserving_aspect(self) -> None:
        (work_w, work_h), (canvas_w, canvas_h) = resolve_model_dims((600, 300), 1024, 1024)

        assert (work_w, work_h) == (1024, 512)
        assert (canvas_w, canvas_h) == (1024, 512)

    def test_the_target_ceiling_scales_uniformly_preserving_aspect(self) -> None:
        (work_w, work_h), (canvas_w, canvas_h) = resolve_model_dims((1532, 1080), 512, 512)

        assert work_w == 512
        assert work_h / work_w == pytest.approx(1080 / 1532, abs=0.01)
        assert canvas_w % 128 == 0 and canvas_h % 128 == 0

    def test_the_canvas_covers_the_work_on_both_axes(self) -> None:
        (work_w, work_h), (canvas_w, canvas_h) = resolve_model_dims((800, 256), 1024, 1024)

        assert canvas_w >= work_w and canvas_h >= work_h


# --- marcas mas largas que la dimension menor de la imagen (bug 2026-08-07) --


class TestAWideMarkIsFullyCovered:
    """Reproducción del bug: una máscara 1500x300 en 1920x1080 producía un crop
    cuadrado clampado a 1080 que dejaba la marca afuera — el modelo pintaba solo
    x∈[410,1489], los píxeles marcados en x=210/400/1500/1690 quedaban
    originales y había un salto 60→255 en 1px en x=410."""

    def _edited(self, model: RecordingModel | None = None) -> np.ndarray:
        photo = base_photo((1920, 1080))
        mask = mask_with_box((1920, 1080), (200, 400, 1700, 700))
        return np.asarray(run_masked_edit(photo, mask, model or RecordingModel(), SETTINGS))

    def test_every_marked_pixel_gets_edited(self) -> None:
        after = self._edited()

        for x in (210, 400, 410, 1500, 1690):
            pixel = after[550, x]
            assert pixel[0] > 200 and pixel[1] < 60, f"pixel marcado en x={x} quedo sin editar"
        interior = after[450:650, 250:1650]
        assert (interior[:, :, 0] > 200).all(), "todo el interior marcado debe quedar editado"

    def test_there_is_no_hard_seam_inside_the_mark(self) -> None:
        after = self._edited()

        row = after[550, 250:1650, 0].astype(int)
        assert np.abs(np.diff(row)).max() < 120, "salto duro = el crop no cubrio la marca"

    def test_model_dims_are_buckets_of_128_on_both_axes(self) -> None:
        model = RecordingModel()

        self._edited(model)

        call = model.calls[0]
        assert call["width"] % 128 == 0 and call["height"] % 128 == 0
        assert call["image_size"] == (call["width"], call["height"])
        assert call["mask_size"] == (call["width"], call["height"])

    def test_a_portrait_mark_taller_than_the_width_is_covered_too(self) -> None:
        photo = base_photo((1080, 1920))
        mask = mask_with_box((1080, 1920), (400, 200, 700, 1700))

        after = np.asarray(run_masked_edit(photo, mask, RecordingModel(), SETTINGS))

        for y in (210, 400, 1500, 1690):
            pixel = after[y, 550]
            assert pixel[0] > 200 and pixel[1] < 60, f"pixel marcado en y={y} quedo sin editar"
        interior = after[250:1650, 450:650]
        assert (interior[:, :, 0] > 200).all()


def test_native_floor_on_a_rectangular_crop_keeps_its_aspect() -> None:
    # Imagen franja: el crop se clampa en alto y queda rectangular, y el piso
    # nativo tiene que escalar uniforme en vez de cuadrarlo por la fuerza.
    photo = base_photo((800, 256))
    mask = mask_with_box((800, 256), (100, 100, 700, 156))
    model = RecordingModel()

    run_masked_edit(
        photo, mask, model, MaskedEditSettings(8, 8, None, 1024, native_side=1024)
    )

    call = model.calls[0]
    assert call["width"] >= 1024, "el piso aplica sobre el lado mayor"
    assert call["height"] < call["width"], "el aspecto del crop no se cuadra"
    assert call["width"] % 128 == 0 and call["height"] % 128 == 0


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
