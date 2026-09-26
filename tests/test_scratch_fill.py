from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.engines.scratch_fill import (
    CLASSIC_ENGINE,
    FAST_ENGINE,
    MIGAN_STRIDE,
    MIGAN_WINDOW,
    FillRequest,
    fill_damage,
    fill_touches_faces,
    hole_widths,
    large_hole_mask,
    migan_infer,
    migan_window_infer,
    window_origins,
)
from app.services.engines.tiled_restore_runner import RestoreCancelled
from app.services.inpaint_mask import feather_mask
from app.services.restore_models import MIGAN_MODEL_ID, VENDORED_MODELS

sys.path.insert(0, str(Path(__file__).parent))
from seam_detector import measure_seams  # noqa: E402

SEAM_THRESHOLD = 2.0


class PaintingMigan:
    """MI-GAN falso: rellena el hueco con la foto + un desvio propio de cada ventana."""

    def __init__(self, step: int = 12) -> None:
        self.windows: list[np.ndarray] = []
        self.step = step

    def __call__(self, window: np.ndarray) -> np.ndarray:
        self.windows.append(window.copy())
        rgb, hole = window[:, :, :3].astype(np.int16), window[:, :, 3:] > 127
        offset = self.step * len(self.windows)
        return np.where(hole, np.clip(rgb + offset, 0, 255), rgb).astype(np.uint8)


class OrtLikeMigan:
    def __init__(self) -> None:
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, output_names, feeds):
        self.feeds.append(feeds)
        return [np.full_like(feeds["image"], 7)]


def smooth_photo(height: int, width: int, seed: int = 7) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    red = 90 + 60 * np.sin(xx / 97) + 30 * np.cos(yy / 61)
    green = 110 + 50 * np.cos((xx + yy) / 83)
    blue = 80 + 40 * np.sin(xx / 51) * np.cos(yy / 71)
    noise = np.random.default_rng(seed).normal(0, 3, (height, width, 3))
    return (np.clip(np.stack([red, green, blue], axis=-1) + noise, 0, 255) / 255.0).astype(np.float32)


def empty_mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=bool)


def scratch_mask(height: int, width: int, column: int, thickness: int = 2) -> np.ndarray:
    mask = empty_mask(height, width)
    mask[:, column : column + thickness] = True
    return mask


def to_u8(image: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------- ventanas de 512 con paso 384


def test_windows_step_384_and_the_last_one_is_aligned_to_the_edge() -> None:
    assert MIGAN_WINDOW == 512 and MIGAN_STRIDE == 384
    origins = window_origins(1000, 1280)

    assert sorted({y for y, _ in origins}) == [0, 384, 488]
    assert sorted({x for _, x in origins}) == [0, 384, 768]


def test_a_photo_smaller_than_a_window_is_one_window_at_its_own_size() -> None:
    photo = smooth_photo(300, 200)
    migan = PaintingMigan()

    fill_damage(photo, scratch_mask(300, 200, 50), FillRequest(), migan=migan)

    (window,) = migan.windows
    assert window.shape == (300, 200, 4)


def test_only_the_windows_that_contain_mask_are_run_at_native_resolution() -> None:
    photo = smooth_photo(1280, 1280)
    mask = empty_mask(1280, 1280)
    mask[100:110, 100:110] = True
    migan = PaintingMigan()

    result = fill_damage(photo, mask, FillRequest(), migan=migan)

    (window,) = migan.windows
    assert window.shape == (512, 512, 4)
    np.testing.assert_array_equal(window[:, :, :3], to_u8(photo[:512, :512]))
    assert result.windows == 1


def test_a_hole_in_the_overlap_is_filled_by_every_window_that_sees_it() -> None:
    photo = smooth_photo(512, 1280)
    mask = empty_mask(512, 1280)
    mask[200:210, 450:460] = True
    migan = PaintingMigan()

    result = fill_damage(photo, mask, FillRequest(), migan=migan)

    assert result.windows == 2
    assert len(migan.windows) == 2


def test_an_empty_mask_never_calls_mi_gan_and_returns_the_photo_untouched() -> None:
    photo = smooth_photo(200, 300)
    migan = PaintingMigan()

    result = fill_damage(photo, empty_mask(200, 300), FillRequest(), migan=migan)

    assert migan.windows == []
    assert result.windows == 0
    np.testing.assert_array_equal(result.image, photo)


# ---------------------------------------------------------------- polaridad y contrato del grafo


def test_mi_gan_gets_uint8_nchw_and_the_inverted_mask_255_means_keep() -> None:
    session = OrtLikeMigan()
    infer = migan_window_infer(session)
    window = np.zeros((4, 6, 4), dtype=np.uint8)
    window[:, :, :3] = 100
    window[1, 2, 3] = 255

    output = infer(window)

    (feeds,) = session.feeds
    assert feeds["image"].dtype == np.uint8 and feeds["image"].shape == (1, 3, 4, 6)
    assert feeds["mask"].dtype == np.uint8 and feeds["mask"].shape == (1, 1, 4, 6)
    assert feeds["mask"][0, 0, 1, 2] == 0
    assert int(feeds["mask"].sum()) == 255 * 23
    assert output.shape == (4, 6, 3) and output.dtype == np.uint8


def test_mi_gan_sees_a_hard_mask_even_when_the_feather_is_soft() -> None:
    photo = smooth_photo(64, 64)
    migan = PaintingMigan()

    fill_damage(photo, scratch_mask(64, 64, 30), FillRequest(), migan=migan)

    (window,) = migan.windows
    assert set(np.unique(window[:, :, 3]).tolist()) == {0, 255}
    assert window[:, 30:32, 3].min() == 255


# ---------------------------------------------------------------- solo cambian los pixeles enmascarados


def test_mi_gan_only_changes_the_masked_pixels_and_their_two_pixel_feather() -> None:
    photo = smooth_photo(700, 900)
    mask = scratch_mask(700, 900, 420, thickness=3)

    result = fill_damage(photo, mask, FillRequest(), migan=PaintingMigan(step=40))

    changed = np.any(result.image != photo, axis=2)
    feather_support = feather_mask(np.where(mask, np.uint8(255), np.uint8(0)), 2) > 0
    assert changed[:, 420:423].all()
    assert not (changed & ~feather_support).any()
    np.testing.assert_array_equal(result.image[:, :400], photo[:, :400])
    np.testing.assert_array_equal(result.image[:, 440:], photo[:, 440:])


def test_mi_gan_output_bleeding_outside_the_hole_is_not_pasted() -> None:
    photo = smooth_photo(128, 128)
    mask = empty_mask(128, 128)
    mask[60:64, 60:64] = True

    def bleeding(window: np.ndarray) -> np.ndarray:
        return np.full(window.shape[:2] + (3,), 255, dtype=np.uint8)

    result = fill_damage(photo, mask, FillRequest(), migan=bleeding)

    np.testing.assert_allclose(result.image[60:64, 60:64], 1.0)
    np.testing.assert_array_equal(result.image[:50], photo[:50])


def test_windows_painting_different_constants_leave_no_seam() -> None:
    photo = smooth_photo(1280, 1280)
    mask = np.ones((1280, 1280), dtype=bool)
    migan = PaintingMigan(step=12)

    result = fill_damage(photo, mask, FillRequest(), migan=migan)

    assert result.windows == 9
    for period in (MIGAN_STRIDE, MIGAN_WINDOW):
        report = measure_seams(to_u8(result.image), period)
        assert report.worst_ratio < SEAM_THRESHOLD, report


def test_gray_photos_are_replicated_for_mi_gan_and_averaged_back() -> None:
    photo = smooth_photo(96, 96)[:, :, 1]
    mask = scratch_mask(96, 96, 40)

    result = fill_damage(photo, mask, FillRequest(), migan=PaintingMigan(step=30))

    assert result.image.shape == (96, 96)
    assert result.image.dtype == np.float32
    np.testing.assert_array_equal(result.image[:, :30], photo[:, :30])
    assert np.all(result.image[:, 40:42] > photo[:, 40:42])


def test_cancelling_stops_before_the_next_window() -> None:
    photo = smooth_photo(1280, 1280)
    cancel = threading.Event()
    migan = PaintingMigan()

    def cancel_after_first(done: int, total: int) -> None:
        cancel.set()

    with pytest.raises(RestoreCancelled):
        fill_damage(
            photo,
            np.ones((1280, 1280), dtype=bool),
            FillRequest(),
            migan=migan,
            cancel_event=cancel,
            on_progress=cancel_after_first,
        )
    assert len(migan.windows) == 1


def test_progress_counts_only_the_windows_with_mask() -> None:
    photo = smooth_photo(1280, 1280)
    mask = empty_mask(1280, 1280)
    mask[100:110, 100:110] = True
    mask[1200:1210, 1200:1210] = True
    calls: list[tuple[int, int]] = []

    fill_damage(photo, mask, FillRequest(), migan=PaintingMigan(), on_progress=lambda d, t: calls.append((d, t)))

    assert calls == [(1, 2), (2, 2)]


# ---------------------------------------------------------------- "Classic": Telea


def test_classic_fills_a_thin_scratch_from_its_surroundings_and_touches_nothing_else() -> None:
    photo = np.full((80, 80, 3), 0.5, dtype=np.float32)
    photo[:, 40:42] = 1.0
    mask = scratch_mask(80, 80, 40)

    result = fill_damage(photo, mask, FillRequest(engine=CLASSIC_ENGINE))

    np.testing.assert_allclose(result.image[:, 40:42], 0.5, atol=1.0 / 255.0)
    np.testing.assert_array_equal(result.image[:, :40], photo[:, :40])
    np.testing.assert_array_equal(result.image[:, 42:], photo[:, 42:])
    assert result.engine == CLASSIC_ENGINE and result.windows == 0


def test_classic_in_16_bits_runs_per_channel_without_8_bit_quantization() -> None:
    gradient = np.linspace(0.2, 0.8, 120, dtype=np.float32)[None, :].repeat(60, axis=0)
    photo = np.stack([gradient, gradient * 0.9, gradient * 0.8], axis=2)
    photo[30:32] = 0.0
    mask = empty_mask(60, 120)
    mask[30:32] = True

    result = fill_damage(photo, mask, FillRequest(engine=CLASSIC_ENGINE, bit_depth=16))

    filled = result.image[30:32]
    on_8_bit_grid = np.isclose(filled * 255.0, np.rint(filled * 255.0), atol=1e-3)
    assert not on_8_bit_grid.all()
    np.testing.assert_allclose(filled, photo[29:30].repeat(2, axis=0), atol=0.01)
    np.testing.assert_array_equal(result.image[:30], photo[:30])


def test_classic_works_on_gray_photos() -> None:
    photo = np.full((40, 40), 0.3, dtype=np.float32)
    photo[:, 20] = 0.9

    result = fill_damage(photo, scratch_mask(40, 40, 20, thickness=1), FillRequest(engine=CLASSIC_ENGINE))

    assert result.image.shape == (40, 40)
    # Telea redondea a 8 bits por su cuenta: unos pocos niveles de ruido alrededor del gris.
    np.testing.assert_allclose(result.image[:, 20], 0.3, atol=2.0 / 255.0)


def test_fast_falls_back_to_classic_when_the_migan_pack_is_missing() -> None:
    photo = np.full((40, 40, 3), 0.5, dtype=np.float32)

    result = fill_damage(photo, scratch_mask(40, 40, 10), FillRequest(engine=FAST_ENGINE), migan=None)

    assert result.engine == CLASSIC_ENGINE


@pytest.mark.parametrize(
    "request_",
    [FillRequest(engine="lama"), FillRequest(bit_depth=12)],
)
def test_unknown_engines_and_bit_depths_are_rejected(request_: FillRequest) -> None:
    with pytest.raises(ValueError):
        fill_damage(np.zeros((8, 8, 3), np.float32), empty_mask(8, 8), request_)


def test_a_mask_of_another_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        fill_damage(np.zeros((8, 8, 3), np.float32), empty_mask(8, 9), FillRequest(engine=CLASSIC_ENGINE))


# ---------------------------------------------------------------- areas grandes


def test_hole_width_is_twice_the_largest_distance_to_the_edge() -> None:
    mask = empty_mask(100, 100)
    mask[10:90, 20:22] = True
    mask[40:70, 50:80] = True

    widths = hole_widths(mask)

    assert sorted(np.round(widths).tolist()) == [2.0, 30.0]


def test_only_holes_wider_than_24_px_are_large() -> None:
    mask = empty_mask(200, 200)
    mask[10:30, 10:30] = True
    mask[100:130, 100:130] = True
    mask[:, 180:183] = True

    large = large_hole_mask(mask)

    assert large[100:130, 100:130].all()
    assert not large[10:30, 10:30].any()
    assert not large[:, 180:183].any()


def test_large_holes_are_counted_and_filled_by_default() -> None:
    photo = np.full((200, 200, 3), 0.5, dtype=np.float32)
    photo[100:130, 100:130] = 0.0
    mask = empty_mask(200, 200)
    mask[100:130, 100:130] = True

    result = fill_damage(photo, mask, FillRequest(engine=CLASSIC_ENGINE))

    assert result.large_holes == 1
    assert not result.large_holes_left_unfilled
    assert result.image[115, 115, 0] > 0.3


def test_leave_large_holes_unfilled_takes_them_out_of_the_mask() -> None:
    photo = np.full((200, 200, 3), 0.5, dtype=np.float32)
    photo[100:130, 100:130] = 0.0
    photo[:, 180:182] = 1.0
    mask = empty_mask(200, 200)
    mask[100:130, 100:130] = True
    mask[:, 180:182] = True

    result = fill_damage(photo, mask, FillRequest(engine=CLASSIC_ENGINE, leave_large_holes=True))

    assert result.large_holes == 1
    assert result.large_holes_left_unfilled
    np.testing.assert_array_equal(result.image[100:130, 100:130], photo[100:130, 100:130])
    np.testing.assert_allclose(result.image[:, 180:182], 0.5, atol=1.0 / 255.0)
    assert result.coverage == pytest.approx(400 / 40000)


# ---------------------------------------------------------------- relleno sobre caras


def test_any_fill_inside_a_face_box_touches_faces() -> None:
    mask = empty_mask(100, 100)
    mask[50, 50] = True

    assert fill_touches_faces(mask, [(40.0, 40.0, 60.0, 60.0)])
    assert not fill_touches_faces(mask, [(0.0, 0.0, 30.0, 30.0)])
    assert not fill_touches_faces(mask, [])


def test_a_fractional_face_box_counts_the_pixels_it_partly_covers() -> None:
    mask = empty_mask(100, 100)
    mask[10, 29] = True

    assert fill_touches_faces(mask, [(29.6, 9.2, 40.0, 20.0)])


def test_a_face_box_outside_the_photo_is_clipped() -> None:
    mask = empty_mask(50, 50)
    mask[49, 49] = True

    assert fill_touches_faces(mask, [(45.0, 45.0, 90.0, 90.0)])
    assert not fill_touches_faces(mask, [(60.0, 60.0, 90.0, 90.0)])


def test_the_result_reports_touches_faces_on_the_mask_actually_filled() -> None:
    photo = np.full((200, 200, 3), 0.5, dtype=np.float32)
    mask = empty_mask(200, 200)
    mask[100:130, 100:130] = True
    face = (95.0, 95.0, 140.0, 140.0)

    filled = fill_damage(photo, mask, FillRequest(engine=CLASSIC_ENGINE, face_boxes=(face,)))
    left = fill_damage(
        photo, mask, FillRequest(engine=CLASSIC_ENGINE, face_boxes=(face,), leave_large_holes=True)
    )

    assert filled.touches_faces
    assert not left.touches_faces


def test_metadata_uses_the_sidecar_keys() -> None:
    photo = np.full((200, 200, 3), 0.5, dtype=np.float32)
    mask = empty_mask(200, 200)
    mask[100:130, 100:130] = True

    result = fill_damage(photo, mask, FillRequest(engine=CLASSIC_ENGINE, leave_large_holes=True))

    assert result.to_metadata() == {
        "engine": "classic",
        "windows": 0,
        "finalCoverage": 0.0,
        "largeHoles": 1,
        "largeHolesLeftUnfilled": True,
        "touchesFaces": False,
    }


# ---------------------------------------------------------------- sesion del dueno unico


class CountingCoordinator:
    def __init__(self) -> None:
        self.acquired: list[str] = []

    def register(self, owner: object) -> None:
        pass

    def acquire(self, device: str, owner: object) -> None:
        self.acquired.append(device)

    def invalidate_device(self, device: str) -> None:
        pass


def write_fake_migan(path: Path) -> None:
    from onnx import TensorProto, helper, save

    # result = image donde mask = 255 y 200 en el hueco: mismo contrato uint8 que migan_pipeline_v2.
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["image"], ["image_f"], to=TensorProto.FLOAT),
            helper.make_node("Cast", ["mask"], ["mask_f"], to=TensorProto.FLOAT),
            helper.make_node("Div", ["mask_f", "peak"], ["keep"]),
            helper.make_node("Sub", ["image_f", "fill"], ["delta"]),
            helper.make_node("Mul", ["delta", "keep"], ["kept"]),
            helper.make_node("Add", ["kept", "fill"], ["blended"]),
            helper.make_node("Cast", ["blended"], ["result"], to=TensorProto.UINT8),
        ],
        "fake_migan",
        [
            helper.make_tensor_value_info("image", TensorProto.UINT8, [1, 3, "h", "w"]),
            helper.make_tensor_value_info("mask", TensorProto.UINT8, [1, 1, "h", "w"]),
        ],
        [helper.make_tensor_value_info("result", TensorProto.UINT8, [1, 3, "h", "w"])],
        initializer=[
            helper.make_tensor("peak", TensorProto.FLOAT, [], [255.0]),
            helper.make_tensor("fill", TensorProto.FLOAT, [], [200.0]),
        ],
    )
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(path))


def migan_settings(tmp_path: Path, model: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), MIGAN_MODEL=str(model))


def test_mi_gan_is_a_vendored_model_of_the_single_owner() -> None:
    vendored = VENDORED_MODELS[MIGAN_MODEL_ID]

    assert vendored.pack == "migan"
    assert vendored.spec.filename == "migan_pipeline_v2.onnx"
    assert vendored.spec.license_spdx == "MIT"
    assert vendored.spec.fp16_filename is None


def test_the_engine_opens_mi_gan_from_the_vendor_path_and_fills_on_cpu(tmp_path: Path) -> None:
    model = tmp_path / "vendor" / "migan_pipeline_v2.onnx"
    model.parent.mkdir()
    write_fake_migan(model)
    coordinator = CountingCoordinator()
    engine = PhotoRestoreEngine(migan_settings(tmp_path, model), coordinator, models={})
    photo = np.full((96, 128, 3), 0.25, dtype=np.float32)
    mask = scratch_mask(96, 128, 60)

    result = fill_damage(photo, mask, FillRequest(), migan=migan_infer(engine, "cpu"))

    assert coordinator.acquired == ["cpu"]
    assert [key.model_id for key in engine.live_sessions("cpu")] == [MIGAN_MODEL_ID]
    assert result.engine == FAST_ENGINE
    np.testing.assert_allclose(result.image[:, 60:62], 200.0 / 255.0, atol=1e-6)
    np.testing.assert_array_equal(result.image[:, :50], photo[:, :50])


def test_a_missing_migan_file_names_its_pack(tmp_path: Path) -> None:
    engine = PhotoRestoreEngine(migan_settings(tmp_path, tmp_path / "nope.onnx"), CountingCoordinator(), models={})
    infer = migan_infer(engine, "cpu")

    with pytest.raises(RuntimeError, match="borrado rapido"):
        infer(np.zeros((8, 8, 4), dtype=np.uint8))
