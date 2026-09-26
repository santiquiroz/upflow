from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.engines.scratch_detect import (
    DEFAULT_SENSITIVITY,
    SCRATCH_MODEL_ID,
    damage_mask,
    detector_input,
    detector_input_size,
    grow_mask,
    probability_to_u8,
    scratch_detector,
    scratch_probability,
    sensitivity_threshold,
)
from app.services.restore_models import RestoreModelSpec


class RecordingInfer:
    def __init__(self, logits_for=None) -> None:
        self.inputs: list[np.ndarray] = []
        self.logits_for = logits_for or (lambda gray: np.zeros_like(gray))

    def __call__(self, tile: np.ndarray) -> np.ndarray:
        self.inputs.append(tile.copy())
        return self.logits_for(tile)


def logits_from_intensity(gray: np.ndarray) -> np.ndarray:
    # FakeBopbtl: lo claro es raya (logit +8) y lo oscuro no (logit -8).
    return np.where(gray > 0.5, 8.0, -8.0).astype(np.float32)


def photo_with_bright_line(height: int = 600, width: int = 800, column: int = 400) -> np.ndarray:
    photo = np.full((height, width, 3), 0.2, dtype=np.float32)
    photo[:, column - 6 : column + 6] = 0.95
    return photo


# ---------------------------------------------------------------- entrada del detector a 256


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (4000, 3000, (336, 256)),
        (3000, 4000, (256, 336)),
        (256, 256, (256, 256)),
        (1000, 250, (1024, 256)),
        (100, 80, (320, 256)),
    ],
)
def test_detector_size_puts_the_short_side_at_256_and_rounds_to_16(width, height, expected) -> None:
    assert detector_input_size(width, height) == expected


def test_detector_sees_the_luma_with_the_short_side_at_256_whatever_the_native_size() -> None:
    photo = np.empty((1500, 2000, 3), dtype=np.float32)
    photo[...] = np.array([0.8, 0.4, 0.2], dtype=np.float32)
    infer = RecordingInfer()

    scratch_probability(photo, infer)

    (seen,) = infer.inputs
    assert seen.shape == (256, 336, 1)
    expected_luma = 0.299 * 0.8 + 0.587 * 0.4 + 0.114 * 0.2
    np.testing.assert_allclose(seen, expected_luma, atol=1e-5)


def test_detector_input_stays_within_zero_and_one_after_the_resize() -> None:
    rng = np.random.default_rng(3)
    photo = (rng.random((700, 500, 3)) > 0.5).astype(np.float32)

    gray = detector_input(photo)

    assert gray.dtype == np.float32
    assert gray.min() >= 0.0 and gray.max() <= 1.0


def test_gray_photos_feed_the_detector_directly() -> None:
    gray_hw = np.full((300, 400), 0.3, dtype=np.float32)
    gray_hw1 = gray_hw[:, :, np.newaxis]

    np.testing.assert_allclose(detector_input(gray_hw), 0.3, atol=1e-5)
    np.testing.assert_allclose(detector_input(gray_hw1), 0.3, atol=1e-5)


# ---------------------------------------------------------------- probabilidad bilineal a tamano nativo


def test_probability_is_the_sigmoid_of_the_logits_at_native_size() -> None:
    photo = np.full((500, 700, 3), 0.4, dtype=np.float32)
    infer = RecordingInfer(lambda gray: np.full_like(gray, 2.0))

    probability = scratch_probability(photo, infer)

    assert probability.shape == (500, 700)
    assert probability.dtype == np.float32
    np.testing.assert_allclose(probability, 1.0 / (1.0 + np.exp(-2.0)), atol=1e-6)


def test_probability_is_rescaled_bilinearly_not_nearest() -> None:
    probability = scratch_probability(photo_with_bright_line(), RecordingInfer(logits_from_intensity))

    row = probability[300]
    intermediate = (row > 0.05) & (row < 0.95)
    assert intermediate.any(), "a nearest resize would only give 0 or 1"
    assert row[400] > 0.9
    assert row[100] < 0.01


def test_huge_logits_do_not_overflow_into_nan() -> None:
    photo = photo_with_bright_line()
    infer = RecordingInfer(lambda gray: np.where(gray > 0.5, 1e4, -1e4).astype(np.float32))

    with np.errstate(over="raise"):
        probability = scratch_probability(photo, infer)

    assert np.isfinite(probability).all()
    assert probability.min() >= 0.0 and probability.max() <= 1.0


# ---------------------------------------------------------------- sensibilidad sin re-inferir


@pytest.mark.parametrize(("sensitivity", "threshold"), [(0.0, 0.6), (0.5, 0.4), (0.75, 0.3), (1.0, 0.2)])
def test_sensitivity_moves_the_threshold_between_0_6_and_0_2(sensitivity, threshold) -> None:
    assert sensitivity_threshold(sensitivity) == pytest.approx(threshold)


def test_default_sensitivity_is_the_0_4_threshold_of_bopbtl() -> None:
    assert sensitivity_threshold(DEFAULT_SENSITIVITY) == pytest.approx(0.4)


@pytest.mark.parametrize("sensitivity", [-0.1, 1.1, float("nan")])
def test_sensitivity_outside_zero_to_one_is_rejected(sensitivity) -> None:
    with pytest.raises(ValueError):
        sensitivity_threshold(sensitivity)


def test_changing_the_sensitivity_reuses_the_probability_without_running_the_model_again() -> None:
    gradient = np.repeat(np.linspace(0.0, 1.0, 800, dtype=np.float32)[np.newaxis, :, np.newaxis], 600, axis=0)
    infer = RecordingInfer(lambda gray: (gray - 0.5) * 10.0)
    probability = scratch_probability(np.repeat(gradient, 3, axis=2), infer)

    masks = [damage_mask(probability, sensitivity) for sensitivity in (0.0, 0.5, 1.0)]

    assert len(infer.inputs) == 1
    assert masks[0].sum() < masks[1].sum() < masks[2].sum()
    assert np.all(masks[2][masks[0]]), "a higher sensitivity keeps every pixel of a lower one"


def test_mask_marks_exactly_the_pixels_above_the_threshold() -> None:
    probability = np.array([[0.1, 0.39, 0.41, 0.9]], dtype=np.float32)

    mask = damage_mask(probability)

    assert mask.dtype == np.bool_
    assert mask.tolist() == [[False, False, True, True]]


# ---------------------------------------------------------------- agrandar / achicar


def test_growing_by_one_pixel_turns_a_dot_into_a_3x3_square() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True

    grown = grow_mask(mask, 1)

    assert grown.sum() == 9
    assert grown[3:6, 3:6].all()


def test_growing_by_three_pixels_reaches_three_pixels_out() -> None:
    mask = np.zeros((15, 15), dtype=bool)
    mask[7, 7] = True

    grown = grow_mask(mask, 3)

    assert grown[7, 4] and grown[7, 10]
    assert not grown[7, 3] and not grown[7, 11]


def test_shrinking_removes_thin_strokes_and_keeps_the_core_of_wide_ones() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[:, 2] = True
    mask[5:15, 8:18] = True

    shrunk = grow_mask(mask, -1)

    assert not shrunk[:, 2].any()
    assert shrunk[6:14, 9:17].all()
    assert not shrunk[5, 8]


def test_zero_grow_returns_the_same_mask() -> None:
    mask = np.eye(6, dtype=bool)

    assert np.array_equal(grow_mask(mask, 0), mask)


@pytest.mark.parametrize("pixels", [-4, 4])
def test_grow_outside_three_pixels_is_rejected(pixels) -> None:
    with pytest.raises(ValueError):
        grow_mask(np.zeros((4, 4), dtype=bool), pixels)


def test_damage_mask_applies_the_grow_after_thresholding() -> None:
    probability = np.zeros((9, 9), dtype=np.float32)
    probability[4, 4] = 0.9

    mask = damage_mask(probability, DEFAULT_SENSITIVITY, grow_px=1)

    assert mask.sum() == 9


def test_probability_png_is_eight_bit_and_rounded() -> None:
    probability = np.array([[0.0, 0.5, 1.0, 0.002]], dtype=np.float32)

    png = probability_to_u8(probability)

    assert png.dtype == np.uint8
    assert png.tolist() == [[0, 128, 255, 1]]


# ---------------------------------------------------------------- detector real en el CPU EP via el dueno unico


def write_logit_graph(path: Path) -> None:
    from onnx import TensorProto, helper, save

    # logits = (gray - 0.5) * 20: sin clamp en el host, lo oscuro da probabilidad ~0.
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["input", "half"], ["centered"]),
            helper.make_node("Mul", ["centered", "gain"], ["output"]),
        ],
        "fake_bopbtl",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, "h", "w"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, "h", "w"])],
        initializer=[
            helper.make_tensor("half", TensorProto.FLOAT, [], [0.5]),
            helper.make_tensor("gain", TensorProto.FLOAT, [], [20.0]),
        ],
    )
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(path))


def detector_spec() -> RestoreModelSpec:
    return RestoreModelSpec(
        id=SCRATCH_MODEL_ID,
        name="BOPBTL scratch detector",
        bundle="core",
        filename="fake-bopbtl.onnx",
        license_spdx="MIT",
        license_url="https://example.com/LICENSE",
        copyright="Copyright (c) test",
        attribution="Test model",
        data_lineage="D1a + D1c",
        commercial_use="yes",
        source_url="https://example.com/model",
        source_revision="abc123",
        source_sha256="a" * 64,
        modifications=("exported to ONNX",),
        tile_min=256,
        channels=1,
    )


class CountingCoordinator:
    def __init__(self) -> None:
        self.acquired: list[str] = []

    def register(self, owner: object) -> None:
        pass

    def acquire(self, device: str, owner: object) -> None:
        self.acquired.append(device)

    def invalidate_device(self, device: str) -> None:
        pass


def test_engine_detector_runs_the_real_graph_on_cpu_without_clamping_the_logits(tmp_path: Path) -> None:
    model_dir = tmp_path / "restore"
    model_dir.mkdir()
    write_logit_graph(model_dir / "fake-bopbtl.onnx")
    settings = Settings(_env_file=None, RESTORE_MODEL_DIR=str(model_dir), RUNTIME_DIR=str(tmp_path / "runtime"))
    coordinator = CountingCoordinator()
    engine = PhotoRestoreEngine(
        settings,
        coordinator,
        models={SCRATCH_MODEL_ID: detector_spec()},
    )

    probability = scratch_detector(engine)(photo_with_bright_line())

    assert coordinator.acquired == ["cpu"]
    assert [key.device for key in engine.live_sessions("cpu")] == ["cpu"]
    assert probability.shape == (600, 800)
    assert probability[300, 400] > 0.99
    assert probability[300, 100] < 0.01
