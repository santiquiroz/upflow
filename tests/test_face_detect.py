from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services.engines.face_detect import (
    BGR_MEAN,
    DETECT_MAX_SIDE,
    NMS_IOU,
    PRE_NMS_TOP_K,
    RETINAFACE_MODEL_ID,
    SCORE_THRESHOLD,
    FaceDetection,
    decode_boxes,
    decode_detections,
    decode_landmarks,
    detect_faces,
    face_detector,
    landmarked_face_detector,
    network_input,
    nms,
    priors,
    session_run,
)
from app.services.engines.photo_restore_engine import NonFiniteOutputError, PhotoRestoreEngine
from app.services.photo_diagnosis import DetectedFace
from app.services.restore_models import RestoreModelSpec

VARIANCES = (0.1, 0.2)


def reference_priors(height: int, width: int) -> np.ndarray:
    # El bucle de PriorBox.generate_anchors de yakhyo (cfg_re34), escrito tal cual como referencia.
    anchors = []
    for step, min_sizes in zip((8, 16, 32), ((16, 32), (64, 128), (256, 512)), strict=True):
        for i, j in itertools.product(range(math.ceil(height / step)), range(math.ceil(width / step))):
            for min_size in min_sizes:
                anchors += [(j + 0.5) * step / width, (i + 0.5) * step / height, min_size / width, min_size / height]
    return np.array(anchors, dtype=np.float32).reshape(-1, 4)


def encode_box(box: np.ndarray, prior: np.ndarray) -> np.ndarray:
    center = (box[:2] + box[2:]) / 2
    size = box[2:] - box[:2]
    return np.concatenate(
        [(center - prior[:2]) / (VARIANCES[0] * prior[2:]), np.log(size / prior[2:]) / VARIANCES[1]]
    )


def encode_points(points: np.ndarray, prior: np.ndarray) -> np.ndarray:
    return ((points - prior[:2]) / (VARIANCES[0] * prior[2:])).reshape(-1)


def face_points(box: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    relative = np.array([[0.3, 0.4], [0.7, 0.4], [0.5, 0.55], [0.35, 0.75], [0.65, 0.75]])
    return np.array([x0, y0]) + relative * np.array([w, h])


def synthetic_outputs(height: int, width: int, faces: list[tuple[int, np.ndarray, float]]):
    anchors = priors(height, width)
    count = len(anchors)
    loc = np.zeros((1, count, 4), np.float32)
    conf = np.tile(np.array([0.99, 0.01], np.float32), (1, count, 1))
    landmarks = np.zeros((1, count, 10), np.float32)
    size = np.array([width, height], np.float32)
    for index, pixel_box, score in faces:
        normalized = pixel_box / np.tile(size, 2)
        loc[0, index] = encode_box(normalized, anchors[index])
        landmarks[0, index] = encode_points(face_points(pixel_box) / size, anchors[index])
        conf[0, index] = (1.0 - score, score)
    return loc, conf, landmarks


def prior_index_near(height: int, width: int, x: float, y: float, min_size: int) -> int:
    anchors = priors(height, width)
    target = np.array([x / width, y / height, min_size / width, min_size / height])
    return int(np.argmin(np.abs(anchors - target).sum(axis=1)))


# ---------------------------------------------------------------- priors


@pytest.mark.parametrize(("height", "width"), [(480, 640), (481, 353), (64, 64)])
def test_priors_match_the_yakhyo_prior_box_loop(height: int, width: int) -> None:
    assert np.allclose(priors(height, width), reference_priors(height, width), atol=1e-7)


def test_priors_count_two_sizes_per_cell_on_three_levels() -> None:
    expected = 2 * sum(math.ceil(480 / s) * math.ceil(640 / s) for s in (8, 16, 32))

    assert priors(480, 640).shape == (expected, 4)
    assert priors(480, 640).dtype == np.float32


def test_first_prior_is_the_16px_anchor_of_the_top_left_cell() -> None:
    assert priors(480, 640)[0].tolist() == pytest.approx([4 / 640, 4 / 480, 16 / 640, 16 / 480])
    assert priors(480, 640)[1].tolist() == pytest.approx([4 / 640, 4 / 480, 32 / 640, 32 / 480])


# ---------------------------------------------------------------- decodificacion


def test_zero_offsets_decode_to_the_prior_box() -> None:
    anchors = np.array([[0.5, 0.25, 0.2, 0.1]], np.float32)

    boxes = decode_boxes(np.zeros((1, 4), np.float32), anchors)

    assert boxes[0].tolist() == pytest.approx([0.4, 0.2, 0.6, 0.3])


def test_decode_boxes_inverts_the_training_encoding() -> None:
    anchors = np.array([[0.5, 0.5, 0.1, 0.1], [0.2, 0.7, 0.05, 0.08]], np.float32)
    truth = np.array([[0.42, 0.47, 0.6, 0.66], [0.17, 0.66, 0.24, 0.75]], np.float32)
    loc = np.stack([encode_box(box, prior) for box, prior in zip(truth, anchors, strict=True)])

    assert np.allclose(decode_boxes(loc, anchors), truth, atol=1e-6)


def test_decode_landmarks_inverts_the_training_encoding() -> None:
    anchors = np.array([[0.5, 0.5, 0.1, 0.1]], np.float32)
    points = np.array([[0.46, 0.48], [0.54, 0.48], [0.5, 0.52], [0.47, 0.56], [0.53, 0.56]], np.float32)

    decoded = decode_landmarks(encode_points(points, anchors[0])[None], anchors)

    assert decoded.shape == (1, 5, 2)
    assert np.allclose(decoded[0], points, atol=1e-6)


# ---------------------------------------------------------------- NMS y umbral


def test_nms_drops_boxes_that_overlap_a_better_one_above_the_threshold() -> None:
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [200, 200, 260, 260]], np.float32)
    scores = np.array([0.95, 0.97, 0.92], np.float32)

    assert nms(boxes, scores) == [1, 2]


def test_nms_keeps_boxes_that_overlap_at_most_the_threshold() -> None:
    # Dos cajas de 100x100 (areas con +1 de biubug6) que se solapan con IoU apenas por debajo de 0,4.
    boxes = np.array([[0, 0, 99, 99], [43, 0, 142, 99]], np.float32)
    scores = np.array([0.95, 0.94], np.float32)
    inter = (99 - 43 + 1) * 100
    assert inter / (2 * 100 * 100 - inter) < NMS_IOU

    assert nms(boxes, scores) == [0, 1]


def test_nms_of_nothing_is_empty() -> None:
    assert nms(np.zeros((0, 4), np.float32), np.zeros(0, np.float32)) == []


def test_decode_detections_returns_pixel_boxes_and_landmarks() -> None:
    height, width = 480, 640
    box = np.array([300.0, 150.0, 364.0, 230.0], np.float32)
    index = prior_index_near(height, width, 332, 190, 64)

    (face,) = decode_detections(synthetic_outputs(height, width, [(index, box, 0.98)]), height, width)

    assert face.score == pytest.approx(0.98)
    assert np.allclose(face.box, box, atol=1e-3)
    assert np.allclose(np.array(face.landmarks), face_points(box), atol=1e-3)


def test_decode_detections_applies_the_score_threshold_inclusively() -> None:
    height, width = 480, 640
    faces = [
        (prior_index_near(height, width, 100, 100, 32), np.array([80.0, 80.0, 120.0, 124.0], np.float32), 0.9),
        (prior_index_near(height, width, 500, 300, 32), np.array([480.0, 280.0, 520.0, 324.0], np.float32), 0.89),
    ]

    found = decode_detections(synthetic_outputs(height, width, faces), height, width)

    assert SCORE_THRESHOLD == 0.9
    assert [round(face.score, 2) for face in found] == [0.9]


def test_decode_detections_merges_duplicates_of_one_face() -> None:
    height, width = 480, 640
    box = np.array([300.0, 150.0, 364.0, 230.0], np.float32)
    shifted = box + np.array([2.0, 1.0, 2.0, 1.0], np.float32)
    first = prior_index_near(height, width, 332, 190, 64)
    second = prior_index_near(height, width, 332, 190, 128)
    outputs = synthetic_outputs(height, width, [(first, box, 0.95), (second, shifted, 0.99)])

    (face,) = decode_detections(outputs, height, width)

    assert face.score == pytest.approx(0.99)
    assert np.allclose(face.box, shifted, atol=1e-3)


def test_decode_detections_sorts_faces_by_score() -> None:
    height, width = 480, 640
    faces = [
        (prior_index_near(height, width, 100, 100, 32), np.array([80.0, 80.0, 120.0, 124.0], np.float32), 0.93),
        (prior_index_near(height, width, 500, 300, 32), np.array([480.0, 280.0, 520.0, 324.0], np.float32), 0.99),
    ]

    found = decode_detections(synthetic_outputs(height, width, faces), height, width)

    assert [round(face.score, 2) for face in found] == [0.99, 0.93]


def test_pre_nms_candidates_are_capped() -> None:
    assert PRE_NMS_TOP_K == 5000


# ---------------------------------------------------------------- entrada de la red


def test_network_input_is_bgr_0_255_minus_the_mean() -> None:
    rgb = np.zeros((2, 3, 3), np.uint8)
    rgb[..., 0] = 200
    rgb[..., 1] = 100
    rgb[..., 2] = 50

    batch = network_input(rgb)

    assert batch.shape == (1, 3, 2, 3)
    assert batch.dtype == np.float32
    assert batch[0, :, 0, 0].tolist() == pytest.approx([50 - 104.0, 100 - 117.0, 200 - 123.0])
    assert BGR_MEAN.tolist() == [104.0, 117.0, 123.0]


@pytest.mark.parametrize("dtype", [np.float32, np.uint16])
def test_network_input_reads_float_and_16_bit_photos_on_the_8_bit_scale(dtype) -> None:
    u8 = np.random.default_rng(0).integers(0, 256, (4, 5, 3)).astype(np.uint8)
    other = u8.astype(np.float32) / 255 if dtype == np.float32 else u8.astype(np.uint16) * 257

    assert np.allclose(network_input(other), network_input(u8), atol=1e-3)


def test_network_input_drops_alpha() -> None:
    rgba = np.zeros((2, 2, 4), np.uint8)

    assert network_input(rgba).shape == (1, 3, 2, 2)


def test_network_input_rejects_gray_images() -> None:
    with pytest.raises(ValueError, match="RGB"):
        network_input(np.zeros((4, 4), np.uint8))


# ---------------------------------------------------------------- deteccion sobre una copia de lado mayor 1280


class ScriptedRun:
    def __init__(self, faces_at_detection_size: list[tuple[int, np.ndarray, float]]) -> None:
        self.faces = faces_at_detection_size
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, batch: np.ndarray):
        self.shapes.append(batch.shape)
        return synthetic_outputs(batch.shape[2], batch.shape[3], self.faces)


def test_detect_faces_runs_on_a_copy_with_long_side_1280_and_scales_back() -> None:
    photo = np.zeros((1920, 2560, 3), np.float32)
    box = np.array([300.0, 150.0, 364.0, 230.0], np.float32)
    run = ScriptedRun([(prior_index_near(960, 1280, 332, 190, 64), box, 0.97)])

    (face,) = detect_faces(photo, run)

    assert run.shapes == [(1, 3, 960, DETECT_MAX_SIDE)]
    assert np.allclose(face.box, box * 2, atol=1e-2)
    assert np.allclose(np.array(face.landmarks), face_points(box) * 2, atol=1e-2)
    assert face.eye_px == pytest.approx(0.4 * 64 * 2, rel=1e-4)


def test_detect_faces_keeps_small_photos_at_native_size() -> None:
    run = ScriptedRun([])

    assert detect_faces(np.zeros((300, 400, 3), np.uint8), run) == ()
    assert run.shapes == [(1, 3, 300, 400)]


def test_a_detection_becomes_the_diagnosis_face_with_its_eye_distance() -> None:
    face = FaceDetection(
        box=(10.0, 20.0, 60.0, 90.0),
        score=0.95,
        landmarks=((20.0, 40.0), (50.0, 40.0), (35.0, 55.0), (25.0, 70.0), (45.0, 70.0)),
    )

    assert face.to_detected_face() == DetectedFace(box=(10.0, 20.0, 60.0, 90.0), score=0.95, eye_px=30.0)
    assert face.scaled(2.0).landmarks[1] == (100.0, 80.0)


# ---------------------------------------------------------------- sesion real en el CPU EP via el dueno unico

GRAPH_SIDE = 64


def write_constant_retinaface(path: Path, outputs) -> None:
    from onnx import TensorProto, helper, numpy_helper, save

    nodes = [helper.make_node("ReduceMean", ["input"], ["mean"], keepdims=0)]
    nodes.append(helper.make_node("Mul", ["mean", "zero"], ["nothing"]))
    initializers = [numpy_helper.from_array(np.array(0.0, np.float32), "zero")]
    graph_outputs = []
    for name, value in zip(("loc", "conf", "landmarks"), outputs, strict=True):
        initializers.append(numpy_helper.from_array(value.astype(np.float32), f"{name}_value"))
        nodes.append(helper.make_node("Add", [f"{name}_value", "nothing"], [name]))
        graph_outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, list(value.shape)))
    graph = helper.make_graph(
        nodes,
        "fake_retinaface",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, GRAPH_SIDE, GRAPH_SIDE])],
        graph_outputs,
        initializer=initializers,
    )
    save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(path))


def retinaface_spec() -> RestoreModelSpec:
    return RestoreModelSpec(
        id=RETINAFACE_MODEL_ID,
        name="RetinaFace-R34",
        bundle="faces",
        filename="fake-retinaface.onnx",
        license_spdx="MIT",
        license_url="https://example.com/LICENSE",
        copyright="Copyright (c) test",
        attribution="Test model",
        data_lineage="D1b + D1a",
        commercial_use="yes",
        source_url="https://example.com/model",
        source_revision="abc123",
        source_sha256="a" * 64,
        modifications=("none",),
        tile_min=GRAPH_SIDE,
        fixed_shape=True,
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


def engine_with_graph(tmp_path: Path, outputs) -> tuple[PhotoRestoreEngine, CountingCoordinator]:
    model_dir = tmp_path / "restore"
    model_dir.mkdir()
    write_constant_retinaface(model_dir / "fake-retinaface.onnx", outputs)
    settings = Settings(_env_file=None, RESTORE_MODEL_DIR=str(model_dir), RUNTIME_DIR=str(tmp_path / "runtime"))
    coordinator = CountingCoordinator()
    engine = PhotoRestoreEngine(settings, coordinator, models={RETINAFACE_MODEL_ID: retinaface_spec()})
    return engine, coordinator


def test_engine_detector_runs_retinaface_on_cpu_through_the_single_owner(tmp_path: Path) -> None:
    box = np.array([20.0, 16.0, 44.0, 48.0], np.float32)
    index = prior_index_near(GRAPH_SIDE, GRAPH_SIDE, 32, 32, 32)
    engine, coordinator = engine_with_graph(
        tmp_path, synthetic_outputs(GRAPH_SIDE, GRAPH_SIDE, [(index, box, 0.96)])
    )

    (face,) = landmarked_face_detector(engine)(np.zeros((GRAPH_SIDE, GRAPH_SIDE, 3), np.float32))

    assert coordinator.acquired == ["cpu"]
    assert [key.device for key in engine.live_sessions("cpu")] == ["cpu"]
    assert np.allclose(face.box, box, atol=1e-3)


def test_diagnosis_detector_returns_detected_faces(tmp_path: Path) -> None:
    box = np.array([20.0, 16.0, 44.0, 48.0], np.float32)
    index = prior_index_near(GRAPH_SIDE, GRAPH_SIDE, 32, 32, 32)
    engine, _ = engine_with_graph(tmp_path, synthetic_outputs(GRAPH_SIDE, GRAPH_SIDE, [(index, box, 0.96)]))

    (face,) = face_detector(engine)(np.zeros((GRAPH_SIDE, GRAPH_SIDE, 3), np.float32))

    assert isinstance(face, DetectedFace)
    assert face.eye_px == pytest.approx(0.4 * 24, rel=1e-4)


class NaNSession:
    class _Input:
        name = "input"

    def get_inputs(self):
        return [self._Input()]

    def run(self, names, feeds):
        count = len(priors(GRAPH_SIDE, GRAPH_SIDE))
        return [np.full((1, count, 4), np.nan, np.float32), np.zeros((1, count, 2)), np.zeros((1, count, 10))]


def test_non_finite_outputs_fail_instead_of_hiding_faces() -> None:
    with pytest.raises(NonFiniteOutputError):
        session_run(NaNSession())(np.zeros((1, 3, GRAPH_SIDE, GRAPH_SIDE), np.float32))
