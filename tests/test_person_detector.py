import numpy as np
import pytest

from app.config import Settings
from app.services.person_detector import (
    PersonBox,
    PersonDetector,
    PersonDetectorUnavailableError,
    decode_yolox,
    letterbox,
    select_people,
)


def test_person_detector_path_default_and_override(tmp_path, monkeypatch):
    monkeypatch.delenv("PERSON_DETECTOR_MODEL", raising=False)
    default = Settings(_env_file=None)
    assert default.person_detector_model_path.as_posix().endswith("vendor/yolox/yolox_s.onnx")

    model = tmp_path / "det.onnx"
    model.write_bytes(b"x")
    custom = Settings(_env_file=None, PERSON_DETECTOR_MODEL=str(model))
    assert custom.person_detector_model_path == model
    assert custom.person_detector_available() is True


def test_person_detector_unavailable_when_file_missing(tmp_path):
    settings = Settings(_env_file=None, PERSON_DETECTOR_MODEL=str(tmp_path / "no.onnx"))
    assert settings.person_detector_available() is False


def test_letterbox_pads_with_114_and_keeps_ratio():
    rgb = np.zeros((400, 800, 3), np.uint8)
    tensor, ratio = letterbox(rgb)
    assert tensor.shape == (1, 3, 640, 640) and ratio == 0.8
    assert tensor[0, :, 600, 0].tolist() == [114.0, 114.0, 114.0]


def test_letterbox_converts_rgb_to_bgr():
    rgb = np.zeros((1, 1, 3), np.uint8)
    rgb[0, 0] = [255, 0, 0]
    tensor, _ = letterbox(rgb, side=1)
    assert tensor[0, :, 0, 0].tolist() == [0.0, 0.0, 255.0]


def test_decode_places_first_cell_at_stride_8():
    raw = np.zeros((1, 8400, 85), np.float32)
    decoded = decode_yolox(raw)
    assert decoded[0, :4].tolist() == [0.0, 0.0, 8.0, 8.0]
    assert decoded[6400, 2] == 16.0


def test_select_people_scales_back_clips_and_sorts():
    decoded = np.zeros((3, 85), np.float32)
    decoded[0, :6] = [300, 200, 100, 200, 0.9, 0.9]
    decoded[1, :6] = [10, 100, 60, 100, 0.9, 0.9]
    decoded[2, :5] = [100, 100, 50, 50, 0.9]
    decoded[2, 6] = 0.9
    people = select_people(decoded, ratio=0.5, image_size=(1000, 800))
    assert [person.box for person in people] == [
        (0, 100, 80, 300),
        (500, 200, 700, 600),
    ]


def test_select_people_clips_right_and_bottom_edges():
    decoded = np.zeros((1, 85), np.float32)
    decoded[0, :6] = [100, 100, 80, 100, 0.9, 0.9]
    people = select_people(decoded, ratio=1, image_size=(100, 100))
    assert [person.box for person in people] == [(60, 50, 100, 100)]


def test_select_people_restores_vertical_image_coordinates():
    decoded = np.zeros((1, 85), np.float32)
    decoded[0, :6] = [320, 320, 64, 128, 0.9, 0.9]
    ratio = 640 / 6000
    people = select_people(decoded, ratio=ratio, image_size=(4000, 6000))
    assert [person.box for person in people] == [(2700, 2400, 3300, 3600)]


def test_select_people_restores_coordinates_for_small_image():
    decoded = np.zeros((1, 85), np.float32)
    decoded[0, :6] = [320, 320, 200, 200, 0.9, 0.9]
    people = select_people(decoded, ratio=2, image_size=(320, 200))
    assert [person.box for person in people] == [(110, 110, 210, 200)]


def test_select_people_returns_empty_without_person_detections():
    decoded = np.zeros((3, 85), np.float32)
    assert select_people(decoded, ratio=1, image_size=(100, 100)) == []


def test_detect_runs_session_once_per_device_and_reports_missing_pack(tmp_path):
    class FakeSession:
        def __init__(self):
            self.calls = 0

        def run(self, _outputs, inputs):
            self.calls += 1
            assert inputs["images"].shape == (1, 3, 640, 640)
            raw = np.zeros((1, 8400, 85), np.float32)
            raw[0, 0, :6] = [0, 0, np.log(4), np.log(4), 0.9, 0.9]
            return [raw]

    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    detector = PersonDetector(
        Settings(_env_file=None, PERSON_DETECTOR_MODEL=str(model))
    )
    sessions = {}
    create_calls = []

    def create_session(device):
        create_calls.append(device)
        sessions[device] = FakeSession()
        return sessions[device]

    detector._create_session = create_session
    rgb = np.zeros((100, 100, 3), np.uint8)
    expected = [PersonBox(box=(0, 0, 2, 2), score=pytest.approx(0.81))]
    assert detector.detect(rgb) == expected
    assert detector.detect(rgb) == expected
    assert detector.detect(rgb, device="gpu") == expected
    assert detector.detect(rgb, device="gpu") == expected
    assert create_calls == ["cpu", "gpu"]
    assert sessions["cpu"].calls == 2
    assert sessions["gpu"].calls == 2

    settings = Settings(_env_file=None, PERSON_DETECTOR_MODEL=str(tmp_path / "no.onnx"))
    with pytest.raises(PersonDetectorUnavailableError, match="encuentra personas"):
        PersonDetector(settings).detect(rgb)
