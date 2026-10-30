import numpy as np
import pytest

from app.config import Settings
from app.services.person_detector import (
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


def test_detect_runs_session_once_per_device_and_reports_missing_pack(tmp_path):
    class FakeSession:
        def __init__(self):
            self.calls = 0

        def run(self, _outputs, inputs):
            self.calls += 1
            assert inputs["images"].shape == (1, 3, 640, 640)
            raw = np.zeros((1, 8400, 85), np.float32)
            raw[0, 0, :6] = [0, 0, np.log(10), np.log(10), 0.9, 0.9]
            return [raw]

    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake")
    detector = PersonDetector(
        Settings(_env_file=None, PERSON_DETECTOR_MODEL=str(model))
    )
    session = FakeSession()
    create_calls = []

    def create_session(device):
        create_calls.append(device)
        return session

    detector._create_session = create_session
    rgb = np.zeros((100, 100, 3), np.uint8)
    assert detector.detect(rgb) == detector.detect(rgb)
    assert create_calls == ["cpu"]
    assert session.calls == 2

    settings = Settings(_env_file=None, PERSON_DETECTOR_MODEL=str(tmp_path / "no.onnx"))
    with pytest.raises(PersonDetectorUnavailableError, match="encuentra personas"):
        PersonDetector(settings).detect(rgb)
