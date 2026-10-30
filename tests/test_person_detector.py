from app.config import Settings


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
