from pathlib import Path

import pytest

from PIL import Image

from app.config import Settings
from app.models import UpscaleJob
from app.services.scale_fit import (
    engine_output_path,
    effective_scale,
    final_output_path,
    fit_output_to_scale,
    native_scale_for_engine_model,
    native_scale_of,
    needs_resize,
    save_image,
    scaled_size,
)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


def make_job(tmp_path: Path, *, scale: int, native_scale: int | None, output_format: str = "png") -> UpscaleJob:
    return UpscaleJob(
        source_path=tmp_path / "source.png",
        original_filename="source.png",
        model_name="realesrgan-x4plus",
        scale=scale,
        output_format=output_format,
        native_scale=native_scale,
        id="scale-fit-job",
    )


def test_native_scale_helpers_parse_model_names_and_effective_scale() -> None:
    assert native_scale_for_engine_model("realesrgan-x4plus") == 4
    assert native_scale_for_engine_model("realesr-animevideov3-x2") == 2
    assert native_scale_for_engine_model("realesr-animevideov3-x3") == 3
    assert native_scale_for_engine_model("unknown-model") == 4
    assert effective_scale(2, 4) == 2
    assert effective_scale(4, 2) == 2
    assert effective_scale(3, None) == 3


def test_paths_and_resize_detection_use_requested_and_native_scales(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job = make_job(tmp_path, scale=2, native_scale=4)

    assert native_scale_of(job) == 4
    assert needs_resize(job) is True
    assert final_output_path(settings, job) == settings.outputs_path / "scale-fit-job.png"
    assert engine_output_path(settings, job) == settings.outputs_path / "scale-fit-job.native.png"

    job.native_scale = None
    assert native_scale_of(job) == 2
    assert needs_resize(job) is False
    assert engine_output_path(settings, job) == final_output_path(settings, job)


def test_fit_output_resizes_with_lanczos_and_records_metadata(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job = make_job(tmp_path, scale=2, native_scale=4)
    native_path = engine_output_path(settings, job)
    native_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 24), "red").save(native_path)

    result = fit_output_to_scale(native_path, job, settings)

    assert result == settings.outputs_path / "scale-fit-job.png"
    assert not native_path.exists()
    with Image.open(result) as image:
        assert image.size == (20, 12)
    assert job.metadata["effective"] == {"resized": True, "resizeFilter": "lanczos"}


def test_fit_output_leaves_native_output_untouched_when_scales_match(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job = make_job(tmp_path, scale=4, native_scale=4)
    native_path = final_output_path(settings, job)
    native_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 10), "blue").save(native_path)

    assert fit_output_to_scale(native_path, job, settings) == native_path
    assert native_path.exists()
    assert job.metadata == {}


def test_scaled_size_and_save_image_preserve_expected_output_properties(tmp_path: Path) -> None:
    assert scaled_size((101, 51), 2, 4) == (50, 26)
    target = tmp_path / "nested" / "image.jpeg"

    save_image(Image.new("RGBA", (8, 6), (1, 2, 3, 128)), target)

    with Image.open(target) as image:
        assert image.mode == "RGB"
        assert image.size == (8, 6)



def test_fit_output_removes_the_native_intermediate_even_when_saving_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from PIL import Image as PilImage

    from app.config import Settings as AppSettings
    from app.services import scale_fit

    settings = AppSettings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    job = SimpleNamespace(id="fit-fail", scale=2, native_scale=4, output_format="png", metadata={})
    native = scale_fit.engine_output_path(settings, job)
    native.parent.mkdir(parents=True, exist_ok=True)
    PilImage.new("RGB", (8, 4), "red").save(native)

    def failing_save(image, target):
        raise OSError("disk full")

    monkeypatch.setattr(scale_fit, "save_image", failing_save)

    with pytest.raises(OSError, match="disk full"):
        scale_fit.fit_output_to_scale(native, job, settings)

    assert not native.exists()
    assert not scale_fit.final_output_path(settings, job).exists()
