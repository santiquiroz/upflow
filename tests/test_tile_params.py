from pathlib import Path

import pytest
from PIL import Image

from app.config import Settings
from app.models import UpscaleJob
from app.services.tile_params import (
    choose_ncnn_tile,
    onnx_tile_overlap,
    onnx_tile_size,
    validate_tile_params,
)


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_job(tmp_path: Path, *, tile_size: int | None, tile_overlap: int | None) -> UpscaleJob:
    return UpscaleJob(
        source_path=tmp_path / "source.png",
        original_filename="source.png",
        model_name="fake",
        scale=2,
        output_format="png",
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )


@pytest.mark.parametrize(
    ("tile_size", "tile_overlap"),
    [(None, None), (0, 0), (32, 16), (64, 8)],
)
def test_validate_tile_params_accepts_valid_values(tile_size: int | None, tile_overlap: int | None) -> None:
    validate_tile_params(tile_size, tile_overlap)


@pytest.mark.parametrize(
    ("tile_size", "tile_overlap", "message"),
    [
        (16, None, "at least 32"),
        (None, -1, "0 or greater"),
        (32, 17, "at least 2x"),
    ],
)
def test_validate_tile_params_rejects_invalid_values(
    tile_size: int | None, tile_overlap: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_tile_params(tile_size, tile_overlap)


def test_onnx_tile_settings_use_job_override_or_defaults(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ONNX_TILE_SIZE=320)
    default_job = make_job(tmp_path, tile_size=None, tile_overlap=None)
    override_job = make_job(tmp_path, tile_size=512, tile_overlap=24)

    assert onnx_tile_size(default_job, settings) == 320
    assert onnx_tile_overlap(default_job) == 16
    assert onnx_tile_size(override_job, settings) == 512
    assert onnx_tile_overlap(override_job) == 24


def test_choose_ncnn_tile_handles_auto_low_memory_and_image_bounds() -> None:
    assert choose_ncnn_tile(1024, 576, None) == 0
    assert choose_ncnn_tile(1024, 576, 12000) == 640
    assert choose_ncnn_tile(1024, 576, 20000) == 832
    assert choose_ncnn_tile(1024, 576, 40000) == 1024
    assert choose_ncnn_tile(1024, 576, 700) == 32

