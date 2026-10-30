from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services.engines import osor_eraser as eraser_module
from app.services.engines.osor_eraser import (
    OsorEraser,
    grow,
    osor_crop_box,
    remove_render_shift,
)


def osor_pack_files(pack: Path) -> None:
    for relative in (
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "scheduler/scheduler_config.json",
        "prompt_embeds.pt",
    ):
        path = pack / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def test_crop_box_uses_context_ratio_and_pad_when_space_is_available() -> None:
    mask = np.zeros((1200, 1400), dtype=bool)
    mask[300:500, 400:700] = True

    left, top, right, bottom = osor_crop_box(mask)

    assert right - left == bottom - top == int(300 * 1.5 + 64)
    assert left <= 400 and top <= 300 and right >= 700 and bottom >= 500


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, 40, 30),
        (90, 70, 120, 100),
        (0, 50, 40, 100),
        (70, 0, 120, 35),
    ],
)
def test_crop_box_clamps_each_axis_to_its_image_dimension(box) -> None:
    mask = np.zeros((100, 120), dtype=bool)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = True

    crop = osor_crop_box(mask)

    assert 0 <= crop[0] < crop[2] <= mask.shape[1]
    assert 0 <= crop[1] < crop[3] <= mask.shape[0]


def test_remove_render_shift_corrects_uniform_shift_outside_changed_area() -> None:
    original = np.full((100, 120, 3), 100, dtype=np.float32)
    filled = original + 20
    filled[35:65, 40:80] += 50
    changed = np.zeros(original.shape[:2], dtype=bool)
    changed[35:65, 40:80] = True

    result = remove_render_shift(filled, original, changed)

    np.testing.assert_allclose(result[~changed], original[~changed], atol=0.1)
    np.testing.assert_allclose(
        result[45:55, 50:70], original[45:55, 50:70] + 50, atol=1e-4
    )


class FakeOsorModel:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def infer(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append((image.copy(), mask.copy()))
        alpha = np.zeros(mask.shape, dtype=np.float32)
        ys, xs = np.nonzero(mask)
        alpha[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1.0
        extra_left = min(mask.shape[1] - 12, xs.max() + 5)
        middle_y = (ys.min() + ys.max()) // 2
        alpha[middle_y - 10:middle_y + 10, extra_left:extra_left + 12] = 1.0
        return np.full((*mask.shape, 3), (0.25, 0.5, 0.75), dtype=np.float32), alpha


def test_fill_group_resizes_to_512_and_stitches_mask_alpha_and_protection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = tmp_path / "pack"
    osor_pack_files(pack)
    settings = Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        OSOR_MODEL_DIR=str(pack),
    )
    eraser = OsorEraser(settings)
    model = FakeOsorModel()
    monkeypatch.setattr(eraser, "_create_model", lambda: model)
    monkeypatch.setattr(eraser_module, "remove_render_shift", lambda filled, *_: filled)

    rows, columns = np.indices((1000, 1000))
    canvas = np.stack(
        (
            20 + columns * 140 // 999,
            50 + rows * 156 // 999,
            np.full(rows.shape, 191),
        ),
        axis=2,
    ).astype(np.uint8)
    group = np.zeros(canvas.shape[:2], dtype=np.uint8)
    group[350:650, 350:650] = 255
    protect = np.zeros(group.shape, dtype=bool)
    protect[400:430, 400:430] = True
    mask = grow(group > 0, 0.03)
    left, top, right, bottom = osor_crop_box(mask)

    result = eraser.fill_group(canvas, group, protect)

    assert result is not canvas
    assert len(model.calls) == 1
    model_image, model_mask = model.calls[0]
    assert min(model_image.shape[:2]) == 512
    assert set(np.unique(model_mask)) == {False, True}
    mask_y, mask_x = np.nonzero(model_mask)
    extra_left = min(model_mask.shape[1] - 12, mask_x.max() + 5)
    middle_y = (mask_y.min() + mask_y.max()) // 2
    alpha_x = round(left + (extra_left + 6) * (right - left) / model_mask.shape[1])
    alpha_y = round(top + middle_y * (bottom - top) / model_mask.shape[0])
    np.testing.assert_array_equal(result[:top], canvas[:top])
    np.testing.assert_array_equal(result[bottom:], canvas[bottom:])
    np.testing.assert_array_equal(result[:, :left], canvas[:, :left])
    np.testing.assert_array_equal(result[:, right:], canvas[:, right:])
    np.testing.assert_array_equal(result[protect], canvas[protect])
    assert np.all(np.abs(result[500, 500].astype(int) - [64, 128, 191]) <= 4)
    assert np.all(np.abs(result[alpha_y, alpha_x].astype(int) - [64, 128, 191]) <= 4)
    assert not np.array_equal(result[alpha_y, alpha_x], canvas[alpha_y, alpha_x])


def test_available_requires_all_six_pack_files(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    settings = Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        OSOR_MODEL_DIR=str(pack),
    )
    eraser = OsorEraser(settings)

    assert eraser.available() is False
    osor_pack_files(pack)
    assert eraser.available() is True


def test_settings_osor_default_and_override(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        MODELS_DIR="models",
        OSOR_MODEL_DIR="",
    )
    assert settings.osor_model_dir == ""
    assert settings.osor_model_path == tmp_path / "runtime" / "models" / "eraser-osor"

    override = tmp_path / "custom-pack"
    configured = Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        OSOR_MODEL_DIR=str(override),
    )
    assert configured.osor_model_path == override


@pytest.mark.skipif(
    os.environ.get("UPFLOW_OSOR_REAL") != "1",
    reason="set UPFLOW_OSOR_REAL=1 to run real OSOR inference",
)
def test_real_pack_runs_on_synthetic_image() -> None:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        pytest.skip("LOCALAPPDATA is unavailable")
    pack = Path(local_app_data) / "Upflow" / "runtime" / "models" / "eraser-osor"
    settings = Settings(_env_file=None, OSOR_MODEL_DIR=str(pack))
    eraser = OsorEraser(settings)
    if not eraser.available():
        pytest.skip("the real OSOR pack is unavailable")
    rows, columns = np.indices((700, 900))
    canvas = np.stack(
        (columns % 256, rows % 256, (rows + columns) % 256), axis=2
    ).astype(np.uint8)
    group = np.zeros(canvas.shape[:2], dtype=np.uint8)
    group[200:500, 300:600] = 255

    result = eraser.fill_group(canvas, group, np.zeros(group.shape, dtype=bool))

    assert result.shape == canvas.shape
    assert result.dtype == np.uint8
    assert np.isfinite(result).all()
