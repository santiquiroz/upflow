from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services.editor_segmenter import (
    SAM_INPUT_SIDE,
    SAM_MEAN_PIXEL,
    EditorSegmenter,
    SegmenterUnavailableError,
    build_decoder_inputs,
    mask_to_png_bytes,
    prepare_encoder_input,
)
from app.services.missing_pack import PACK_LABELS


def make_settings(tmp_path: Path, installed: bool = True) -> Settings:
    encoder = tmp_path / "encoder.onnx"
    decoder = tmp_path / "decoder.onnx"
    if installed:
        encoder.write_bytes(b"onnx")
        decoder.write_bytes(b"onnx")
    return Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        EDITOR_SEGMENTER_ENCODER=str(encoder),
        EDITOR_SEGMENTER_DECODER=str(decoder),
    )


# --- helpers puros ---------------------------------------------------------


def test_prepare_encoder_input_pads_to_fixed_square_with_mean_pixel() -> None:
    rgb = np.full((512, 256, 3), 200, dtype=np.uint8)

    canvas, scale = prepare_encoder_input(rgb)

    assert canvas.shape == (SAM_INPUT_SIDE, SAM_INPUT_SIDE, 3)
    assert canvas.dtype == np.float32
    assert scale == pytest.approx(2.0)
    assert canvas[:1024, :512].mean() == pytest.approx(200.0, abs=1.0)
    np.testing.assert_allclose(canvas[0, 600], np.array(SAM_MEAN_PIXEL, dtype=np.float32))


def test_build_decoder_inputs_scales_click_and_pads_point() -> None:
    embeddings = np.zeros((1, 256, 64, 64), dtype=np.float32)

    inputs = build_decoder_inputs(embeddings, x=100.0, y=50.0, scale=2.0, orig_h=512, orig_w=256)

    np.testing.assert_allclose(inputs["point_coords"][0, 0], [200.0, 100.0])
    np.testing.assert_allclose(inputs["point_coords"][0, 1], [0.0, 0.0])
    np.testing.assert_allclose(inputs["point_labels"][0], [1.0, -1.0])
    np.testing.assert_allclose(inputs["orig_im_size"], [512.0, 256.0])
    assert inputs["mask_input"].shape == (1, 1, 256, 256)
    assert float(inputs["has_mask_input"][0]) == 0.0


def test_mask_to_png_is_grayscale_white_object() -> None:
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True

    decoded = Image.open(io.BytesIO(mask_to_png_bytes(mask)))

    assert decoded.mode == "L"
    pixels = np.asarray(decoded)
    assert pixels[1, 2] == 255
    assert pixels[0, 0] == 0


# --- servicio (sesiones fake, sin onnxruntime real) -------------------------


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, _outputs, inputs):
        self.calls += 1
        assert inputs["input_image"].shape == (SAM_INPUT_SIDE, SAM_INPUT_SIDE, 3)
        return [np.zeros((1, 256, 64, 64), dtype=np.float32)]


class FakeDecoder:
    def __init__(self) -> None:
        self.inputs: list[dict] = []

    def run(self, _outputs, inputs):
        self.inputs.append(inputs)
        orig_h, orig_w = (int(v) for v in inputs["orig_im_size"])
        masks = np.full((1, 1, orig_h, orig_w), -1.0, dtype=np.float32)
        masks[0, 0, :10, :10] = 1.0
        return [masks, np.ones((1, 1), dtype=np.float32), np.zeros((1, 1, 256, 256), np.float32)]


def make_segmenter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[EditorSegmenter, FakeEncoder, FakeDecoder, Path]:
    settings = make_settings(tmp_path)
    segmenter = EditorSegmenter(settings)
    encoder, decoder = FakeEncoder(), FakeDecoder()

    def fake_create(model_path: Path, device: str):
        return encoder if "encoder" in str(model_path) else decoder

    monkeypatch.setattr(segmenter, "_create_session", fake_create)
    image_path = tmp_path / "foto.png"
    Image.new("RGB", (64, 32), (10, 20, 30)).save(image_path)
    return segmenter, encoder, decoder, image_path


@pytest.mark.asyncio
async def test_segment_returns_png_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segmenter, _, decoder, image_path = make_segmenter(tmp_path, monkeypatch)

    png = await segmenter.segment(image_path, 5.0, 5.0, "dml:0")

    decoded = Image.open(io.BytesIO(png))
    assert decoded.size == (64, 32)
    assert np.asarray(decoded)[5, 5] == 255
    assert np.asarray(decoded)[20, 40] == 0
    coords = decoder.inputs[0]["point_coords"][0, 0]
    assert coords[0] == pytest.approx(5.0 * (SAM_INPUT_SIDE / 64))


@pytest.mark.asyncio
async def test_embeddings_are_cached_per_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    segmenter, encoder, _, image_path = make_segmenter(tmp_path, monkeypatch)

    await segmenter.segment(image_path, 1.0, 1.0, "dml:0")
    await segmenter.segment(image_path, 9.0, 9.0, "dml:0")

    assert encoder.calls == 1


@pytest.mark.asyncio
async def test_segment_sin_modelos_dice_que_paquete_falta_sin_dictar_comandos(tmp_path: Path) -> None:
    segmenter = EditorSegmenter(make_settings(tmp_path, installed=False))

    with pytest.raises(SegmenterUnavailableError) as error:
        await segmenter.segment(tmp_path / "x.png", 1.0, 1.0, "cpu")

    assert PACK_LABELS["mobilesam"] in str(error.value)
    assert ".ps1" not in str(error.value)
