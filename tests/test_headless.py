from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app import headless
from app.config import Settings
from app.services.scale_fit import final_output_path


def test_output_and_engine_formats(tmp_path):
    assert headless.output_format_for(Path("a.webp"), None) == "webp"
    assert headless.output_format_for(Path("a.any"), "jxl") == "jxl"
    with pytest.raises(headless.UsageError):
        headless.output_format_for(Path("a.tiff"), None)
    assert headless.engine_format_for("jxl") == "png"
    assert headless.engine_format_for("webp") == "webp"


def test_deterministic_job_id(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    assert headless.deterministic_job_id(source, model="m", scale=2) == headless.deterministic_job_id(
        source, model="m", scale=2
    )
    assert headless.deterministic_job_id(source, model="m", scale=2) != headless.deterministic_job_id(
        source, model="m", scale=3
    )
    assert headless.deterministic_job_id(source, model="m", scale=2).startswith("cli-")


@pytest.mark.asyncio
async def test_headless_context_model_and_device_checks(tmp_path):
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    ctx = headless.build_context(settings)
    ctx.ncnn_engine.available = lambda: False
    with pytest.raises(headless.ModelNotInstalledError):
        headless.ensure_model_installed(ctx, "realesrgan-x4plus")
    ctx.ncnn_engine.available = lambda: True
    headless.ensure_model_installed(ctx, "realesrgan-x4plus")
    with pytest.raises(headless.ModelNotInstalledError):
        headless.ensure_model_installed(ctx, "nope")
    with pytest.raises(headless.DeviceError):
        await headless.resolve_device(ctx, "auto")
    with pytest.raises(headless.DeviceError):
        await headless.resolve_device(ctx, "no-such-device")


@pytest.mark.asyncio
async def test_upscale_image_with_fake_engine(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    ctx = headless.build_context(settings)
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 8), (10, 20, 30)).save(source)
    ctx.devices.validate = lambda device_id: {"id": device_id}
    ctx.ncnn_engine.available = lambda: True

    async def fake_run(job):
        produced = final_output_path(ctx.settings, job)
        produced.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 16), (10, 20, 30)).save(produced)
        job.output_path = produced
        job.metadata["effective"] = {
            "engine": "fake",
            "tileSize": 0,
            "tileOverlap": 10,
            "tileSizeMeaning": "auto",
            "resized": True,
        }
        return produced

    ctx.job_manager.run_inline = fake_run
    output = tmp_path / "out" / "result.png"
    result = await headless.upscale_image(ctx, source, output, scale=2, device="dml:0")
    assert result["ok"] is True
    assert result["width"] == 32
    assert result["height"] == 16
    assert result["scale"] == 2
    assert result["nativeScale"] == 4
    assert result["tile"]["size"] == 0
    assert output.exists()
    assert not any(settings.outputs_path.iterdir())
    assert result["output"] == str(output.resolve())

    async def fake_run_check(job):
        assert job.native_scale == 4
        assert job.scale == 2
        assert job.id.startswith("cli-")
        return await fake_run(job)

    ctx.job_manager.run_inline = fake_run_check
    await headless.upscale_image(ctx, source, tmp_path / "out" / "second.png", scale=2, device="dml:0")


@pytest.mark.asyncio
async def test_upscale_image_jxl_and_invalid_inputs(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    ctx = headless.build_context(settings)
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 8)).save(source)
    ctx.devices.validate = lambda device_id: {"id": device_id}
    ctx.ncnn_engine.available = lambda: True

    async def fake_run(job):
        produced = final_output_path(ctx.settings, job)
        produced.parent.mkdir(parents=True, exist_ok=True)
        source.copy_to(produced) if hasattr(source, "copy_to") else produced.write_bytes(source.read_bytes())
        job.output_path = produced
        job.metadata["effective"] = {"tileSize": 0, "tileOverlap": 10, "resized": False}
        return produced

    ctx.job_manager.run_inline = fake_run
    monkeypatch.setattr(headless, "convert_with_ffmpeg", lambda settings, source, target, fmt: target.write_bytes(source.read_bytes()))
    result = await headless.upscale_image(ctx, source, tmp_path / "result.jxl", scale=2, device="dml:0")
    assert result["format"] == "jxl"
    assert (tmp_path / "result.jxl").exists()
    with pytest.raises(headless.UsageError):
        await headless.upscale_image(ctx, tmp_path / "missing.png", tmp_path / "out.png")
    with pytest.raises(headless.UsageError):
        await headless.upscale_image(ctx, source, tmp_path / "bad.png", scale=9, device="dml:0")


@pytest.mark.asyncio
async def test_ffmpeg_failure_leaves_no_engine_output_behind(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    ctx = headless.build_context(settings)
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 8)).save(source)
    ctx.devices.validate = lambda device_id: {"id": device_id}
    ctx.ncnn_engine.available = lambda: True

    async def fake_run(job):
        produced = final_output_path(ctx.settings, job)
        produced.parent.mkdir(parents=True, exist_ok=True)
        produced.write_bytes(source.read_bytes())
        job.output_path = produced
        job.metadata["effective"] = {"tileSize": 0, "tileOverlap": 10}
        return produced

    ctx.job_manager.run_inline = fake_run

    def failing_convert(settings, source_png, target, fmt):
        raise headless.InferenceError("ffmpeg could not encode jxl: boom")

    monkeypatch.setattr(headless, "convert_with_ffmpeg", failing_convert)

    with pytest.raises(headless.InferenceError, match="boom"):
        await headless.upscale_image(ctx, source, tmp_path / "result.jxl", scale=4, device="dml:0")

    assert not any(settings.outputs_path.iterdir())
    assert not (tmp_path / "result.jxl").exists()
