from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from PIL import Image

from app.services.engines.auto_eraser import (
    ERASER_AUTO_MODEL_ID,
    AutoEraser,
)


def _image_and_mask(size: int = 1000) -> tuple[Image.Image, Image.Image]:
    base = Image.new("RGB", (size, size), (80, 100, 120))
    mask = Image.new("L", (size, size), 0)
    mask.paste(255, (100, 100, size - 100, size - 100))
    return base, mask


def _auto_eraser(lama, osor, sdxl_factory) -> AutoEraser:
    return AutoEraser(
        settings=Mock(),
        lama_eraser=lama,
        osor_eraser=osor,
        people_finder_factory=lambda _device: None,
        sdxl_edit_factory=sdxl_factory,
    )


@pytest.mark.asyncio
async def test_auto_eraser_never_builds_sdxl_when_osor_handles_large():
    base, mask = _image_and_mask()
    lama = Mock()
    lama.fill_group.side_effect = lambda canvas, *_args: canvas
    osor = Mock()
    osor.available.return_value = True
    osor.fill_group.side_effect = lambda canvas, *_args: canvas
    sdxl_factory = Mock(side_effect=AssertionError("SDXL no debe cargarse"))
    eraser = _auto_eraser(lama, osor, sdxl_factory)

    _, choices = await eraser.erase(base, mask, "cpu")

    assert choices[0]["engine"] == "osor"
    sdxl_factory.assert_not_called()


@pytest.mark.asyncio
async def test_auto_eraser_without_osor_uses_sdxl_factory_once():
    base, mask = _image_and_mask()
    lama = Mock()
    lama.fill_group.side_effect = lambda canvas, *_args: canvas
    edit = Mock(side_effect=lambda image, *_args: image)
    factory = Mock(return_value=edit)
    eraser = _auto_eraser(lama, None, factory)

    await eraser.erase(base, mask, "cpu")

    factory.assert_called_once_with()
    assert edit.call_count == 2


@pytest.mark.asyncio
async def test_auto_eraser_without_osor_or_sdxl_falls_back_to_lama():
    base, mask = _image_and_mask()
    lama = Mock()
    lama.fill_group.side_effect = lambda canvas, *_args: canvas
    factory = Mock(return_value=None)
    eraser = _auto_eraser(lama, None, factory)

    _, choices = await eraser.erase(base, mask, "cpu")

    factory.assert_called_once_with()
    assert choices[0]["engine"] == "lama"


@pytest.mark.asyncio
async def test_eraser_auto_job_records_zones_and_reserves_cpu(tmp_path: Path):
    from app.config import Settings
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.generation_job_manager import GenerationJobManager
    from app.services.model_registry import ModelRegistry
    from app.services.storage import StorageService

    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    manager = GenerationJobManager(
        settings,
        engine=None,
        device_semaphores=DeviceSemaphores(settings),
        registry=ModelRegistry(settings),
        upscale_engine=None,
        auto_eraser=Mock(
            available=Mock(return_value=True),
            erase=AsyncMock(
                return_value=(Image.new("RGB", (32, 32)), [{"engine": "lama"}])
            ),
        ),
    )
    base_path, mask_path = tmp_path / "base.png", tmp_path / "mask.png"
    Image.new("RGB", (32, 32)).save(base_path)
    Image.new("L", (32, 32), 255).save(mask_path)

    job = await manager.create_job(
        prompt="borrar",
        model_id=ERASER_AUTO_MODEL_ID,
        device="dml:0",
        init_image_path=base_path,
        mask_image_path=mask_path,
    )
    await manager._run_engine(job)

    assert manager._reservation_device(job) == "cpu"
    assert job.metadata["eraseAuto"] == [{"engine": "lama"}]


def test_sdxl_inpaint_edit_factory_loads_pipeline_lazily(tmp_path, monkeypatch):
    from app.config import Settings
    from app.services.engines.generation_onnx import GenerationEngine
    from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry
    from app.services.storage import StorageService

    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    registry = ModelRegistry(settings)
    model_path = settings.models_path / "generation" / "sdxl-inpaint"
    model_path.mkdir(parents=True)
    (model_path / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLInpaintPipeline"}),
        encoding="utf-8",
    )
    registry.register(
        ModelEntry(
            id="sdxl-inpaint",
            name="SDXL inpaint",
            kind=ModelKind.diffusion_onnx,
            source="hf:test/sdxl-inpaint",
            size_bytes=1,
            file_path="generation/sdxl-inpaint",
        )
    )
    engine = GenerationEngine(settings, Mock(), registry=registry)
    engine._get_pipeline = Mock(return_value=object())
    engine._masked_edit_model = Mock(
        return_value=lambda image, *_args: image
    )
    from app.services import inpaint_pipeline

    monkeypatch.setattr(
        inpaint_pipeline,
        "run_masked_edit",
        lambda image, _mask, model, _settings: model(image, None, 64, 64),
    )
    edit = engine.sdxl_edit_for_installed_inpaint("dml:0")

    assert edit is not None
    engine._get_pipeline.assert_not_called()
    edit(Image.new("RGB", (64, 64)), np.full((64, 64), 255, dtype=np.uint8), 0.5)
    engine._get_pipeline.assert_called_once()
