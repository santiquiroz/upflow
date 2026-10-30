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

    _, choices = await eraser.erase(base, mask, "cpu")

    factory.assert_called_once_with("cpu")
    assert choices[0]["engine"] in {"hibrido", "sdxl"}
    assert "scores" in choices[0]


@pytest.mark.asyncio
async def test_auto_eraser_without_osor_or_sdxl_falls_back_to_lama():
    base, mask = _image_and_mask()
    lama = Mock()
    lama.fill_group.side_effect = lambda canvas, *_args: canvas
    factory = Mock(return_value=None)
    eraser = _auto_eraser(lama, None, factory)

    _, choices = await eraser.erase(base, mask, "cpu")

    factory.assert_called_once_with("cpu")
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
            osor_available=Mock(return_value=True),
            erase=AsyncMock(
                return_value=(Image.new("RGB", (32, 32)), [{"engine": "lama"}])
            ),
        ),
    )
    base_path, mask_path, protect_path = (
        tmp_path / "base.png", tmp_path / "mask.png", tmp_path / "protect.png"
    )
    Image.new("RGB", (32, 32)).save(base_path)
    Image.new("L", (32, 32), 255).save(mask_path)
    Image.new("L", (32, 32), 0).save(protect_path)

    job = await manager.create_job(
        prompt="borrar",
        model_id=ERASER_AUTO_MODEL_ID,
        device="dml:0",
        init_image_path=base_path,
        mask_image_path=mask_path,
        protect_mask_path=protect_path,
        erase_engine="lama",
        erase_grow_px=7,
        erase_feather_px=3,
    )
    await manager._run_engine(job)

    assert manager._reservation_device(job) == "cpu"
    assert job.metadata["eraseAuto"] == [{"engine": "lama"}]
    call = manager.auto_eraser.erase.await_args
    assert call.kwargs["protect_image"].size == (32, 32)
    assert call.kwargs["engine"] == "lama"
    assert call.kwargs["grow_px"] == 7
    assert call.kwargs["feather_px"] == 3


@pytest.mark.parametrize(
    ("osor_available", "erase_engine", "expected"),
    [(False, None, "dml:0"), (True, None, "cpu"), (True, "sdxl", "dml:0")],
)
def test_auto_eraser_reserves_device_based_on_osor_and_engine(
    tmp_path: Path, osor_available: bool, erase_engine: str | None, expected: str
):
    from app.config import Settings
    from app.models import GenerationJob
    from app.services.device_semaphores import DeviceSemaphores
    from app.services.generation_job_manager import GenerationJobManager
    from app.services.model_registry import ModelRegistry

    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None).model_copy(
        update={"default_device": "dml:0"}
    )
    auto_eraser = Mock(
        osor_available=Mock(return_value=osor_available),
    )
    manager = GenerationJobManager(
        settings,
        engine=None,
        device_semaphores=DeviceSemaphores(settings),
        registry=ModelRegistry(settings),
        upscale_engine=None,
        auto_eraser=auto_eraser,
    )
    job = GenerationJob(
        prompt="borrar",
        model_id=ERASER_AUTO_MODEL_ID,
        erase_engine=erase_engine,
    )

    assert manager._reservation_device(job) == expected


@pytest.mark.asyncio
async def test_auto_eraser_growth_expands_mask_before_fill():
    base, mask = _image_and_mask(320)
    captured = []
    lama = Mock()
    lama.fill_group.side_effect = (
        lambda canvas, group, _core: captured.append(group.copy()) or canvas
    )

    eraser = _auto_eraser(lama, None, Mock(return_value=None))
    await eraser.erase(base, mask, "cpu", grow_px=0)
    await eraser.erase(base, mask, "cpu", grow_px=5)

    assert np.count_nonzero(captured[1]) > np.count_nonzero(captured[0])


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
