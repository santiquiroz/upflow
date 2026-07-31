from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import create_generation_job, upload_init_image
from app.config import Settings
from app.schemas import CreateGenerationJobRequest
from app.services.device_semaphores import DeviceSemaphores
from app.services.generation_job_manager import GenerationJobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry
from app.services.storage import StorageService

MODEL_ID = "gen--owner--sdxl"


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


def png_bytes(size: tuple[int, int] = (64, 48), color=(10, 20, 30)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def upload_file(name: str, payload: bytes) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(payload))


class FakeEngine:
    def __init__(self) -> None:
        self.requests: list = []

    async def run(self, **kwargs):
        self.requests.append(kwargs["request"])
        output: Path = kwargs["output_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"png")
        return output


def register_model(
    registry: ModelRegistry, settings: Settings, declared_class: str, model_id: str = MODEL_ID
) -> None:
    model_dir = settings.models_path / "generation" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": declared_class}), encoding="utf-8"
    )
    registry.register(
        ModelEntry(
            id=model_id,
            name="owner/model",
            kind=ModelKind.diffusion_onnx,
            source="hf:owner/model",
            size_bytes=1,
            file_path=f"generation/{model_id}",
        )
    )


def make_manager(
    tmp_path: Path, declared_class: str = "StableDiffusionXLPipeline"
) -> tuple[GenerationJobManager, Settings, FakeEngine]:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    engine = FakeEngine()
    manager = GenerationJobManager(
        settings,
        engine,
        DeviceSemaphores(settings),
        registry=registry,
        upscale_engine=None,
        onnx_upscale_engine=None,
    )
    register_model(registry, settings, declared_class)
    return manager, settings, engine


async def upload_token(settings: Settings, payload: bytes, name: str = "img.png") -> str:
    response = await upload_init_image(
        file=upload_file(name, payload), storage=StorageService(settings), settings=settings
    )
    return response.init_image_token


def make_payload(**overrides) -> CreateGenerationJobRequest:
    base = {"prompt": "un gato con sombrero", "modelId": MODEL_ID}
    base.update(overrides)
    return CreateGenerationJobRequest(**base)


# ---------------------------------------------------------------------------
# POST /generation/jobs con máscara
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_creates_an_inpaint_job_with_default_strength(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128), color=(255, 255, 255)), "mask.png")

    response = await create_generation_job(
        make_payload(initImageToken=init_token, maskImageToken=mask_token),
        generation_jobs=manager,
        settings=settings,
    )

    job = manager.get_job(response.id)
    assert job.mask_image_path is not None
    assert job.init_image_path is not None
    assert job.strength == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_explicit_strength_wins_over_the_inpaint_default(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")

    response = await create_generation_job(
        make_payload(initImageToken=init_token, maskImageToken=mask_token, strength=0.5),
        generation_jobs=manager,
        settings=settings,
    )

    assert manager.get_job(response.id).strength == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_img2img_without_mask_keeps_its_old_default_strength(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))

    response = await create_generation_job(
        make_payload(initImageToken=init_token), generation_jobs=manager, settings=settings
    )

    assert manager.get_job(response.id).strength == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_a_mask_without_base_image_is_a_400(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")

    with pytest.raises(HTTPException) as exc_info:
        await create_generation_job(
            make_payload(maskImageToken=mask_token), generation_jobs=manager, settings=settings
        )

    assert exc_info.value.status_code == 400
    assert "imagen base" in exc_info.value.detail


@pytest.mark.asyncio
async def test_a_model_without_inpaint_support_is_a_400(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path, declared_class="LatentConsistencyModelPipeline")
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")

    with pytest.raises(HTTPException) as exc_info:
        await create_generation_job(
            make_payload(initImageToken=init_token, maskImageToken=mask_token),
            generation_jobs=manager,
            settings=settings,
        )

    assert exc_info.value.status_code == 400
    assert "no soporta inpainting" in exc_info.value.detail


@pytest.mark.asyncio
async def test_an_unknown_mask_token_is_a_400(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))

    with pytest.raises(HTTPException) as exc_info:
        await create_generation_job(
            make_payload(initImageToken=init_token, maskImageToken="deadbeef"),
            generation_jobs=manager,
            settings=settings,
        )

    assert exc_info.value.status_code == 400
