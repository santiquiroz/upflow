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
    assert job.strength == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_uploaded_protect_token_is_stored_on_the_generation_job(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")
    protect_token = await upload_token(
        settings, png_bytes((128, 128), color=(255, 255, 255)), "protect.png"
    )

    response = await create_generation_job(
        make_payload(
            initImageToken=init_token, maskImageToken=mask_token, protectMaskToken=protect_token
        ),
        generation_jobs=manager,
        settings=settings,
    )

    job = manager.get_job(response.id)
    assert job.protect_mask_path is not None
    assert job.protect_mask_path.is_file()
    assert job.protect_mask_path != job.mask_image_path
    from PIL import Image

    with Image.open(job.protect_mask_path) as protect:
        assert protect.size == (128, 128)
        assert protect.convert("L").getpixel((64, 64)) == 255


@pytest.mark.asyncio
async def test_unknown_protect_token_is_rejected_before_enqueue(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")

    with pytest.raises(HTTPException) as error:
        await create_generation_job(
            make_payload(
                initImageToken=init_token, maskImageToken=mask_token,
                protectMaskToken="deadbeef",
            ),
            generation_jobs=manager,
            settings=settings,
        )

    assert error.value.status_code == 400
    assert manager.jobs == {}


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
async def test_auto_erase_without_a_mask_is_a_400(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))

    with pytest.raises(HTTPException) as error:
        await create_generation_job(
            make_payload(initImageToken=init_token, eraseStrategy="auto"),
            generation_jobs=manager,
            settings=settings,
        )

    assert error.value.status_code == 400
    assert manager.jobs == {}


@pytest.mark.asyncio
async def test_auto_erase_with_a_mask_is_carried_to_the_job(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")

    response = await create_generation_job(
        make_payload(
            initImageToken=init_token, maskImageToken=mask_token, eraseStrategy="auto"
        ),
        generation_jobs=manager,
        settings=settings,
    )

    job = manager.get_job(response.id)
    assert job.erase_strategy == "auto"
    assert job.mask_image_path is not None


@pytest.mark.asyncio
async def test_erase_mask_adjustments_are_carried_to_the_job(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")

    response = await create_generation_job(
        make_payload(
            initImageToken=init_token, maskImageToken=mask_token,
            eraseStrategy="auto", maskGrowPx=-12, maskFeatherPx=9,
        ),
        generation_jobs=manager, settings=settings,
    )

    job = manager.get_job(response.id)
    assert job.erase_grow_px == -12
    assert job.erase_feather_px == 9


@pytest.mark.asyncio
async def test_auto_erase_request_carries_protect_and_records_engine_choices(tmp_path: Path):
    manager, settings, engine = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")
    protect_token = await upload_token(settings, png_bytes((128, 128)), "protect.png")
    response = await create_generation_job(
        make_payload(
            initImageToken=init_token, maskImageToken=mask_token,
            protectMaskToken=protect_token, eraseStrategy="auto", device="cpu",
        ),
        generation_jobs=manager, settings=settings,
    )
    choices = [{"engine": "lama", "index": 0, "box": [0, 0, 128, 128]}]
    original_run = engine.run

    async def run_with_choices(**kwargs):
        kwargs["request"].erase_auto = choices
        return await original_run(**kwargs)

    engine.run = run_with_choices
    job = manager.get_job(response.id)

    await manager._run_engine(job)

    request = engine.requests[0]
    assert request.erase_strategy == "auto"
    assert request.protect_mask_path == job.protect_mask_path
    assert request.mask_image_path == job.mask_image_path
    assert job.metadata["eraseAuto"] == choices


@pytest.mark.parametrize(
    "payload",
    [
        {"maskGrowPx": -201},
        {"maskGrowPx": 201},
        {"maskFeatherPx": -1},
        {"maskFeatherPx": 101},
    ],
)
def test_erase_mask_adjustments_validate_ranges(payload):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        make_payload(**payload)


@pytest.mark.asyncio
async def test_forced_erase_engine_is_carried_to_generation_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(Settings, "lama_available", lambda _settings: True)
    manager, settings, engine = make_manager(tmp_path)
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")

    response = await create_generation_job(
        make_payload(
            initImageToken=init_token, maskImageToken=mask_token,
            eraseEngine="lama", device="cpu",
        ),
        generation_jobs=manager, settings=settings,
    )
    job = manager.get_job(response.id)
    await manager._run_engine(job)

    assert engine.requests[0].erase_engine == "lama"


@pytest.mark.asyncio
async def test_forced_erase_engine_without_mask_is_400(tmp_path: Path):
    manager, settings, _ = make_manager(tmp_path)

    with pytest.raises(HTTPException) as error:
        await create_generation_job(
            make_payload(eraseEngine="lama"),
            generation_jobs=manager, settings=settings,
        )

    assert error.value.status_code == 400
    assert manager.jobs == {}


@pytest.mark.asyncio
async def test_forced_osor_without_pack_is_400(tmp_path: Path, monkeypatch):
    manager, settings, _ = make_manager(tmp_path)
    monkeypatch.setattr(Settings, "osor_available", lambda _settings: False)
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")

    with pytest.raises(HTTPException) as error:
        await create_generation_job(
            make_payload(
                initImageToken=init_token, maskImageToken=mask_token, eraseEngine="osor",
            ),
            generation_jobs=manager, settings=settings,
        )

    assert error.value.status_code == 400
    assert "OSOR" in error.value.detail
    assert manager.jobs == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("erase_engine", ["lama", "lama+osor"])
async def test_forced_lama_without_pack_is_400(
    tmp_path: Path, monkeypatch, erase_engine: str
):
    manager, settings, _ = make_manager(tmp_path)
    monkeypatch.setattr(Settings, "lama_available", lambda _settings: False)
    monkeypatch.setattr(Settings, "osor_available", lambda _settings: True)
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")

    with pytest.raises(HTTPException) as error:
        await create_generation_job(
            make_payload(
                initImageToken=init_token,
                maskImageToken=mask_token,
                eraseEngine=erase_engine,
            ),
            generation_jobs=manager, settings=settings,
        )

    assert error.value.status_code == 400
    assert "LaMa" in error.value.detail
    assert manager.jobs == {}


@pytest.mark.asyncio
async def test_forced_sdxl_without_inpaint_model_is_400(tmp_path: Path):
    manager, settings, _ = make_manager(
        tmp_path, declared_class="StableDiffusion3Pipeline"
    )
    init_token = await upload_token(settings, png_bytes((128, 128)))
    mask_token = await upload_token(settings, png_bytes((128, 128)), "mask.png")

    with pytest.raises(HTTPException) as error:
        await create_generation_job(
            make_payload(
                initImageToken=init_token, maskImageToken=mask_token, eraseEngine="sdxl",
            ),
            generation_jobs=manager, settings=settings,
        )

    assert error.value.status_code == 400
    assert "SDXL" in error.value.detail
    assert manager.jobs == {}


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
