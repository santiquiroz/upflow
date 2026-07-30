from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import _resolve_init_image, create_generation_job, upload_init_image
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


def png_bytes(size: tuple[int, int] = (64, 48)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
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
) -> tuple[GenerationJobManager, Settings]:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    manager = GenerationJobManager(
        settings,
        FakeEngine(),
        DeviceSemaphores(settings),
        registry=registry,
        upscale_engine=None,
        onnx_upscale_engine=None,
    )
    register_model(registry, settings, declared_class)
    return manager, settings


# ---------------------------------------------------------------------------
# POST /generation/init-image
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uploading_an_init_image_returns_its_token_and_size(tmp_path: Path):
    settings = make_settings(tmp_path)

    response = await upload_init_image(
        file=upload_file("foto.png", png_bytes((80, 40))),
        storage=StorageService(settings),
        settings=settings,
    )

    assert response.width == 80
    assert response.height == 40
    assert response.original_filename == "foto.png"
    assert _resolve_init_image(settings, response.init_image_token) is not None


@pytest.mark.asyncio
async def test_a_file_that_is_not_an_image_is_rejected_and_not_kept(tmp_path: Path):
    settings = make_settings(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await upload_init_image(
            file=upload_file("virus.png", b"no soy una imagen"),
            storage=StorageService(settings),
            settings=settings,
        )

    assert exc_info.value.status_code == 400
    # No queda basura en staging.
    assert list(settings.uploads_path.glob("*")) == []


# ---------------------------------------------------------------------------
# Resolucion del token
# ---------------------------------------------------------------------------


def test_no_token_means_text_to_image(tmp_path: Path):
    assert _resolve_init_image(make_settings(tmp_path), None) is None


@pytest.mark.parametrize(
    "token",
    ["", "../../etc/passwd", "abc/def", "ZZZZ", "a" * 65, "nothex!"],
)
def test_a_malformed_token_is_rejected_before_touching_the_filesystem(tmp_path: Path, token: str):
    # El token viene del cliente: sin validarlo, uno con separadores de ruta haria
    # que el glob mire fuera del directorio de uploads.
    with pytest.raises(HTTPException) as exc_info:
        _resolve_init_image(make_settings(tmp_path), token)
    assert exc_info.value.status_code == 400


def test_a_well_formed_but_unknown_token_is_rejected(tmp_path: Path):
    with pytest.raises(HTTPException) as exc_info:
        _resolve_init_image(make_settings(tmp_path), "deadbeef")
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Crear el job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_job_with_an_init_image_carries_it_and_the_strength(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    uploaded = await upload_init_image(
        file=upload_file("in.png", png_bytes()),
        storage=StorageService(settings),
        settings=settings,
    )

    response = await create_generation_job(
        payload=CreateGenerationJobRequest(
            prompt="un gato",
            modelId=MODEL_ID,
            initImageToken=uploaded.init_image_token,
            strength=0.42,
        ),
        generation_jobs=manager,
        settings=settings,
    )

    job = manager.get_job(response.id)
    assert job.init_image_path is not None
    assert job.strength == 0.42


@pytest.mark.asyncio
async def test_a_job_without_an_init_image_stays_text_to_image(tmp_path: Path):
    manager, settings = make_manager(tmp_path)

    response = await create_generation_job(
        payload=CreateGenerationJobRequest(prompt="un gato", modelId=MODEL_ID),
        generation_jobs=manager,
        settings=settings,
    )

    assert manager.get_job(response.id).init_image_path is None


@pytest.mark.asyncio
async def test_a_text_only_model_is_rejected_with_400_when_creating_the_job(tmp_path: Path):
    """Se rechaza al CREAR y no a mitad de la ejecucion.

    Flux corre perfecto para texto a imagen y no tiene camino de imagen a imagen en
    optimum. Descubrirlo despues de encolar seria un job que falla segundos mas
    tarde con un error de carga de clase.
    """
    manager, settings = make_manager(tmp_path, declared_class="FluxPipeline")
    uploaded = await upload_init_image(
        file=upload_file("in.png", png_bytes()),
        storage=StorageService(settings),
        settings=settings,
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_generation_job(
            payload=CreateGenerationJobRequest(
                prompt="un gato", modelId=MODEL_ID, initImageToken=uploaded.init_image_token
            ),
            generation_jobs=manager,
            settings=settings,
        )

    assert exc_info.value.status_code == 400
    assert "imagen a imagen" in exc_info.value.detail


@pytest.mark.asyncio
async def test_a_text_only_model_still_works_for_text_to_image(tmp_path: Path):
    # El rechazo es de la CAPACIDAD, no del modelo: sin imagen de partida el mismo
    # modelo tiene que seguir funcionando igual que siempre.
    manager, settings = make_manager(tmp_path, declared_class="FluxPipeline")

    response = await create_generation_job(
        payload=CreateGenerationJobRequest(prompt="un gato", modelId=MODEL_ID),
        generation_jobs=manager,
        settings=settings,
    )
    assert manager.get_job(response.id) is not None


@pytest.mark.parametrize("strength", [0, -0.1, 1.1])
def test_a_strength_outside_the_range_is_rejected_by_the_schema(strength: float):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CreateGenerationJobRequest(prompt="x", modelId=MODEL_ID, strength=strength)


@pytest.mark.asyncio
async def test_an_unreadable_model_index_does_not_block_the_job(tmp_path: Path):
    # Una lectura fallida no es un veredicto: el motor va a fallar con su propio
    # mensaje si de verdad no sirve, y rechazar aca convertiria un problema de
    # lectura en "tu modelo no sirve".
    manager, settings = make_manager(tmp_path)
    (settings.models_path / "generation" / MODEL_ID / "model_index.json").write_text(
        "{ no es json", encoding="utf-8"
    )
    uploaded = await upload_init_image(
        file=upload_file("in.png", png_bytes()),
        storage=StorageService(settings),
        settings=settings,
    )

    response = await create_generation_job(
        payload=CreateGenerationJobRequest(
            prompt="un gato", modelId=MODEL_ID, initImageToken=uploaded.init_image_token
        ),
        generation_jobs=manager,
        settings=settings,
    )
    assert manager.get_job(response.id) is not None


# ---------------------------------------------------------------------------
# Autorizacion
# ---------------------------------------------------------------------------


def _required_permissions(path: str, method: str) -> set:
    """Los Permission que una ruta exige de verdad.

    Se lee el closure de require(): las dependencias de servicio (get_storage,
    get_generation_job_manager) difieren por ruta y no dicen nada de seguridad.
    """
    from app.main import app
    from app.services.auth.permissions import Permission

    for route in app.routes:
        if getattr(route, "path", None) != path or method not in getattr(route, "methods", set()):
            continue
        found = set()
        for dependency in route.dependant.dependencies:
            call = dependency.call
            for cell in getattr(call, "__closure__", None) or ():
                if isinstance(cell.cell_contents, Permission):
                    found.add(cell.cell_contents)
        return found
    raise AssertionError(f"ruta no encontrada: {method} {path}")


def test_uploading_an_init_image_requires_the_same_permission_as_creating_a_job():
    """Subir escribe un archivo en el directorio de uploads.

    Sin esto un llamador sin autenticar podia dejar archivos ahi. Todos los
    endpoints que suben o crean jobs en este router exigen jobs_create; este es uno
    mas. Se compara CONTRA el endpoint de crear el job para que endurecer ese no
    deje este atras.
    """
    from app.services.auth.permissions import Permission

    upload = _required_permissions("/api/v1/generation/init-image", "POST")
    create = _required_permissions("/api/v1/generation/jobs", "POST")

    assert Permission.jobs_create in upload
    assert upload == create
