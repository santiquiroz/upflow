from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import (
    cancel_transcribe_job,
    create_transcribe_job,
    download_transcribe_job,
    get_transcribe_job,
    transcribe_job_to_response,
)
from app.config import Settings
from app.models import JobStatus
from app.services.device_semaphores import DeviceSemaphores
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry
from app.services.storage import StorageService
from app.services.transcribe_job_manager import TranscribeJobManager

MODEL_ID = "asr--onnx-community--whisper-tiny.en"


class FakeEngine:
    async def run(self, **kwargs) -> str:
        kwargs["progress_cb"](1, 1)
        return "texto transcripto"


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


def make_manager(tmp_path: Path) -> tuple[TranscribeJobManager, Settings]:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    model_dir = settings.models_path / "asr" / MODEL_ID
    model_dir.mkdir(parents=True, exist_ok=True)
    registry.register(
        ModelEntry(
            id=MODEL_ID,
            name="onnx-community/whisper-tiny.en",
            kind=ModelKind.asr_onnx,
            source="hf:onnx-community/whisper-tiny.en",
            size_bytes=1,
            file_path=f"asr/{MODEL_ID}",
        )
    )
    manager = TranscribeJobManager(
        settings, FakeEngine(), DeviceSemaphores(settings), registry=registry
    )
    return manager, settings


def upload(name: str = "charla.wav") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(b"RIFF....WAVE"))


async def create(manager: TranscribeJobManager, settings: Settings, **kwargs):
    params = {
        # current_user_from_request soporta None: es el camino de los tests de ruta.
        "request": None,
        "file": upload(),
        "model_id": MODEL_ID,
        "transcribe_jobs": manager,
        "storage": StorageService(settings),
        "settings": settings,
    }
    params.update(kwargs)
    return await create_transcribe_job(**params)


# ---------------------------------------------------------------------------
# POST /transcribe/jobs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creating_a_job_accepts_the_upload(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings)

    assert response.status is JobStatus.queued
    assert response.status_url.endswith(response.job_id)
    assert manager.get_job(response.job_id) is not None


@pytest.mark.asyncio
async def test_an_unknown_model_is_a_400(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        await create(manager, settings, model_id="no-existe")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_a_rejected_job_leaves_no_upload_behind(tmp_path: Path):
    # El archivo ya se guardo cuando la validacion falla: sin el finally quedaria
    # basura en uploads por cada intento fallido.
    manager, settings = make_manager(tmp_path)
    with pytest.raises(HTTPException):
        await create(manager, settings, model_id="no-existe")

    assert list(settings.uploads_path.glob("*")) == []


@pytest.mark.asyncio
async def test_a_malformed_language_is_a_400(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        await create(manager, settings, language="espanol")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_an_omitted_language_is_treated_as_absent(tmp_path: Path):
    """Los tests de ruta de este repo llaman las corrutinas DIRECTO, sin FastAPI, asi
    que un Form no provisto llega como su sentinel y no como None. Tratar cualquier
    cosa que no sea un str no vacio como ausente cubre los dos caminos."""
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings)

    assert manager.get_job(response.job_id).language is None


@pytest.mark.asyncio
async def test_an_empty_language_string_is_also_absent(tmp_path: Path):
    # Un select vacio en la UI manda "" y eso no es un idioma.
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings, language="")

    assert manager.get_job(response.job_id).language is None


# ---------------------------------------------------------------------------
# GET / cancel / download
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_finished_job_exposes_the_text_and_a_download_url(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings)
    await manager._process_next()

    response = await get_transcribe_job(created.job_id, manager)

    assert response.status is JobStatus.completed
    # El TEXTO viaja en la respuesta: la UI no tiene que descargar un archivo para
    # mostrarlo.
    assert response.text == "texto transcripto"
    assert response.download_url.endswith(f"{created.job_id}/download")


@pytest.mark.asyncio
async def test_a_pending_job_has_no_download_url(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings)

    response = await get_transcribe_job(created.job_id, manager)
    assert response.download_url is None


@pytest.mark.asyncio
async def test_an_unknown_job_is_a_404(tmp_path: Path):
    manager, _s = make_manager(tmp_path)
    for route in (get_transcribe_job, cancel_transcribe_job, download_transcribe_job):
        with pytest.raises(HTTPException) as exc_info:
            await route("nope", manager)
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_downloading_a_pending_job_is_a_409(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings)

    with pytest.raises(HTTPException) as exc_info:
        await download_transcribe_job(created.job_id, manager)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_the_download_is_named_after_the_original_audio(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings, file=upload("entrevista.mp3"))
    await manager._process_next()

    response = await download_transcribe_job(created.job_id, manager)
    assert response.filename == "entrevista.txt"


@pytest.mark.asyncio
async def test_cancelling_a_queued_job_reports_it_cancelled(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings)

    response = await cancel_transcribe_job(created.job_id, manager)
    assert response.status is JobStatus.cancelled


# ---------------------------------------------------------------------------
# Serializacion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_response_serializes_camel_case(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings)
    await manager._process_next()

    dumped = transcribe_job_to_response(manager.get_job(created.job_id)).model_dump(
        by_alias=True
    )
    for key in ("originalFilename", "modelId", "createdAt", "progressPct", "downloadUrl"):
        assert key in dumped
    assert "text" in dumped


# ---------------------------------------------------------------------------
# A traves de la app real
# ---------------------------------------------------------------------------


def test_the_transcribe_routes_are_registered_with_their_permissions():
    from app.main import app
    from app.services.auth.permissions import Permission

    expected = {
        ("/api/v1/transcribe/jobs", "POST"): Permission.jobs_create,
        ("/api/v1/transcribe/jobs/{job_id}", "GET"): Permission.jobs_read_own,
        ("/api/v1/transcribe/jobs/{job_id}/cancel", "POST"): Permission.jobs_cancel_own,
        ("/api/v1/transcribe/jobs/{job_id}/download", "GET"): Permission.jobs_read_own,
        ("/api/v1/asr/models/install", "POST"): Permission.models_install,
    }

    for (path, method), permission in expected.items():
        found = [
            route
            for route in app.routes
            if getattr(route, "path", None) == path
            and method in getattr(route, "methods", set())
        ]
        assert found, f"ruta no registrada: {method} {path}"
        permissions = {
            cell.cell_contents
            for dependency in found[0].dependant.dependencies
            for cell in getattr(dependency.call, "__closure__", None) or ()
            if isinstance(cell.cell_contents, Permission)
        }
        assert permission in permissions, f"{method} {path} no exige {permission}"


def test_the_asr_search_route_is_registered():
    from app.main import app

    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/api/v1/asr/models/search" in paths
    assert "/api/v1/asr/models/install/{install_id}" in paths
