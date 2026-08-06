from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.api.routes import (
    cancel_shape3d_job,
    create_shape3d_job,
    download_shape3d_job,
    get_shape3d_job,
    shape3d_job_to_response,
)
from app.config import Settings
from app.models import JobStatus
from app.schemas import Shape3dJobRequest
from app.services.shape3d_job_manager import Shape3dJobManager
from app.services.storage import StorageService


def tetrahedron() -> np.ndarray:
    a, b, c, d = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)
    return np.array([[a, c, b], [a, b, d], [a, d, c], [b, c, d]], dtype=np.float64)


class FakeEngine:
    def available(self) -> bool:
        return True

    def generate_from_text(self, prompt: str, **_kwargs) -> np.ndarray:
        return tetrahedron()


class MissingEngine:
    def available(self) -> bool:
        return False


def make_manager(tmp_path: Path, engine=None) -> Shape3dJobManager:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return Shape3dJobManager(settings, engine or FakeEngine())


class RequestSinUsuario:
    """current_user_from_request soporta None: es el camino de los tests de ruta."""


@pytest.mark.asyncio
async def test_creating_a_job_returns_it_queued(tmp_path: Path):
    manager = make_manager(tmp_path)

    respuesta = await create_shape3d_job(
        payload=Shape3dJobRequest(prompt="un soporte"), request=None, jobs=manager
    )

    assert respuesta.status is JobStatus.queued
    assert respuesta.download_url is None


@pytest.mark.asyncio
async def test_a_finished_job_carries_the_verdict_and_the_file(tmp_path: Path):
    manager = make_manager(tmp_path)
    creado = await create_shape3d_job(
        payload=Shape3dJobRequest(prompt="un soporte"), request=None, jobs=manager
    )
    await manager._process_next()

    respuesta = await get_shape3d_job(job_id=creado.id, jobs=manager)

    assert respuesta.status is JobStatus.completed
    assert respuesta.can_print is True
    assert respuesta.download_url
    archivo = await download_shape3d_job(job_id=creado.id, jobs=manager)
    assert Path(archivo.path).exists()


@pytest.mark.asyncio
async def test_without_the_model_it_is_a_409_with_the_command(tmp_path: Path):
    manager = make_manager(tmp_path, MissingEngine())

    with pytest.raises(HTTPException) as exc_info:
        await create_shape3d_job(
            payload=Shape3dJobRequest(prompt="algo"), request=None, jobs=manager
        )

    assert exc_info.value.status_code == 409
    assert "download-shap-e" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_an_empty_prompt_is_a_400(tmp_path: Path):
    manager = make_manager(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await create_shape3d_job(
            payload=Shape3dJobRequest(prompt="  "), request=None, jobs=manager
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_an_unknown_job_is_a_404(tmp_path: Path):
    manager = make_manager(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await get_shape3d_job(job_id="no-existe", jobs=manager)

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_cancelling_reports_it_cancelled(tmp_path: Path):
    manager = make_manager(tmp_path)
    creado = await create_shape3d_job(
        payload=Shape3dJobRequest(prompt="algo"), request=None, jobs=manager
    )

    respuesta = await cancel_shape3d_job(job_id=creado.id, jobs=manager)

    assert respuesta.status is JobStatus.cancelled


@pytest.mark.asyncio
async def test_the_download_url_only_appears_when_the_file_exists(tmp_path: Path):
    # Ofrecer una descarga que no esta es peor que no ofrecerla.
    manager = make_manager(tmp_path)
    job = await manager.create_job(prompt="algo")

    assert shape3d_job_to_response(job).download_url is None
