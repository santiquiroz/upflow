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


# ---------------------------------------------------------------------------
# Lo que encontro la revision de seguridad sobre este mismo codigo, clavado como
# test para que no vuelva: permisos en las rutas, propiedad del trabajo, y la
# cuota que se descontaba solo al admitir y nunca al consumir.
# ---------------------------------------------------------------------------


class UsuarioFalso:
    def __init__(self, user_id: str) -> None:
        self.id = user_id
        self.username = user_id
        self.role = "user"
        self.permissions = frozenset()


class RequestConUsuario:
    def __init__(self, user) -> None:
        # El atributo se llama `current_user`: asi lo lee `current_user_from_request`.
        self.state = type("S", (), {"current_user": user})()


@pytest.mark.asyncio
async def test_a_job_of_another_user_is_a_404_not_a_403(tmp_path: Path):
    # Un 403 confirmaria que el trabajo existe, que es informacion de otro.
    manager = make_manager(tmp_path)
    job = await manager.create_job(prompt="algo", owner=UsuarioFalso("ana"))

    with pytest.raises(HTTPException) as exc_info:
        await get_shape3d_job(
            job_id=job.id, jobs=manager, request=RequestConUsuario(UsuarioFalso("beto"))
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_the_owner_can_see_their_own_job(tmp_path: Path):
    manager = make_manager(tmp_path)
    ana = UsuarioFalso("ana")
    job = await manager.create_job(prompt="algo", owner=ana)

    respuesta = await get_shape3d_job(
        job_id=job.id, jobs=manager, request=RequestConUsuario(ana)
    )

    assert respuesta.id == job.id


@pytest.mark.asyncio
async def test_downloading_someone_elses_mesh_is_a_404(tmp_path: Path):
    manager = make_manager(tmp_path)
    job = await manager.create_job(prompt="algo", owner=UsuarioFalso("ana"))
    await manager._process_next()

    with pytest.raises(HTTPException) as exc_info:
        await download_shape3d_job(
            job_id=job.id, jobs=manager, request=RequestConUsuario(UsuarioFalso("beto"))
        )

    assert exc_info.value.status_code == 404


class CuotaFalsa:
    def __init__(self) -> None:
        self.admitidos: list = []
        self.consumos: list[tuple] = []

    def check_admission(self, user) -> None:
        self.admitidos.append(user.id)

    def record_usage(self, user_id, gpu_seconds) -> None:
        self.consumos.append((user_id, gpu_seconds))


@pytest.mark.asyncio
async def test_the_quota_records_what_was_consumed_not_only_what_was_admitted(tmp_path: Path):
    # Sin esto, `check_admission` deja pasar el primer trabajo y despues nada se
    # descuenta nunca: la cuota queda decorativa.
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    cuota = CuotaFalsa()
    manager = Shape3dJobManager(settings, FakeEngine(), quota_service=cuota)
    await manager.create_job(prompt="algo", owner=UsuarioFalso("ana"))

    await manager._process_next()

    assert cuota.admitidos == ["ana"]
    assert len(cuota.consumos) == 1
    assert cuota.consumos[0][0] == "ana"


@pytest.mark.asyncio
async def test_a_failed_job_still_costs_quota(tmp_path: Path):
    # El tiempo de maquina se gasto igual: no cobrarlo deja un camino gratis
    # para consumir la maquina pidiendo cosas que fallan.
    class MotorQueFalla:
        def available(self) -> bool:
            return True

        def generate_from_text(self, prompt: str, **_kwargs):
            raise RuntimeError("se rompio")

    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    cuota = CuotaFalsa()
    manager = Shape3dJobManager(settings, MotorQueFalla(), quota_service=cuota)
    await manager.create_job(prompt="algo", owner=UsuarioFalso("ana"))

    await manager._process_next()

    assert len(cuota.consumos) == 1
