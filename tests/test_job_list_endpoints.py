"""Listado por familia de las tres familias que no lo tenian.

Transcripcion, descarga y 3D corrian en el servidor pero no se podian enumerar:
recargar el navegador perdia el trabajo para siempre. Estos tests fijan el
contrato que ya cumplian las otras cuatro familias — solo los propios por
defecto, `?all=true` solo con permiso — porque es lo que vuelve seguro exponer
la lista en multiusuario.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth_deps import get_current_user, get_user_store
from app.api.routes import router
from app.config import Settings
from app.models import DownloadJob, JobStatus, Shape3dJob, TranscribeJob
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import ROLE_PERMISSIONS, Role
from app.services.device_semaphores import DeviceSemaphores
from app.services.download_job_manager import DownloadJobManager
from app.services.model_registry import ModelRegistry
from app.services.shape3d_job_manager import Shape3dJobManager
from app.services.storage import StorageService
from app.services.transcribe_job_manager import TranscribeJobManager


def make_user(user_id: str, role: Role = Role.user) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id,
        username=user_id,
        role=role,
        permissions=ROLE_PERMISSIONS[role],
        must_change_password=False,
        quota_overrides={},
    )


class Sesion:
    """Quien esta pidiendo. Se cambia en el medio del test para probar que un
    dueño no ve lo del otro sin tener que levantar el login real."""

    def __init__(self) -> None:
        self.user = make_user("u1")


@pytest.fixture
def api(tmp_path: Path) -> tuple[TestClient, FastAPI, Sesion]:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    app = FastAPI()
    app.include_router(router)
    sesion = Sesion()
    # require() resuelve get_user_store aunque esta app minima no lo tenga, y
    # get_current_user se pisa para elegir el usuario sin pasar por el login.
    app.dependency_overrides[get_user_store] = lambda: None
    app.dependency_overrides[get_current_user] = lambda: sesion.user
    app.state.settings = settings
    app.state.transcribe_jobs = TranscribeJobManager(
        settings, None, DeviceSemaphores(settings), registry=ModelRegistry(settings)
    )
    app.state.download_jobs = DownloadJobManager(settings)
    app.state.shape3d_jobs = Shape3dJobManager(settings, None)
    return TestClient(app), app, sesion


def seed_transcribe(app: FastAPI, job_id: str, owner_id: str | None, status: JobStatus) -> None:
    app.state.transcribe_jobs.jobs[job_id] = TranscribeJob(
        source_path=Path("charla.wav"),
        original_filename="charla.wav",
        model_id="asr--whisper-tiny",
        id=job_id,
        status=status,
        owner_id=owner_id,
    )


def seed_download(app: FastAPI, job_id: str, owner_id: str | None, status: JobStatus) -> None:
    app.state.download_jobs.jobs[job_id] = DownloadJob(
        url="https://example.com/v", id=job_id, status=status, owner_id=owner_id
    )


def seed_shape3d(app: FastAPI, job_id: str, owner_id: str | None, status: JobStatus) -> None:
    app.state.shape3d_jobs.jobs[job_id] = Shape3dJob(
        prompt="una traba", id=job_id, status=status, owner_id=owner_id
    )


LISTINGS = [
    pytest.param("/api/v1/transcribe/jobs", seed_transcribe, id="transcribe"),
    pytest.param("/api/v1/download/jobs", seed_download, id="download"),
    pytest.param("/api/v1/print/generate", seed_shape3d, id="shape3d"),
]


def ids_of(response) -> list[str]:
    return [job["id"] for job in response.json()["jobs"]]


@pytest.mark.parametrize("path,seed", LISTINGS)
def test_listing_is_empty_when_there_is_nothing(api, path: str, seed) -> None:
    client, _app, _sesion = api

    response = client.get(path)

    assert response.status_code == 200
    assert response.json() == {"jobs": []}


@pytest.mark.parametrize("path,seed", LISTINGS)
def test_listing_returns_the_jobs_of_whoever_asks(api, path: str, seed) -> None:
    client, app, _sesion = api
    seed(app, "propio", "u1", JobStatus.running)

    response = client.get(path)

    assert response.status_code == 200
    assert ids_of(response) == ["propio"]


@pytest.mark.parametrize("path,seed", LISTINGS)
def test_one_owner_never_sees_the_jobs_of_another(api, path: str, seed) -> None:
    client, app, sesion = api
    seed(app, "de-u1", "u1", JobStatus.running)
    seed(app, "de-u2", "u2", JobStatus.running)

    sesion.user = make_user("u2")
    response = client.get(path)

    assert ids_of(response) == ["de-u2"]


@pytest.mark.parametrize("path,seed", LISTINGS)
def test_asking_for_everyone_without_permission_is_403(api, path: str, seed) -> None:
    client, app, _sesion = api
    seed(app, "de-u1", "u1", JobStatus.running)

    response = client.get(path, params={"all": "true"})

    assert response.status_code == 403


@pytest.mark.parametrize("path,seed", LISTINGS)
def test_an_admin_asking_for_everyone_sees_every_owner(api, path: str, seed) -> None:
    client, app, sesion = api
    seed(app, "de-u1", "u1", JobStatus.running)
    seed(app, "de-u2", "u2", JobStatus.running)

    sesion.user = make_user("jefa", Role.admin)
    response = client.get(path, params={"all": "true"})

    assert response.status_code == 200
    assert sorted(ids_of(response)) == ["de-u1", "de-u2"]
    assert sorted(job["ownerId"] for job in response.json()["jobs"]) == ["u1", "u2"]


@pytest.mark.parametrize("path,seed", LISTINGS)
def test_a_finished_job_still_appears_until_retention_prunes_it(api, path: str, seed) -> None:
    """El listado devuelve lo vivo MAS lo terminado reciente, igual que las otras
    cuatro familias: quien filtra por estado es la UI, y quien poda es el
    RetentionSweeper. Un listado que escondiera lo completado dejaria sin
    descarga a quien recargo justo al terminar."""
    client, app, _sesion = api
    seed(app, "termino", "u1", JobStatus.completed)
    seed(app, "corriendo", "u1", JobStatus.running)

    response = client.get(path)

    assert sorted(ids_of(response)) == ["corriendo", "termino"]
