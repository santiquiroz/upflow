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
from app.services.subtitles import TranscriptSegment
from app.services.transcribe_job_manager import TranscribeJobManager

MODEL_ID = "asr--onnx-community--whisper-tiny.en"


class FakeEngine:
    async def run(self, **kwargs) -> list[TranscriptSegment]:
        kwargs["progress_cb"](1, 1)
        # Segmentos con tiempo: es lo que devuelve el motor real desde que los
        # subtitulos existen. Su concatenacion da "texto transcripto".
        return [
            TranscriptSegment(start=0.0, end=1.0, text="texto"),
            TranscriptSegment(start=1.0, end=2.0, text="transcripto"),
        ]


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
async def test_the_chosen_output_reaches_the_job(tmp_path: Path):
    # Sin esto el modo de salida es un campo muerto: el manager lo sabe manejar
    # pero nadie se lo puede pedir desde afuera.
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings, file=upload("clip.mp4"), output_mode="video")

    assert manager.get_job(response.job_id).output_mode == "video"


@pytest.mark.asyncio
async def test_the_output_defaults_to_text(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    response = await create(manager, settings)

    assert manager.get_job(response.job_id).output_mode == "text"


@pytest.mark.asyncio
async def test_an_unknown_output_mode_is_a_400(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    with pytest.raises(HTTPException) as exc_info:
        await create(manager, settings, output_mode="holograma")
    assert exc_info.value.status_code == 400


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
async def test_the_subtitled_video_is_offered_only_when_it_exists(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings)
    await manager._process_next()
    job = manager.get_job(created.job_id)

    assert transcribe_job_to_response(job).video_url is None

    video = settings.outputs_path / f"{job.id}.subtitled.mp4"
    video.write_bytes(b"video con subs")
    job.subtitled_video_path = video

    assert transcribe_job_to_response(job).video_url is not None


@pytest.mark.asyncio
async def test_downloading_the_video_serves_the_subtitled_file(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings, file=upload("clip.mp4"))
    await manager._process_next()
    job = manager.get_job(created.job_id)
    video = settings.outputs_path / f"{job.id}.subtitled.mp4"
    video.write_bytes(b"video con subs")
    job.subtitled_video_path = video

    response = await download_transcribe_job(created.job_id, manager, fmt="video")

    assert Path(response.path) == video
    assert response.filename == "clip.mp4"


@pytest.mark.asyncio
async def test_asking_for_a_video_that_was_never_made_is_a_409(tmp_path: Path):
    # Pedir el video de un job que se hizo en modo texto no es un formato
    # desconocido: es un archivo que no existe.
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings)
    await manager._process_next()

    with pytest.raises(HTTPException) as exc_info:
        await download_transcribe_job(created.job_id, manager, fmt="video")

    assert exc_info.value.status_code == 409


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


# ---------------------------------------------------------------------------
# Pertenencia
#
# Una transcripcion ES el contenido de un audio ajeno, asi que leer la de otro es
# una fuga de datos. Estas rutas salieron SIN el chequeo y lo detecto el review
# automatico de commit; el resto de las rutas de jobs del repo si lo tenian.
# ---------------------------------------------------------------------------


def make_user(user_id: str, role=None):
    from app.services.auth.identity import AuthenticatedUser
    from app.services.auth.permissions import ROLE_PERMISSIONS, Role

    effective = role or Role.user
    return AuthenticatedUser(
        id=user_id,
        username=user_id,
        role=effective,
        permissions=ROLE_PERMISSIONS[effective],
        must_change_password=False,
        quota_overrides={},
    )


class RequestWithUser:
    def __init__(self, user) -> None:
        self.state = type("S", (), {"current_user": user})()


async def create_owned(manager: TranscribeJobManager, settings: Settings, owner_id: str):
    job = await manager.create_job(
        source_path=make_upload_file(settings),
        original_filename="charla.wav",
        model_id=MODEL_ID,
    )
    job.owner_id = owner_id
    return job


def make_upload_file(settings: Settings) -> Path:
    path = settings.uploads_path / "charla.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF....WAVE")
    return path


@pytest.mark.asyncio
async def test_another_user_cannot_read_someone_elses_transcript(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    job = await create_owned(manager, settings, owner_id="alice")

    with pytest.raises(HTTPException) as exc_info:
        await get_transcribe_job(job.id, manager, RequestWithUser(make_user("bob")))

    # 404 y no 403: un 403 confirmaria que el job existe.
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_the_owner_can_read_their_own_transcript(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    job = await create_owned(manager, settings, owner_id="alice")

    response = await get_transcribe_job(job.id, manager, RequestWithUser(make_user("alice")))
    assert response.id == job.id


@pytest.mark.asyncio
async def test_an_admin_can_read_any_transcript(tmp_path: Path):
    from app.services.auth.permissions import Role

    manager, settings = make_manager(tmp_path)
    job = await create_owned(manager, settings, owner_id="alice")

    response = await get_transcribe_job(
        job.id, manager, RequestWithUser(make_user("root", Role.admin))
    )
    assert response.id == job.id


@pytest.mark.asyncio
async def test_another_user_cannot_cancel_someone_elses_job(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    job = await create_owned(manager, settings, owner_id="alice")

    with pytest.raises(HTTPException) as exc_info:
        await cancel_transcribe_job(job.id, manager, RequestWithUser(make_user("bob")))

    assert exc_info.value.status_code == 404
    assert job.status is JobStatus.queued


@pytest.mark.asyncio
async def test_another_user_cannot_download_someone_elses_transcript(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    job = await create_owned(manager, settings, owner_id="alice")
    await manager._process_next()

    with pytest.raises(HTTPException) as exc_info:
        await download_transcribe_job(job.id, manager, RequestWithUser(make_user("bob")))

    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# La rama `current_user is not None` es inalcanzable por HTTP
#
# Un review automatico marco ese guard como fail-open. No lo es, y estos tests lo
# fijan en vez de dejarlo como razonamiento:
#
#   - require(...) es dependencia de ruta, y depende de get_current_user.
#   - get_current_user con auth ACTIVA resuelve la sesion o tira 401 ANTES del
#     handler; con auth apagada devuelve el usuario admin de modo local.
#
# O sea: ninguna request llega al cuerpo del handler con state.current_user vacio.
# El guard existe solo para los tests que llaman la corrutina DIRECTO, que es la
# convencion de este repo y esta documentada en el propio parametro.
# ---------------------------------------------------------------------------


def test_transcribe_endpoints_require_login_in_multi_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from app.api import auth_routes
    from app.config import get_settings
    from app.main import app

    auth_routes._login_attempts.clear()
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("AUTH_MODE", "multi")
    monkeypatch.setenv("AUTH_SECRET", "s" * 32)
    get_settings.cache_clear()
    try:
        with TestClient(app, client=("127.0.0.1", 12345)) as client:
            client.post(
                "/api/v1/auth/setup", json={"username": "admin", "password": "adminpass1"}
            )
            for path in (
                "/api/v1/transcribe/jobs/some-id",
                "/api/v1/transcribe/jobs/some-id/download",
            ):
                assert client.get(path).status_code == 401, path
            assert client.post("/api/v1/transcribe/jobs/some-id/cancel").status_code == 401
    finally:
        get_settings.cache_clear()


def test_off_mode_is_a_single_admin_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con auth apagada el usuario resuelto es admin, asi que ve sus propios jobs.

    Es deliberado y es el caso de uso principal: una app de escritorio de un solo
    usuario. Fallar cerrado ahi la dejaria inutilizable.
    """
    from app.api.auth_deps import off_mode_user
    from app.services.auth.permissions import Permission

    user = off_mode_user()
    assert Permission.jobs_read_all in user.permissions


@pytest.mark.asyncio
async def test_the_download_can_be_asked_for_a_subtitle_file(tmp_path: Path):
    """El mismo job entrega transcripcion o subtitulos: los segmentos ya estan,
    asi que el formato se elige al descargar y no al crear el trabajo."""
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings)
    await manager._process_next()

    response = await download_transcribe_job(created.job_id, manager, request=None, fmt="srt")

    body = response.body.decode("utf-8")
    assert "-->" in body
    assert "texto" in body
    assert response.media_type == "application/x-subrip"
    assert ".srt" in response.headers["content-disposition"]


@pytest.mark.asyncio
async def test_an_unknown_download_format_is_refused(tmp_path: Path):
    manager, settings = make_manager(tmp_path)
    created = await create(manager, settings)
    await manager._process_next()

    with pytest.raises(HTTPException) as excinfo:
        await download_transcribe_job(created.job_id, manager, request=None, fmt="doc")

    assert excinfo.value.status_code == 400
