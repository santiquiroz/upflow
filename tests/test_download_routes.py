from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth_deps import get_user_store
from app.api.routes import router
from app.config import Settings
from app.models import JobStatus
from app.services.download_job_manager import DownloadJobManager


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch):
    """Ningun test toca el DNS de verdad.

    La guarda de SSRF resuelve el host, asi que sin esto la suite dependeria de la red y
    fallaria offline. Los tests que necesitan otra resolucion la pisan encima.
    """
    from app.services import download_job_manager as manager_module

    monkeypatch.setattr(
        manager_module.socket,
        "getaddrinfo",
        lambda host, port, **kwargs: [(0, 0, 0, "", ("93.184.216.34", 0))],
    )


def make_client(tmp_path: Path) -> tuple[TestClient, DownloadJobManager]:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    manager = DownloadJobManager(settings)
    app = FastAPI()
    app.include_router(router)
    # require() resuelve get_user_store aunque la rama AUTH_MODE=off no lo lea; esta app
    # minima no tiene app.state.user_store como la real.
    app.dependency_overrides[get_user_store] = lambda: None
    app.state.download_jobs = manager
    app.state.settings = settings
    return TestClient(app), manager


# ---------------------------------------------------------------------------
# Crear
# ---------------------------------------------------------------------------


def test_a_job_is_created_and_reports_what_was_asked(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/api/v1/download/jobs",
        json={"url": "https://example.com/v", "maxHeight": 720},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == JobStatus.queued.value
    assert body["maxHeight"] == 720
    assert body["url"] == "https://example.com/v"


def test_a_non_web_scheme_is_refused_by_the_route(tmp_path: Path):
    """El rechazo tiene que estar en la API y no solo en la UI.

    La ruta es alcanzable directo, y un file:// seria leer disco local del servidor.
    """
    client, _ = make_client(tmp_path)

    response = client.post(
        "/api/v1/download/jobs", json={"url": "file:///C:/Windows/win.ini"}
    )

    assert response.status_code == 400


def test_an_unsupported_height_is_refused(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.post(
        "/api/v1/download/jobs", json={"url": "https://example.com/v", "maxHeight": 999}
    )

    assert response.status_code == 400


def test_the_defaults_favor_the_cheap_request(tmp_path: Path):
    """Los defaults son la decision de producto mas importante de este modulo.

    4K por defecto, o playlist entera por defecto, es como se llega a esperar horas por
    algo que nadie pidio.
    """
    client, _ = make_client(tmp_path)

    body = client.post("/api/v1/download/jobs", json={"url": "https://example.com/v"}).json()

    assert body["maxHeight"] == 1080
    assert body["audioOnly"] is False


# ---------------------------------------------------------------------------
# Consultar y cancelar
# ---------------------------------------------------------------------------


def test_a_job_can_be_read_back(tmp_path: Path):
    client, _ = make_client(tmp_path)
    job_id = client.post("/api/v1/download/jobs", json={"url": "https://example.com/v"}).json()["id"]

    response = client.get(f"/api/v1/download/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["id"] == job_id


def test_an_unknown_job_is_404_and_not_403(tmp_path: Path):
    # Un 403 confirmaria que existe, y con el la URL que otro decidio bajar.
    client, _ = make_client(tmp_path)

    assert client.get("/api/v1/download/jobs/no-existe").status_code == 404


def test_cancelling_reports_the_new_status(tmp_path: Path):
    client, _ = make_client(tmp_path)
    job_id = client.post("/api/v1/download/jobs", json={"url": "https://example.com/v"}).json()["id"]

    response = client.post(f"/api/v1/download/jobs/{job_id}/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == JobStatus.cancelled.value


def test_cancelling_an_unknown_job_is_404(tmp_path: Path):
    client, _ = make_client(tmp_path)

    assert client.post("/api/v1/download/jobs/no-existe/cancel").status_code == 404


# ---------------------------------------------------------------------------
# Lo que la respuesta NO debe decir
# ---------------------------------------------------------------------------


def test_per_file_entries_are_names_and_never_paths(tmp_path: Path):
    """La lista de archivos lleva NOMBRES, uno por archivo.

    Antes este test afirmaba que la respuesta no contenía ninguna ruta del servidor, y
    lo hacía con `str(tmp_path) not in str(body)`. Esa aserción es CIEGA en Windows:
    `str(dict)` duplica los backslashes, así que la ruta con escape simple nunca es
    subcadena del dict con escape doble. Pasaba por accidente y no habría detectado una
    fuga. Ahora se comprueba campo por campo.
    """
    client, manager = make_client(tmp_path)
    job_id = client.post("/api/v1/download/jobs", json={"url": "https://example.com/v"}).json()["id"]
    manager.get_job(job_id).output_paths = [tmp_path / "sub" / "video.mp4", tmp_path / "sub" / "b.mp4"]

    body = client.get(f"/api/v1/download/jobs/{job_id}").json()

    assert body["outputFiles"] == ["video.mp4", "b.mp4"]
    for name in body["outputFiles"]:
        assert "/" not in name and "\\" not in name


def test_the_output_directory_is_reported_on_purpose(tmp_path: Path):
    """El directorio SÍ viaja, y es una decisión, no un descuido.

    Decir el nombre del archivo sin decir dónde quedó vuelve la descarga inútil: hay que
    salir a buscarlo. Es la carpeta que el usuario ya ve en Ajustes, en su propia
    máquina, así que no revela nada que no controle. Las rutas por archivo siguen
    afuera.
    """
    client, manager = make_client(tmp_path)
    job_id = client.post("/api/v1/download/jobs", json={"url": "https://example.com/v"}).json()["id"]
    manager.get_job(job_id).output_paths = [tmp_path / "sub" / "video.mp4"]

    body = client.get(f"/api/v1/download/jobs/{job_id}").json()

    assert body["outputDirectory"] == str(tmp_path / "sub")


def test_a_job_with_no_files_reports_no_directory(tmp_path: Path):
    # Un job en cola no tiene dónde, y inventar una carpeta sería adivinar.
    client, _ = make_client(tmp_path)
    job_id = client.post("/api/v1/download/jobs", json={"url": "https://example.com/v"}).json()["id"]

    assert client.get(f"/api/v1/download/jobs/{job_id}").json()["outputDirectory"] == ""


# ---------------------------------------------------------------------------
# El probe
# ---------------------------------------------------------------------------


def test_probe_refuses_a_non_web_scheme(tmp_path: Path):
    client, _ = make_client(tmp_path)

    response = client.post("/api/v1/download/probe", json={"url": "file:///etc/passwd"})

    assert response.status_code == 400


def test_probe_reports_the_sites_own_reason_when_extraction_fails(tmp_path: Path, monkeypatch):
    """Cuando un sitio cambia, el motivo es lo unico util que se puede dar."""
    from app.services.fetch import engine as fetch_engine

    def boom(url: str):
        raise RuntimeError("ERROR: [vimeo] 1: Failed to fetch OAuth token: HTTP Error 401")

    monkeypatch.setattr(fetch_engine, "probe", boom)
    client, _ = make_client(tmp_path)

    response = client.post("/api/v1/download/probe", json={"url": "https://vimeo.com/1"})

    assert response.status_code == 422
    assert "401" in response.json()["detail"]
    assert not response.json()["detail"].startswith("ERROR: ")


def test_probe_says_when_yt_dlp_is_missing(tmp_path: Path, monkeypatch):
    from app.services.fetch import engine as fetch_engine

    def unavailable(url: str):
        raise fetch_engine.FetchUnavailable("yt-dlp no esta instalado. Instalalo con: pip install yt-dlp")

    monkeypatch.setattr(fetch_engine, "probe", unavailable)
    client, _ = make_client(tmp_path)

    response = client.post("/api/v1/download/probe", json={"url": "https://example.com/v"})

    assert response.status_code == 503
    assert "pip install yt-dlp" in response.json()["detail"]


def test_probe_surfaces_that_a_url_is_a_playlist_before_anything_downloads(
    tmp_path: Path, monkeypatch
):
    """Es el punto del probe: ver los 200 items ANTES de disparar 200 descargas."""
    from app.services.fetch import engine as fetch_engine

    monkeypatch.setattr(
        fetch_engine,
        "probe",
        lambda url: fetch_engine.MediaInfo(
            title="Una playlist",
            duration_seconds=None,
            uploader="alguien",
            extractor="Youtube",
            is_playlist=True,
            entry_count=200,
            available_heights=(360, 720, 1080),
        ),
    )
    client, _ = make_client(tmp_path)

    body = client.post("/api/v1/download/probe", json={"url": "https://example.com/list"}).json()

    assert body["isPlaylist"] is True
    assert body["entryCount"] == 200
    assert body["availableHeights"] == [360, 720, 1080]
