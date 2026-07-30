from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from app.config import Settings
from app.models import JobStatus
from app.services.download_job_manager import (
    DownloadJobManager,
    describe_failure,
    validate_url,
)
from fetchflow import engine as fetch_engine


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


def make_manager(tmp_path: Path) -> DownloadJobManager:
    return DownloadJobManager(Settings(RUNTIME_DIR=str(tmp_path), _env_file=None))


# ---------------------------------------------------------------------------
# Validacion de la URL
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", ["https://example.com/v", "http://example.com/v"])
def test_web_urls_are_accepted(url: str):
    validate_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///C:/Windows/System32/config/SAM",
        "data:text/plain;base64,SGVsbG8=",
        "ftp://example.com/x",
        "/etc/passwd",
    ],
)
def test_non_web_schemes_are_refused(url: str):
    """No es purismo: un file:// es leer disco local con la URL de disfraz.

    yt-dlp los soporta de verdad, asi que sin esta guarda un pedido de "descarga" leeria
    archivos del equipo -- la forma exacta de un LFI.
    """
    with pytest.raises(ValueError):
        validate_url(url)


def test_a_url_without_host_is_refused():
    with pytest.raises(ValueError, match="host"):
        validate_url("https:///solo-path")


def test_surrounding_whitespace_does_not_break_a_valid_url():
    # Pegar desde el navegador arrastra espacios; rechazar por eso seria hostil.
    validate_url("  https://example.com/v  ")


# ---------------------------------------------------------------------------
# Creacion del job
# ---------------------------------------------------------------------------


async def test_a_job_starts_queued_with_what_was_asked(tmp_path: Path):
    manager = make_manager(tmp_path)

    job = await manager.create_job(url="https://example.com/v", max_height=720)

    assert job.status == JobStatus.queued
    assert job.max_height == 720
    assert manager.get_job(job.id) is job


async def test_a_bad_height_is_refused_before_queueing(tmp_path: Path):
    manager = make_manager(tmp_path)

    with pytest.raises(ValueError):
        await manager.create_job(url="https://example.com/v", max_height=999)

    assert manager.queue_depth() == 0


async def test_a_playlist_limit_is_only_checked_when_the_playlist_is_wanted(tmp_path: Path):
    manager = make_manager(tmp_path)

    await manager.create_job(url="https://example.com/v", playlist_limit=9999)

    with pytest.raises(ValueError):
        await manager.create_job(
            url="https://example.com/v", include_playlist=True, playlist_limit=9999
        )


# ---------------------------------------------------------------------------
# Cancelacion
# ---------------------------------------------------------------------------


async def test_cancelling_a_queued_job_marks_it_without_touching_the_engine(tmp_path: Path):
    manager = make_manager(tmp_path)
    job = await manager.create_job(url="https://example.com/v")

    assert manager.cancel_job(job.id) is True
    assert job.status == JobStatus.cancelled
    assert job.finished_at is not None


async def test_cancelling_a_running_job_signals_the_event(tmp_path: Path):
    manager = make_manager(tmp_path)
    job = await manager.create_job(url="https://example.com/v")
    job.status = JobStatus.running
    event = threading.Event()
    manager._cancel_events[job.id] = event

    assert manager.cancel_job(job.id) is True
    assert event.is_set(), "sin esto el motor nunca se entera y la descarga sigue"


async def test_cancelling_a_finished_job_does_nothing(tmp_path: Path):
    manager = make_manager(tmp_path)
    job = await manager.create_job(url="https://example.com/v")
    job.status = JobStatus.completed

    assert manager.cancel_job(job.id) is False


def test_cancelling_an_unknown_job_is_false(tmp_path: Path):
    assert make_manager(tmp_path).cancel_job("no-existe") is False


# ---------------------------------------------------------------------------
# Como termina un job
# ---------------------------------------------------------------------------


async def test_a_cancelled_download_is_not_reported_as_a_failure(tmp_path: Path, monkeypatch):
    """Lo pidio el usuario: mostrarlo en rojo seria mentirle sobre lo que paso."""
    manager = make_manager(tmp_path)
    job = await manager.create_job(url="https://example.com/v")

    def raise_cancelled(*args, **kwargs):
        raise fetch_engine.FetchCancelled("cancelado")

    monkeypatch.setattr(manager, "_download_blocking", raise_cancelled)
    await manager._run_job(job)

    assert job.status == JobStatus.cancelled
    assert job.error is None


async def test_a_missing_yt_dlp_says_how_to_fix_it(tmp_path: Path, monkeypatch):
    manager = make_manager(tmp_path)
    job = await manager.create_job(url="https://example.com/v")

    def raise_unavailable(*args, **kwargs):
        raise fetch_engine.FetchUnavailable("yt-dlp no esta instalado. Instalalo con: pip install yt-dlp")

    monkeypatch.setattr(manager, "_download_blocking", raise_unavailable)
    await manager._run_job(job)

    assert job.status == JobStatus.failed
    assert "pip install yt-dlp" in job.error


async def test_a_real_failure_keeps_the_reason_the_site_gave(tmp_path: Path, monkeypatch):
    """Cuando un sitio cambia y la extraccion se rompe, el motivo es lo unico util.

    Es la diferencia entre un descargador que sirve y uno que no.
    """
    manager = make_manager(tmp_path)
    job = await manager.create_job(url="https://example.com/v")

    def raise_extractor_error(*args, **kwargs):
        raise RuntimeError("ERROR: [vimeo] 12345: Failed to fetch OAuth token: HTTP Error 401")

    monkeypatch.setattr(manager, "_download_blocking", raise_extractor_error)
    await manager._run_job(job)

    assert job.status == JobStatus.failed
    assert "401" in job.error
    assert not job.error.startswith("ERROR: "), "el prefijo de yt-dlp es ruido para el usuario"


def test_a_failure_without_a_message_still_says_something():
    # Un mensaje vacio dejaria la UI con un error en blanco.
    assert describe_failure(RuntimeError("")) == "RuntimeError"


async def test_a_successful_job_carries_its_files(tmp_path: Path, monkeypatch):
    manager = make_manager(tmp_path)
    job = await manager.create_job(url="https://example.com/v")

    def succeed(target_job, cancel_event):
        target_job.output_paths = [tmp_path / "video.mp4"]

    monkeypatch.setattr(manager, "_download_blocking", succeed)
    await manager._run_job(job)

    assert job.status == JobStatus.completed
    assert job.output_paths == [tmp_path / "video.mp4"]
    assert job.error is None


async def test_the_cancel_event_is_dropped_when_the_job_ends(tmp_path: Path, monkeypatch):
    # Sin esto el dict crece por cada descarga y nunca se vacia.
    manager = make_manager(tmp_path)
    job = await manager.create_job(url="https://example.com/v")
    monkeypatch.setattr(manager, "_download_blocking", lambda *a, **k: None)

    await manager._run_job(job)

    assert job.id not in manager._cancel_events


# ---------------------------------------------------------------------------
# SSRF: la URL la pide el SERVIDOR, no el navegador
#
# Sin esta guarda, cualquiera que pueda crear un job hace que el servidor pida una URL
# arbitraria. Con el extractor generico de yt-dlp eso alcanza para sondear servicios
# internos, el endpoint de metadata de la nube o la propia app en localhost.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8000/api/v1/settings",
        "http://169.254.169.254/latest/meta-data/",  # metadata de la nube
        "http://10.0.0.5/video.mp4",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
    ],
)
def test_internal_ip_literals_are_refused(url: str):
    # Un host que ya es IP no pasa por DNS: se evalua solo.
    with pytest.raises(ValueError, match="interna"):
        validate_url(url)


@pytest.mark.parametrize("host", ["localhost", "localhost.", "interno.corp"])
def test_a_hostname_that_resolves_inward_is_refused(host: str, monkeypatch):
    """El chequeo es por direccion RESUELTA y no por nombre.

    Un "localhost." con punto final resuelve igual pero no matchea una comparacion de
    texto ingenua, y un nombre interno cualquiera no se parece a nada sospechoso.
    """
    from app.services import download_job_manager as manager_module

    monkeypatch.setattr(
        manager_module.socket,
        "getaddrinfo",
        lambda h, port, **kwargs: [(0, 0, 0, "", ("127.0.0.1", 0))],
    )

    with pytest.raises(ValueError, match="interna"):
        validate_url(f"http://{host}/admin")


def test_every_resolved_address_is_checked_not_just_the_first(monkeypatch):
    """Un DNS con round-robin puede devolver una publica y una privada.

    Quedarse con la primera dejaria pasar la privada la mitad de las veces.
    """
    import socket as socket_module

    from app.services import download_job_manager as manager_module

    def mixed(host, port, **kwargs):
        return [
            (0, 0, 0, "", ("93.184.216.34", 0)),
            (0, 0, 0, "", ("127.0.0.1", 0)),
        ]

    monkeypatch.setattr(manager_module.socket, "getaddrinfo", mixed)

    with pytest.raises(ValueError, match="interna"):
        validate_url("http://round-robin.example.com/")


def test_a_public_address_is_allowed(monkeypatch):
    from app.services import download_job_manager as manager_module

    monkeypatch.setattr(
        manager_module.socket,
        "getaddrinfo",
        lambda host, port, **kwargs: [(0, 0, 0, "", ("93.184.216.34", 0))],
    )

    validate_url("https://example.com/video")


def test_a_host_that_does_not_resolve_is_refused(monkeypatch):
    # Sin resolucion no se puede saber a donde apunta, y asumir que es publico seria
    # justo el agujero que esta guarda cierra.
    import socket as socket_module

    from app.services import download_job_manager as manager_module

    def fail(host, port, **kwargs):
        raise socket_module.gaierror("no such host")

    monkeypatch.setattr(manager_module.socket, "getaddrinfo", fail)

    with pytest.raises(ValueError, match="resolver"):
        validate_url("https://no-existe.invalid/v")
