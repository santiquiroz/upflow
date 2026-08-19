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


# ---------------------------------------------------------------------------
# Lo que viaja al motor
# ---------------------------------------------------------------------------


async def test_the_fetch_request_carries_every_knob(tmp_path: Path, monkeypatch):
    """El cableado job -> FetchRequest es lo unico que une la API con el motor.

    Sin este test, perder un campo en el pass-through no rompe ninguna suite y el
    knob queda mudo: la UI lo muestra, el backend lo guarda y el motor nunca lo ve.
    """
    from types import SimpleNamespace

    from app.services import download_job_manager as manager_module
    from fetchflow.options import FetchPlan

    manager = make_manager(tmp_path)
    job = await manager.create_job(
        url="https://example.com/v",
        audio_only=True,
        audio_format="flac",
        audio_bitrate_kbps=192,
        subtitle_languages=["es"],
    )

    captured: dict = {}
    monkeypatch.setattr(
        manager_module.fetch_engine,
        "probe",
        lambda url: SimpleNamespace(title="t", uploader="u", extractor="e"),
    )
    def capture(request, ffmpeg_dir, **extra):
        # **extra y no la firma exacta: lo que este test afirma es el CONTENIDO
        # del pedido, y clavarle la firma lo hace fallar cada vez que el plan
        # aprende una opcion nueva sin que el pedido cambie en nada.
        captured["request"] = request
        return FetchPlan()

    monkeypatch.setattr(manager_module, "build_plan", capture)
    monkeypatch.setattr(
        manager_module.fetch_engine,
        "download",
        lambda plan, url, on_progress=None, cancel_event=None: [],
    )

    manager._download_blocking(job, threading.Event())

    request = captured["request"]
    assert request.audio_only is True
    assert request.audio_format == "flac"
    assert request.audio_bitrate_kbps == 192
    assert request.subtitle_languages == ("es",)


async def test_the_container_reaches_the_fetch_request(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from app.services import download_job_manager as manager_module
    from fetchflow.options import FetchPlan

    manager = make_manager(tmp_path)
    job = await manager.create_job(url="https://example.com/v", video_container="mkv")

    captured: dict = {}
    monkeypatch.setattr(
        manager_module.fetch_engine,
        "probe",
        lambda url: SimpleNamespace(title="t", uploader="u", extractor="e"),
    )

    def capture(request, ffmpeg_dir, **extra):
        # **extra y no la firma exacta: lo que este test afirma es el CONTENIDO
        # del pedido, y clavarle la firma lo hace fallar cada vez que el plan
        # aprende una opcion nueva sin que el pedido cambie en nada.
        captured["request"] = request
        return FetchPlan()

    monkeypatch.setattr(manager_module, "build_plan", capture)
    monkeypatch.setattr(
        manager_module.fetch_engine,
        "download",
        lambda plan, url, on_progress=None, cancel_event=None: [],
    )

    manager._download_blocking(job, threading.Event())

    assert captured["request"].video_container == "mkv"


# ---------------------------------------------------------------------------
# Encadenado: bajar y seguir trabajando sobre lo bajado
# ---------------------------------------------------------------------------


def manager_con_encadenado(tmp_path: Path, resultado):
    llamadas: list[tuple] = []

    async def follow_up(job, owner):
        llamadas.append((job.id, owner))
        if isinstance(resultado, Exception):
            raise resultado
        return resultado

    manager = DownloadJobManager(
        Settings(RUNTIME_DIR=str(tmp_path), _env_file=None), follow_up=follow_up
    )
    return manager, llamadas


async def test_sin_pedirlo_no_se_encadena_nada(tmp_path: Path, monkeypatch):
    manager, llamadas = manager_con_encadenado(tmp_path, ["no-deberia"])
    job = await manager.create_job(url="https://example.com/v")
    monkeypatch.setattr(manager, "_download_blocking", lambda *a, **k: None)

    await manager._run_job(job)

    assert llamadas == []
    assert job.followup_job_ids == []


async def test_pedirlo_encadena_y_deja_los_ids_para_seguirlos(tmp_path: Path, monkeypatch):
    manager, llamadas = manager_con_encadenado(tmp_path, ["a1", "a2"])
    job = await manager.create_job(url="https://example.com/v", then_separate=True)
    monkeypatch.setattr(manager, "_download_blocking", lambda *a, **k: None)

    await manager._run_job(job)

    assert len(llamadas) == 1
    # Sin los ids la descarga termina y las pistas salen sin que nada las anuncie.
    assert job.followup_job_ids == ["a1", "a2"]


async def test_un_encadenado_fallido_no_vuelve_roja_la_descarga(tmp_path: Path, monkeypatch):
    manager, _ = manager_con_encadenado(tmp_path, RuntimeError("modelo no instalado"))
    job = await manager.create_job(url="https://example.com/v", then_separate=True)
    monkeypatch.setattr(manager, "_download_blocking", lambda *a, **k: None)

    await manager._run_job(job)

    # El archivo esta en disco: llamar fallida a la descarga seria mentir sobre
    # lo unico que si paso.
    assert job.status == JobStatus.completed
    assert job.error is None
    assert job.followup_error == "modelo no instalado"


async def test_una_descarga_fallida_no_encadena(tmp_path: Path, monkeypatch):
    manager, llamadas = manager_con_encadenado(tmp_path, ["no-deberia"])
    job = await manager.create_job(url="https://example.com/v", then_separate=True)

    def explota(*args, **kwargs):
        raise RuntimeError("404")

    monkeypatch.setattr(manager, "_download_blocking", explota)
    await manager._run_job(job)

    # Separar un archivo que no existe no falla distinto: falla peor, tarde y
    # con un mensaje sobre audio.
    assert job.status == JobStatus.failed
    assert llamadas == []


async def test_el_encadenado_corre_a_nombre_de_quien_pidio_la_descarga(tmp_path: Path, monkeypatch):
    manager, llamadas = manager_con_encadenado(tmp_path, [])
    usuario = type("U", (), {"id": "u-7"})()
    job = await manager.create_job(
        url="https://example.com/v", then_separate=True, owner=usuario
    )
    monkeypatch.setattr(manager, "_download_blocking", lambda *a, **k: None)

    await manager._run_job(job)

    # Sin esto la separacion —la parte cara— no le cuenta a nadie en su cuota.
    assert llamadas[0][1] is usuario
    # Y el registro no se queda con el usuario despues de terminar.
    assert manager._owners == {}


# ---------------------------------------------------------------------------
# El PO Token que YouTube exige
# ---------------------------------------------------------------------------


def test_sin_acunador_instalado_no_hay_token(tmp_path: Path):
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    settings.__dict__["deno_binary"] = str(tmp_path / "no-esta.exe")

    assert settings.po_token_command() == []
    # Vacio y no una ruta a algo inexistente: el que llama decide sin tener que
    # chequear el disco de nuevo.
    assert settings.yt_dlp_js_runtimes() == {}


def test_hacen_falta_las_dos_piezas(tmp_path: Path):
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    deno = tmp_path / "deno.exe"
    deno.write_bytes(b"x")
    settings.__dict__["deno_binary"] = str(deno)
    settings.__dict__["ceca_entrypoint"] = str(tmp_path / "falta.ts")

    # Con el motor JS pero sin acuñador no se puede acuñar nada. Decir que esta
    # disponible mandaria a diagnosticar la descarga en vez de la instalacion.
    assert not settings.po_token_available()
    assert settings.po_token_command() == []


def test_el_acunador_corre_con_permisos_acotados(tmp_path: Path):
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    for nombre, attr in (("deno.exe", "deno_binary"), ("main.ts", "ceca_entrypoint")):
        ruta = tmp_path / nombre
        ruta.write_bytes(b"x")
        settings.__dict__[attr] = str(ruta)

    command = settings.po_token_command()

    # Este proceso ejecuta JavaScript que baja de YouTube: puede hablar por red,
    # pero no se le da permiso de escritura.
    assert "--allow-net" in command
    assert not any(arg.startswith("--allow-write") for arg in command)
    assert "--allow-all" not in command


async def test_un_acunador_roto_no_rompe_las_descargas(tmp_path: Path, monkeypatch):
    manager = make_manager(tmp_path)

    class AcunadorRoto:
        def get(self):
            from fetchflow.potoken import PoTokenUnavailable

            raise PoTokenUnavailable("se cayo")

    manager._po_tokens = AcunadorRoto()

    # Sin token se sigue igual: los sitios que no lo piden funcionan, y romper
    # TODAS las descargas por una que no va a andar seria peor. El 403 de
    # YouTube ya se traduce a un motivo legible.
    assert manager._po_token() is None
