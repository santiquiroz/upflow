from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from app.services.fetch.engine import (
    FetchCancelled,
    FetchProgress,
    _apply_path_entries,
    _caused_by_cancel,
    _progress_bridge,
)

# Todo lo de aca se prueba SIN red y sin yt-dlp: el puente de progreso, la deteccion de
# cancelacion y el manejo del PATH son funciones puras o casi.


# ---------------------------------------------------------------------------
# El puente de progreso
# ---------------------------------------------------------------------------


def test_downloading_events_become_typed_progress():
    seen: list[FetchProgress] = []
    hook = _progress_bridge(seen.append, None)

    hook({"status": "downloading", "downloaded_bytes": 500, "total_bytes": 1000, "filename": "a.mp4"})

    assert seen[0].downloaded_bytes == 500
    assert seen[0].fraction == 0.5
    assert seen[0].filename == "a.mp4"


def test_non_downloading_events_are_ignored():
    # 'finished' y 'error' llegan por el mismo hook; reportarlos como progreso haria
    # saltar la barra al final y volver.
    seen: list[FetchProgress] = []
    hook = _progress_bridge(seen.append, None)

    hook({"status": "finished", "downloaded_bytes": 1000})

    assert seen == []


def test_an_unknown_total_leaves_the_fraction_undefined():
    """Sin total no hay porcentaje honesto.

    Devolver 0 o 1 seria inventar: algunos sitios no publican el tamaño y la UI tiene
    que poder mostrar una barra indeterminada en vez de una mentira.
    """
    seen: list[FetchProgress] = []
    hook = _progress_bridge(seen.append, None)

    hook({"status": "downloading", "downloaded_bytes": 500})

    assert seen[0].total_bytes is None
    assert seen[0].fraction is None


def test_the_estimated_total_is_used_when_the_exact_one_is_missing():
    seen: list[FetchProgress] = []
    hook = _progress_bridge(seen.append, None)

    hook({"status": "downloading", "downloaded_bytes": 250, "total_bytes_estimate": 1000})

    assert seen[0].fraction == 0.25


def test_the_fraction_never_exceeds_one():
    # El total estimado se queda corto seguido; una barra que pasa del 100% se ve rota.
    seen: list[FetchProgress] = []
    hook = _progress_bridge(seen.append, None)

    hook({"status": "downloading", "downloaded_bytes": 1500, "total_bytes_estimate": 1000})

    assert seen[0].fraction == 1.0


# ---------------------------------------------------------------------------
# Cancelacion
# ---------------------------------------------------------------------------


def test_a_set_cancel_event_stops_the_download_from_the_hook():
    """Asi se cancela: lanzando desde el hook.

    Verificado contra yt-dlp real -- corta limpio, deja un .part identificable y la
    excepcion propaga con su causa raiz intacta.
    """
    event = threading.Event()
    event.set()
    hook = _progress_bridge(None, event)

    with pytest.raises(FetchCancelled):
        hook({"status": "downloading", "downloaded_bytes": 1})


def test_cancellation_is_checked_even_on_events_that_are_not_progress():
    # Si solo se mirara en 'downloading', una descarga entre archivos de una playlist
    # ignoraria el cancelar hasta el proximo byte.
    event = threading.Event()
    event.set()
    hook = _progress_bridge(None, event)

    with pytest.raises(FetchCancelled):
        hook({"status": "finished"})


def test_an_unset_event_lets_the_download_continue():
    hook = _progress_bridge(None, threading.Event())

    hook({"status": "downloading", "downloaded_bytes": 1})  # no lanza


def test_a_cancellation_wrapped_by_yt_dlp_is_still_recognized():
    """yt-dlp envuelve la excepcion del hook en DownloadError.

    Sin recorrer la cadena, una cancelacion se reportaria como fallo y el usuario veria
    un error rojo por algo que pidio el mismo.
    """
    try:
        try:
            raise FetchCancelled("cancelado")
        except FetchCancelled as inner:
            raise RuntimeError("DownloadError simulado") from inner
    except RuntimeError as wrapped:
        assert _caused_by_cancel(wrapped) is True


def test_a_real_failure_is_not_mistaken_for_a_cancellation():
    assert _caused_by_cancel(RuntimeError("HTTP 403")) is False


def test_a_cycle_in_the_exception_chain_does_not_hang():
    # Defensivo: una cadena ciclica colgaria el proceso entero en un while.
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a

    assert _caused_by_cancel(a) is False


# ---------------------------------------------------------------------------
# El PATH, que es el detalle que rompe en produccion
# ---------------------------------------------------------------------------


def test_the_ffmpeg_dir_is_prepended_to_path(monkeypatch):
    """Medido: pasar solo `ffmpeg_location` falla.

    Un intento real murio con "ffmpeg is not installed" mientras el postprocesador
    reportaba available=True -- el camino de descarga parcial mira el PATH.
    """
    monkeypatch.setenv("PATH", "/usr/bin")

    _apply_path_entries((Path("/vendor/ffmpeg/bin"),))

    assert os.environ["PATH"].startswith(str(Path("/vendor/ffmpeg/bin")))
    assert "/usr/bin" in os.environ["PATH"]


def test_an_entry_already_present_is_not_duplicated(monkeypatch):
    entry = str(Path("/vendor/ffmpeg/bin"))
    monkeypatch.setenv("PATH", f"{entry}{os.pathsep}/usr/bin")

    _apply_path_entries((Path("/vendor/ffmpeg/bin"),))

    assert os.environ["PATH"].count(entry) == 1


def test_no_entries_leaves_path_untouched(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")

    _apply_path_entries(())

    assert os.environ["PATH"] == "/usr/bin"


# ---------------------------------------------------------------------------
# Que archivos se devuelven
#
# Un smoke test real contra yt-dlp descargaba perfecto (69 eventos de progreso, fraccion
# 0 -> 1) y devolvia lista VACIA: los eventos `finished` reportan los componentes de
# ANTES del merge (video suelto y audio suelto), que ffmpeg borra al unirlos. La verdad
# esta en `requested_downloads[].filepath` del info que devuelve yt-dlp.
# ---------------------------------------------------------------------------


def test_the_final_merged_file_is_what_gets_returned():
    from app.services.fetch.engine import _produced_files

    info = {"requested_downloads": [{"filepath": "/out/video.mp4"}]}

    assert _produced_files(info) == [Path("/out/video.mp4")]


def test_every_entry_of_a_playlist_contributes_its_file():
    from app.services.fetch.engine import _produced_files

    info = {
        "entries": [
            {"requested_downloads": [{"filepath": "/out/1.mp4"}]},
            {"requested_downloads": [{"filepath": "/out/2.mp4"}]},
        ]
    }

    assert _produced_files(info) == [Path("/out/1.mp4"), Path("/out/2.mp4")]


def test_a_null_playlist_entry_is_skipped():
    # yt-dlp mete None por cada item que no pudo extraer; iterarlo crudo explota.
    from app.services.fetch.engine import _produced_files

    info = {"entries": [None, {"requested_downloads": [{"filepath": "/out/2.mp4"}]}]}

    assert _produced_files(info) == [Path("/out/2.mp4")]


def test_info_without_downloads_yields_nothing_instead_of_failing():
    from app.services.fetch.engine import _produced_files

    assert _produced_files({}) == []
    assert _produced_files({"requested_downloads": []}) == []


# ---------------------------------------------------------------------------
# Alturas ofrecidas
# ---------------------------------------------------------------------------


def test_storyboard_formats_are_not_offered_as_qualities():
    """Un probe real devolvia (27, 45, 90, 144, ...): las tres primeras son storyboards.

    Son tiras de miniaturas, no calidades elegibles. Ofrecerlas seria ofrecer basura.
    """
    from app.services.fetch.engine import _heights_from_formats

    formats = [
        {"height": 27, "vcodec": "none"},
        {"height": 45, "vcodec": "none"},
        {"height": 360, "vcodec": "avc1"},
        {"height": 1080, "vcodec": "vp9"},
        {"height": None, "vcodec": "none", "acodec": "opus"},
    ]

    assert _heights_from_formats(formats) == (360, 1080)
