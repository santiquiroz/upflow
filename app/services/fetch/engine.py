from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.services.fetch.options import FetchPlan

# Capa fina sobre yt_dlp. Todo lo que decide QUE pedir vive en options.py; aca solo se
# ejecuta, se reporta progreso y se cancela.
#
# Se importa yt_dlp adentro de las funciones a proposito: el paquete es pesado y la app
# tiene que arrancar aunque no este instalado, con la capacidad marcada como faltante en
# vez de un ImportError al importar el modulo.


class FetchCancelled(Exception):
    """Lo cancelo el usuario. Se distingue de un fallo para no reportarlo como error."""


class FetchUnavailable(Exception):
    """yt-dlp no esta instalado."""


@dataclass(frozen=True, slots=True)
class FetchProgress:
    downloaded_bytes: int
    total_bytes: int | None
    # Nombre del archivo en curso: en una playlist es lo unico que dice en que va.
    filename: str | None

    @property
    def fraction(self) -> float | None:
        if not self.total_bytes:
            return None
        return min(1.0, self.downloaded_bytes / self.total_bytes)


@dataclass(frozen=True, slots=True)
class MediaInfo:
    title: str
    duration_seconds: int | None
    uploader: str | None
    extractor: str
    is_playlist: bool
    entry_count: int
    available_heights: tuple[int, ...]


def _require_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise FetchUnavailable(
            "yt-dlp no esta instalado. Instalalo con: pip install yt-dlp"
        ) from exc
    return yt_dlp


def _apply_path_entries(entries: tuple[Path, ...]) -> None:
    """Agrega los directorios al PATH del proceso.

    NO es redundante con `ffmpeg_location`. Medido: pasar solo la opcion falla con
    "ffmpeg is not installed" mientras el postprocesador reporta available=True, porque
    el camino de descarga parcial usa otro chequeo que mira el PATH.
    """
    if not entries:
        return
    current = os.environ.get("PATH", "")
    missing = [str(p) for p in entries if str(p) not in current]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, current])


def _heights_from_formats(formats: list[dict]) -> tuple[int, ...]:
    """Alturas de los formatos de VIDEO reales.

    El filtro por vcodec no es cosmetico: sin el aparecen los storyboards (27, 45, 90),
    que son tiras de miniaturas y no calidades elegibles. Ofrecerlas en la UI seria
    ofrecer basura.
    """
    return tuple(
        sorted(
            {
                f["height"]
                for f in formats
                if f.get("height") and f.get("vcodec") not in (None, "none")
            }
        )
    )


def _produced_files(info: dict[str, Any]) -> list[Path]:
    """Los archivos FINALES, leidos del info que devuelve yt-dlp.

    No se pueden tomar de los eventos `finished` del hook: esos reportan los componentes
    de ANTES del merge (el video suelto y el audio suelto), que ffmpeg borra al unirlos.
    Un smoke test real con merge devolvia lista vacia con la descarga perfecta.
    """
    entries = info.get("entries")
    sources = [e for e in entries if e] if entries is not None else [info]

    paths: list[Path] = []
    for entry in sources:
        for download in entry.get("requested_downloads") or []:
            filepath = download.get("filepath")
            if filepath:
                paths.append(Path(filepath))
    return paths


def probe(url: str) -> MediaInfo:
    """Que hay en esta URL, sin descargar nada.

    Es lo que permite mostrar titulo, duracion y calidades ANTES de comprometerse, y
    tambien lo que hace visible que una URL es una playlist de 200 items.
    """
    yt_dlp = _require_yt_dlp()
    options = {"quiet": True, "no_warnings": True, "skip_download": True, "noprogress": True}
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = info.get("entries")
    if entries is not None:
        entries = [e for e in entries if e]
        first = entries[0] if entries else {}
        return MediaInfo(
            title=info.get("title") or first.get("title") or url,
            duration_seconds=None,
            uploader=info.get("uploader") or first.get("uploader"),
            extractor=info.get("extractor_key") or "unknown",
            is_playlist=True,
            entry_count=len(entries),
            available_heights=_heights_from_formats(first.get("formats") or []),
        )

    return MediaInfo(
        title=info.get("title") or url,
        duration_seconds=info.get("duration"),
        uploader=info.get("uploader"),
        extractor=info.get("extractor_key") or "unknown",
        is_playlist=False,
        entry_count=1,
        available_heights=_heights_from_formats(info.get("formats") or []),
    )


def _progress_bridge(
    on_progress: Callable[[FetchProgress], None] | None,
    cancel_event: threading.Event | None,
) -> Callable[[dict], None]:
    """El hook que yt-dlp llama, convertido en progreso tipado y punto de cancelacion.

    Lanzar desde aca es como se cancela: verificado que corta limpio y deja un .part
    identificable, con la excepcion propagando su causa raiz intacta.
    """

    def hook(status: dict[str, Any]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise FetchCancelled("cancelado")
        if on_progress is None or status.get("status") != "downloading":
            return
        on_progress(
            FetchProgress(
                downloaded_bytes=status.get("downloaded_bytes") or 0,
                total_bytes=status.get("total_bytes") or status.get("total_bytes_estimate"),
                filename=status.get("filename"),
            )
        )

    return hook


def download(
    plan: FetchPlan,
    url: str,
    *,
    on_progress: Callable[[FetchProgress], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[Path]:
    """Ejecuta el plan y devuelve los archivos producidos.

    Bloqueante: el llamador lo corre en un hilo. Devuelve una lista porque una playlist
    produce varios y el resto del pipeline necesita saber cuales.
    """
    yt_dlp = _require_yt_dlp()
    _apply_path_entries(plan.env_path_entries)

    options = {
        **plan.options,
        "progress_hooks": [_progress_bridge(on_progress, cancel_event)],
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        if _caused_by_cancel(exc):
            raise FetchCancelled("cancelado") from exc
        raise

    # Los .part son restos de un intento cortado, no resultados.
    return [p for p in _produced_files(info) if p.exists() and p.suffix != ".part"]


def _caused_by_cancel(exc: BaseException) -> bool:
    """yt-dlp envuelve la excepcion del hook en DownloadError; hay que mirar la cadena.

    Sin esto una cancelacion se reportaria como fallo y el usuario veria un error rojo
    por algo que pidio el mismo.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, FetchCancelled):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False
