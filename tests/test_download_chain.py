"""Pegar un link y recibir las pistas, sin pasar por el modulo de audio a mano.

Lo que estas pruebas fijan no es que "se llame a separar": es el detalle que hace
la diferencia entre encadenar bien y romper la descarga. El trabajo de audio
BORRA su fuente al terminar, asi que el archivo bajado no puede ser su entrada
directa; y un fallo del encadenado no puede volver roja una descarga que dejo el
archivo en disco.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.models import DownloadJob, JobStatus
from app.services.download_chain import (
    link_or_copy,
    stage_for_audio_job,
    start_separation_followups,
)


class AudioJobsFalso:
    def __init__(self, falla: Exception | None = None) -> None:
        self.falla = falla
        self.pedidos: list[dict] = []

    async def create_job(self, **kwargs):
        if self.falla is not None:
            raise self.falla
        self.pedidos.append(kwargs)
        return type("Job", (), {"id": kwargs["job_id"]})()


def descarga(tmp_path: Path, nombres: list[str], **extra) -> DownloadJob:
    job = DownloadJob(url="https://example.com/x", then_separate=True, **extra)
    for nombre in nombres:
        destino = tmp_path / nombre
        destino.write_bytes(b"audio")
        job.output_paths.append(destino)
    return job


def test_borrar_la_entrada_del_trabajo_de_audio_no_borra_lo_bajado(tmp_path: Path) -> None:
    bajado = tmp_path / "cancion.mp3"
    bajado.write_bytes(b"contenido")

    _, staged = stage_for_audio_job(bajado, tmp_path)
    staged.unlink()

    # Este ES el motivo de que exista el staging: el manager de audio borra su
    # fuente al terminar, y sin esto la separacion exitosa se comia la descarga.
    assert bajado.exists()
    assert bajado.read_bytes() == b"contenido"


def test_el_staged_tiene_el_mismo_contenido_y_otro_nombre(tmp_path: Path) -> None:
    bajado = tmp_path / "cancion.mp3"
    bajado.write_bytes(b"contenido")

    token, staged = stage_for_audio_job(bajado, tmp_path)

    assert staged != bajado
    assert staged.name == f"{token}-cancion.mp3"
    assert staged.read_bytes() == b"contenido"


def test_si_el_disco_no_soporta_hardlink_copia(tmp_path: Path, monkeypatch) -> None:
    origen = tmp_path / "a.mp3"
    origen.write_bytes(b"datos")
    monkeypatch.setattr("app.services.download_chain.os.link", _no_soportado)

    link_or_copy(origen, tmp_path / "b.mp3")

    # Cobrar el doble de disco es mejor que no poder encadenar en un FAT32.
    assert (tmp_path / "b.mp3").read_bytes() == b"datos"


def _no_soportado(*_args, **_kwargs):
    raise OSError("cross-device link")


def test_una_playlist_dispara_un_trabajo_por_archivo(tmp_path: Path) -> None:
    job = descarga(tmp_path, ["1.mp3", "2.mp3", "3.mp3"])
    audio = AudioJobsFalso()

    ids = asyncio.run(
        start_separation_followups(job, audio_jobs=audio, uploads_path=tmp_path, owner=None)
    )

    assert len(ids) == 3
    # Cada archivo es una cancion distinta: son trabajos separados, no uno con
    # tres entradas.
    assert [p["original_filename"] for p in audio.pedidos] == ["1.mp3", "2.mp3", "3.mp3"]
    assert all(p["separate"] is True for p in audio.pedidos)


def test_el_modelo_pedido_viaja_al_trabajo_de_audio(tmp_path: Path) -> None:
    job = descarga(tmp_path, ["a.mp3"], then_separation_model="umx_4stem")
    audio = AudioJobsFalso()

    asyncio.run(
        start_separation_followups(job, audio_jobs=audio, uploads_path=tmp_path, owner=None)
    )

    assert audio.pedidos[0]["separation_model"] == "umx_4stem"


def test_el_id_del_trabajo_es_el_token_del_archivo(tmp_path: Path) -> None:
    job = descarga(tmp_path, ["a.mp3"])
    audio = AudioJobsFalso()

    ids = asyncio.run(
        start_separation_followups(job, audio_jobs=audio, uploads_path=tmp_path, owner=None)
    )

    # Igual que en una subida manual: el archivo staged se rastrea hasta el
    # trabajo que lo consume mirando el nombre.
    staged = list(tmp_path.glob(f"{ids[0]}-*"))
    assert len(staged) == 1


def test_un_rechazo_no_deja_el_archivo_staged_tirado(tmp_path: Path) -> None:
    job = descarga(tmp_path, ["a.mp3"])
    audio = AudioJobsFalso(falla=ValueError("cuota agotada"))

    with pytest.raises(ValueError):
        asyncio.run(
            start_separation_followups(job, audio_jobs=audio, uploads_path=tmp_path, owner=None)
        )

    # Nadie mas conoce ese nombre: si no lo limpia el que lo creo, queda para
    # siempre ocupando disco.
    assert [p.name for p in tmp_path.iterdir()] == ["a.mp3"]
