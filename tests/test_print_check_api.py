from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import check_print, list_printers
from app.config import Settings
from app.services.storage import StorageService
from app.services.stl_writer import write_stl

# ---------------------------------------------------------------------------
# El veredicto de impresion tiene que ser PEDIBLE desde afuera. Un servicio que
# nadie puede invocar es codigo muerto, y en este repo ya paso: el modo de salida
# de la transcripcion existia en el manager y ninguna ruta lo aceptaba.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


def closed_box(size: float = 20.0) -> np.ndarray:
    e = np.array(
        [
            [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
            [0, 0, size], [size, 0, size], [size, size, size], [0, size, size],
        ],
        dtype=np.float64,
    )
    caras = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3),
    ]
    return np.array([[e[i] for i in c] for c in caras], dtype=np.float64)


def upload(tmp_path: Path, triangles: np.ndarray, name: str = "pieza.stl") -> UploadFile:
    destino = write_stl(tmp_path / "fuente.stl", triangles)
    return UploadFile(filename=name, file=io.BytesIO(destino.read_bytes()))


@pytest.mark.asyncio
async def test_a_printable_box_comes_back_as_printable(tmp_path: Path):
    settings = make_settings(tmp_path)

    respuesta = await check_print(
        file=upload(tmp_path, closed_box()),
        printer="ender-3",
        settings_dep=settings,
        storage=StorageService(settings),
    )

    assert respuesta.can_print
    assert respuesta.blockers == []
    assert respuesta.size_mm[0] == pytest.approx(20.0, abs=0.01)


@pytest.mark.asyncio
async def test_an_open_mesh_comes_back_with_the_reason(tmp_path: Path):
    settings = make_settings(tmp_path)

    respuesta = await check_print(
        file=upload(tmp_path, closed_box()[:-2]),
        printer="ender-3",
        settings_dep=settings,
        storage=StorageService(settings),
    )

    assert not respuesta.can_print
    assert respuesta.blockers


@pytest.mark.asyncio
async def test_the_target_size_reaches_the_check(tmp_path: Path):
    settings = make_settings(tmp_path)

    respuesta = await check_print(
        file=upload(tmp_path, closed_box()),
        printer="ender-3",
        target_axis="x",
        target_mm=150.0,
        settings_dep=settings,
        storage=StorageService(settings),
    )

    assert respuesta.size_mm[0] == pytest.approx(150.0, abs=0.05)


@pytest.mark.asyncio
async def test_an_unknown_printer_is_a_400(tmp_path: Path):
    settings = make_settings(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await check_print(
            file=upload(tmp_path, closed_box()),
            printer="no-existe",
            settings_dep=settings,
            storage=StorageService(settings),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_a_file_that_is_not_an_stl_is_a_400(tmp_path: Path):
    settings = make_settings(tmp_path)
    basura = UploadFile(filename="foto.stl", file=io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\0" * 200))

    with pytest.raises(HTTPException) as exc_info:
        await check_print(
            file=basura,
            printer="ender-3",
            settings_dep=settings,
            storage=StorageService(settings),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_the_uploaded_file_does_not_stay_on_disk(tmp_path: Path):
    # Un STL de 50 MB por consulta llenaria el disco en una tarde.
    settings = make_settings(tmp_path)

    await check_print(
        file=upload(tmp_path, closed_box()),
        printer="ender-3",
        settings_dep=settings,
        storage=StorageService(settings),
    )

    assert list(settings.uploads_path.glob("*.stl")) == []


@pytest.mark.asyncio
async def test_the_printer_list_is_available(tmp_path: Path):
    respuesta = await list_printers()

    assert any(p.id == "ender-3" for p in respuesta.printers)
    assert all(p.bed_mm[2] > 0 for p in respuesta.printers)
