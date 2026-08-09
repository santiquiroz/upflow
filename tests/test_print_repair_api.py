from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.auth_deps import off_mode_user
from app.api.routes import download_repaired_mesh, repair_print_mesh
from app.config import Settings
from app.services.storage import StorageService
from app.services.stl_reader import read_stl
from app.services.stl_writer import write_stl

# ---------------------------------------------------------------------------
# La reparacion devuelve DOS cosas y las dos importan: el archivo reparado y la
# medicion de como quedo. Devolver solo el archivo obligaria al usuario a confiar;
# devolver solo el reporte lo dejaria sin la pieza.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


def fake_request() -> SimpleNamespace:
    # Suficiente para _print_token_owners (app.state) y _owner_id (state).
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()), state=SimpleNamespace())


def tetrahedron() -> np.ndarray:
    a, b, c, d = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 0.0, 10.0)
    return np.array([[a, c, b], [a, b, d], [a, d, c], [b, c, d]], dtype=np.float64)


def upload(tmp_path: Path, triangles: np.ndarray) -> UploadFile:
    destino = write_stl(tmp_path / "fuente.stl", triangles)
    return UploadFile(filename="pieza.stl", file=io.BytesIO(destino.read_bytes()))


@pytest.mark.asyncio
async def test_a_hole_comes_back_closed(tmp_path: Path):
    settings = make_settings(tmp_path)

    respuesta = await repair_print_mesh(
        request=fake_request(),
        file=upload(tmp_path, tetrahedron()[:-1]),
        settings_dep=settings,
        storage=StorageService(settings),
        current_user=off_mode_user(),
    )

    assert respuesta.watertight
    assert respuesta.download_url


@pytest.mark.asyncio
async def test_the_repaired_file_can_be_downloaded_and_is_really_closed(tmp_path: Path):
    settings = make_settings(tmp_path)
    req = fake_request()
    respuesta = await repair_print_mesh(
        request=req,
        file=upload(tmp_path, tetrahedron()[:-1]),
        settings_dep=settings,
        storage=StorageService(settings),
        current_user=off_mode_user(),
    )
    token = respuesta.download_url.rsplit("/", 1)[-1]

    archivo = await download_repaired_mesh(
        token=token, request=req, settings_dep=settings, current_user=off_mode_user()
    )

    from app.services.mesh_inspect import inspect_mesh

    assert inspect_mesh(read_stl(Path(archivo.path))).is_watertight


@pytest.mark.asyncio
async def test_a_mesh_it_cannot_close_says_so_instead_of_claiming_success(tmp_path: Path):
    # Dos laminas separadas no forman un solido. El reporte tiene que decirlo.
    lamina = np.array(
        [
            [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)],
            [(90.0, 0.0, 0.0), (100.0, 0.0, 0.0), (90.0, 10.0, 0.0)],
        ],
        dtype=np.float64,
    )
    settings = make_settings(tmp_path)

    respuesta = await repair_print_mesh(
        request=fake_request(),
        file=upload(tmp_path, lamina),
        settings_dep=settings,
        storage=StorageService(settings),
        current_user=off_mode_user(),
    )

    assert not respuesta.can_print
    assert respuesta.blockers


@pytest.mark.asyncio
async def test_an_unknown_token_is_a_404(tmp_path: Path):
    settings = make_settings(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await download_repaired_mesh(
            token="no-existe",
            request=fake_request(),
            settings_dep=settings,
            current_user=off_mode_user(),
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_a_token_cannot_escape_the_outputs_folder(tmp_path: Path):
    # `..` en el token abriria cualquier archivo del disco.
    settings = make_settings(tmp_path)

    with pytest.raises(HTTPException):
        await download_repaired_mesh(
            token="../../../etc/passwd",
            request=fake_request(),
            settings_dep=settings,
            current_user=off_mode_user(),
        )


@pytest.mark.asyncio
async def test_the_uploaded_original_does_not_stay_on_disk(tmp_path: Path):
    settings = make_settings(tmp_path)

    await repair_print_mesh(
        request=fake_request(),
        file=upload(tmp_path, tetrahedron()[:-1]),
        settings_dep=settings,
        storage=StorageService(settings),
        current_user=off_mode_user(),
    )

    assert list(settings.uploads_path.glob("*.stl")) == []


# ---------------------------------------------------------------------------
# Ownership del token (2026-08-08): el token solo le sirve a quien lo creo.
# Mismo 404 para "no existe" y "no es tuyo", igual que con los jobs.
# ---------------------------------------------------------------------------


def _plain_user(user_id: str):
    admin = off_mode_user()
    return SimpleNamespace(
        id=user_id, username=user_id, role=admin.role,
        permissions=frozenset(), must_change_password=False, quota_overrides={},
    )


@pytest.mark.asyncio
async def test_another_user_cannot_download_someone_elses_token(tmp_path: Path):
    settings = make_settings(tmp_path)
    req = fake_request()
    req.state.current_user = _plain_user("user-a")
    respuesta = await repair_print_mesh(
        request=req,
        file=upload(tmp_path, tetrahedron()[:-1]),
        settings_dep=settings,
        storage=StorageService(settings),
        current_user=req.state.current_user,
    )
    token = respuesta.download_url.rsplit("/", 1)[-1]

    req.state.current_user = _plain_user("user-b")
    with pytest.raises(HTTPException) as exc_info:
        await download_repaired_mesh(
            token=token, request=req, settings_dep=settings,
            current_user=req.state.current_user,
        )
    assert exc_info.value.status_code == 404

    # El dueño si puede.
    req.state.current_user = _plain_user("user-a")
    archivo = await download_repaired_mesh(
        token=token, request=req, settings_dep=settings,
        current_user=req.state.current_user,
    )
    assert Path(archivo.path).exists()


@pytest.mark.asyncio
async def test_admin_can_download_any_token(tmp_path: Path):
    settings = make_settings(tmp_path)
    req = fake_request()
    req.state.current_user = _plain_user("user-a")
    respuesta = await repair_print_mesh(
        request=req,
        file=upload(tmp_path, tetrahedron()[:-1]),
        settings_dep=settings,
        storage=StorageService(settings),
        current_user=req.state.current_user,
    )
    token = respuesta.download_url.rsplit("/", 1)[-1]

    req.state.current_user = off_mode_user()
    archivo = await download_repaired_mesh(
        token=token, request=req, settings_dep=settings, current_user=off_mode_user()
    )
    assert Path(archivo.path).exists()
