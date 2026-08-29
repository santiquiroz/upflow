from __future__ import annotations
import asyncio, io, sys
from pathlib import Path
import pytest
from fastapi import HTTPException, UploadFile
from app.api import routes
from app.config import Settings
from app.services.storage import StorageService


class UsuarioLocal:
    id = 'local'
    permissions = frozenset()


class Req:
    def __init__(self, *tokens):
        estado = type("E", (), {"print_token_owners": {t: "local" for t in tokens}})()
        self.app = type("A", (), {"state": estado})()
        self.state = type("R", (), {"current_user": None})()


def make_settings(tmp: Path) -> Settings:
    s = Settings(RUNTIME_DIR=str(tmp), _env_file=None)
    StorageService(s).ensure_directories()
    return s


def upload(nombre="malla.glb", data=b"glTF fake"):
    return UploadFile(filename=nombre, file=io.BytesIO(data))


def test_fit_sin_vistas_filtra_ruta_absoluta(tmp_path):
    s = make_settings(tmp_path)
    token = "a" * 32
    req = Req(token)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(routes.score_mesh_fit(
            token=token, request=req, file=upload(),
            height_meters=1.7, scale_view=None, resolution=512,
            settings_dep=s, storage=StorageService(s), current_user=UsuarioLocal(),
        ))
    print("STATUS:", ei.value.status_code)
    print("DETAIL:", repr(ei.value.detail))
    assert str(tmp_path) not in str(ei.value.detail), "FILTRA RUTA ABSOLUTA"


def test_generate_motor_no_listo_filtra_ruta_absoluta(tmp_path):
    s = make_settings(tmp_path)
    req = Req()
    with pytest.raises(HTTPException) as ei:
        asyncio.run(routes.generate_mesh_route(
            request=req, file=upload("img.png", b"\x89PNG\r\n\x1a\n"),
            engine="triposg", steps=25, guidance=7.0,
            settings_dep=s, storage=StorageService(s), current_user=UsuarioLocal(),
        ))
    print("STATUS:", ei.value.status_code)
    print("DETAIL:", repr(ei.value.detail))
    assert str(tmp_path) not in str(ei.value.detail), "FILTRA RUTA ABSOLUTA"
