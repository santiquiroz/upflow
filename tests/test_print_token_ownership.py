"""Los STL por token opaco solo le sirven a quien los creo.

`/print/repaired/{token}` y `/print/parts/{token}` no se sirven por job sino por
un token de 32 hex, asi que la unica defensa posible es el registro de dueño.
Cuando la auditoria lo marco, ese registro no existia: cualquiera con el token
bajaba el archivo. Existe desde entonces, pero ningun test lo cubria — y un
control de acceso sin prueba es una afirmacion, no un hecho.

Todo lo denegado responde 404, no 403: "no existe" y "no es tuyo" tienen que ser
indistinguibles, porque un 403 confirmaria que el token es valido.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.api.auth_deps import get_current_user, get_user_store
from app.api.routes import MAX_PRINT_TOKENS, router
from app.config import Settings, get_settings
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import ROLE_PERMISSIONS, Role
from app.services.storage import StorageService
from app.services.stl_writer import write_stl


def make_user(user_id: str, role: Role = Role.user) -> AuthenticatedUser:
    return AuthenticatedUser(
        id=user_id,
        username=user_id,
        role=role,
        permissions=ROLE_PERMISSIONS[role],
        must_change_password=False,
        quota_overrides={},
    )


class Sesion:
    def __init__(self) -> None:
        self.user = make_user("u1")


@pytest.fixture
def api(tmp_path: Path) -> tuple[TestClient, FastAPI, Sesion, Settings]:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    app = FastAPI()
    app.include_router(router)
    sesion = Sesion()

    # El dueño sale de request.state.current_user, que en produccion pone el
    # middleware de auth. Sin este middleware de mentira todos serian el mismo
    # usuario anonimo y el test no probaria el aislamiento que dice probar.
    @app.middleware("http")
    async def _quien_pide(request: Request, call_next):
        request.state.current_user = sesion.user
        return await call_next(request)

    app.dependency_overrides[get_user_store] = lambda: None
    app.dependency_overrides[get_current_user] = lambda: sesion.user
    # get_settings() construye un Settings NUEVO desde el entorno: sin pisarlo,
    # los endpoints buscarian los STL en la carpeta real del usuario, no en tmp.
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.settings = settings
    return TestClient(app), app, sesion, settings


def seed_token(app: FastAPI, settings: Settings, token: str, owner_id: str, suffix: str) -> None:
    """Un token ya emitido: registrado a nombre de alguien y con su archivo."""
    owners = getattr(app.state, "print_token_owners", None)
    if owners is None:
        owners = {}
        app.state.print_token_owners = owners
    owners[token] = owner_id
    write_stl(settings.outputs_path / f"{token}.{suffix}.stl", _un_triangulo())


def _un_triangulo() -> np.ndarray:
    return np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]], dtype=np.float32)


ENDPOINTS = [
    pytest.param("/api/v1/print/repaired", "repaired", id="repaired"),
    pytest.param("/api/v1/print/parts", "part", id="parts"),
]


@pytest.mark.parametrize("path,suffix", ENDPOINTS)
def test_el_dueño_baja_lo_suyo(api, path: str, suffix: str) -> None:
    client, app, _sesion, settings = api
    token = "a" * 32
    seed_token(app, settings, token, "u1", suffix)

    response = client.get(f"{path}/{token}")

    assert response.status_code == 200
    assert response.content.startswith(b"solid") or len(response.content) > 0


@pytest.mark.parametrize("path,suffix", ENDPOINTS)
def test_otro_usuario_no_baja_el_token_ajeno(api, path: str, suffix: str) -> None:
    client, app, sesion, settings = api
    token = "b" * 32
    seed_token(app, settings, token, "u1", suffix)

    sesion.user = make_user("u2")
    response = client.get(f"{path}/{token}")

    assert response.status_code == 404


@pytest.mark.parametrize("path,suffix", ENDPOINTS)
def test_un_admin_baja_cualquiera(api, path: str, suffix: str) -> None:
    client, app, sesion, settings = api
    token = "c" * 32
    seed_token(app, settings, token, "u1", suffix)

    sesion.user = make_user("jefa", role=Role.admin)
    response = client.get(f"{path}/{token}")

    assert response.status_code == 200


@pytest.mark.parametrize("path,suffix", ENDPOINTS)
def test_un_token_bien_formado_que_nadie_emitio_no_sirve(api, path: str, suffix: str) -> None:
    client, app, _sesion, settings = api
    # El archivo EXISTE en disco: sin registro de dueño, adivinar el nombre no alcanza.
    write_stl(settings.outputs_path / f"{'d' * 32}.{suffix}.stl", _un_triangulo())

    response = client.get(f"{path}/{'d' * 32}")

    assert response.status_code == 404


@pytest.mark.parametrize("path,suffix", ENDPOINTS)
def test_un_token_que_no_es_hex_no_toca_el_disco(api, path: str, suffix: str) -> None:
    """`g`*32 tiene el largo justo, asi que llega al handler y lo para el regex.

    Con `..%2f..` el 404 podria venir del ruteo sin ejecutar una linea nuestra,
    y un test que pasa sin ejercitar la defensa no prueba que la defensa exista.
    """
    client, app, _sesion, settings = api
    # Registrado y con archivo: lo unico que puede rechazarlo es la forma del token.
    owners = {"g" * 32: "u1"}
    app.state.print_token_owners = owners
    write_stl(settings.outputs_path / f"{'g' * 32}.{suffix}.stl", _un_triangulo())

    assert client.get(f"{path}/{'g' * 32}").status_code == 404
    assert client.get(f"{path}/..%2f..%2fetc%2fpasswd").status_code == 404


def test_el_registro_no_crece_sin_limite(api) -> None:
    """El techo es lo que evita que el registro sea una fuga de memoria.

    Cuesta un token viejo, y eso es aceptable a proposito: el archivo se baja
    apenas se genera. Lo que no es aceptable es un dict que crece para siempre.
    """
    _client, app, _sesion, _settings = api
    from app.api.routes import _register_print_token

    class _Req:
        def __init__(self, app_: FastAPI) -> None:
            self.app = app_
            self.state = type("S", (), {"current_user": make_user("u1")})()

    peticion = _Req(app)
    for n in range(MAX_PRINT_TOKENS + 10):
        _register_print_token(peticion, f"{n:032x}")

    assert len(app.state.print_token_owners) == MAX_PRINT_TOKENS
    assert f"{0:032x}" not in app.state.print_token_owners
    assert f"{MAX_PRINT_TOKENS + 9:032x}" in app.state.print_token_owners
