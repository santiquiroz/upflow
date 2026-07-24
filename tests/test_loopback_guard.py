from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.security import LoopbackGuardMiddleware, is_loopback_host


def test_is_loopback_host_accepts_known_local_hosts() -> None:
    assert is_loopback_host("127.0.0.1") is True
    assert is_loopback_host("::1") is True
    assert is_loopback_host("localhost") is True


def test_is_loopback_host_accepts_testclient_sentinel() -> None:
    # Starlette's TestClient defaults request.client.host to "testclient" when
    # the test doesn't override `client=`. Nearly all of this repo's existing
    # 1207 tests instantiate `TestClient(app)` this way, and AUTH_MODE=off must
    # leave every one of them passing unchanged -- so the guard treats this
    # sentinel as local. No real network peer can ever present this literal
    # Host value.
    assert is_loopback_host("testclient") is True


def test_is_loopback_host_rejects_remote_ip() -> None:
    assert is_loopback_host("203.0.113.5") is False


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_middleware_allows_default_testclient_when_off() -> None:
    app = _make_app()
    app.add_middleware(LoopbackGuardMiddleware, auth_mode="off")

    with TestClient(app) as client:
        response = client.get("/ping")

    assert response.status_code == 200


def test_middleware_allows_loopback_client_when_off() -> None:
    app = _make_app()
    app.add_middleware(LoopbackGuardMiddleware, auth_mode="off")

    with TestClient(app, client=("127.0.0.1", 12345)) as client:
        response = client.get("/ping")

    assert response.status_code == 200


def test_middleware_rejects_remote_client_when_off() -> None:
    app = _make_app()
    app.add_middleware(LoopbackGuardMiddleware, auth_mode="off")

    with TestClient(app, client=("203.0.113.5", 12345)) as client:
        response = client.get("/ping")

    assert response.status_code == 403
    assert "AUTH_MODE=multi" in response.json()["detail"]


def test_middleware_allows_remote_client_when_multi() -> None:
    app = _make_app()
    app.add_middleware(LoopbackGuardMiddleware, auth_mode="multi")

    with TestClient(app, client=("203.0.113.5", 12345)) as client:
        response = client.get("/ping")

    assert response.status_code == 200
