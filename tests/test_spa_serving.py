from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router as api_router
from app.main import configure_web_routes


def write_fake_spa_build(dist_dir: Path) -> None:
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>spa-root</body></html>", encoding="utf-8")
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "app.js").write_text("console.log('spa')", encoding="utf-8")


def test_spa_serves_index_html_at_root(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    write_fake_spa_build(dist_dir)
    app = FastAPI()

    configure_web_routes(app, frontend_dist=dist_dir)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "spa-root" in response.text


def test_spa_falls_back_to_index_html_for_unknown_client_routes(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    write_fake_spa_build(dist_dir)
    app = FastAPI()

    configure_web_routes(app, frontend_dist=dist_dir)

    with TestClient(app) as client:
        response = client.get("/models")

    assert response.status_code == 200
    assert "spa-root" in response.text


def test_spa_returns_friendly_error_when_index_html_missing(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir(parents=True)
    app = FastAPI()

    configure_web_routes(app, frontend_dist=dist_dir)

    with TestClient(app) as client:
        root_response = client.get("/")
        fallback_response = client.get("/models")

    assert root_response.status_code == 503
    assert "npm run build" in root_response.text
    assert "Traceback" not in root_response.text
    assert fallback_response.status_code == 503
    assert fallback_response.text == root_response.text


def test_spa_exposes_built_assets_under_assets_path(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    write_fake_spa_build(dist_dir)
    app = FastAPI()

    configure_web_routes(app, frontend_dist=dist_dir)

    with TestClient(app) as client:
        response = client.get("/assets/app.js")

    assert response.status_code == 200
    assert "console.log" in response.text


def test_spa_catch_all_does_not_shadow_api_router(tmp_path: Path) -> None:
    """Mirrors main.py's registration order (api_router before the SPA
    catch-all) and asserts /api/v1/* still resolves to the real API endpoint
    instead of being swallowed by the SPA fallback.
    """
    dist_dir = tmp_path / "dist"
    write_fake_spa_build(dist_dir)
    app = FastAPI()
    app.state.job_manager = SimpleNamespace(queue_depth=lambda: 0)
    app.state.video_job_manager = SimpleNamespace(queue_depth=lambda: 0)
    app.include_router(api_router)

    configure_web_routes(app, frontend_dist=dist_dir)

    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "spa-root" not in response.text


def test_unmatched_api_path_returns_404_not_the_spa(tmp_path: Path) -> None:
    """A nonexistent /api/* path (renamed/removed endpoint, stale frontend)
    must 404 instead of falling through to the SPA catch-all and returning
    index.html — otherwise the client gets HTML where it expects JSON.
    """
    dist_dir = tmp_path / "dist"
    write_fake_spa_build(dist_dir)
    app = FastAPI()
    app.state.job_manager = SimpleNamespace(queue_depth=lambda: 0)
    app.state.video_job_manager = SimpleNamespace(queue_depth=lambda: 0)
    app.include_router(api_router)

    configure_web_routes(app, frontend_dist=dist_dir)

    with TestClient(app) as client:
        response = client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    assert "spa-root" not in response.text
