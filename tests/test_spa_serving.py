from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router as api_router
from app.config import Settings
from app.main import configure_web_routes


def build_settings(**overrides) -> Settings:
    return Settings(RUNTIME_DIR="runtime", **overrides)


def write_fake_spa_build(dist_dir: Path) -> None:
    dist_dir.mkdir(parents=True)
    (dist_dir / "index.html").write_text("<html><body>spa-root</body></html>", encoding="utf-8")
    assets_dir = dist_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "app.js").write_text("console.log('spa')", encoding="utf-8")


def test_serve_spa_disabled_falls_back_to_jinja_web_router(tmp_path: Path) -> None:
    app = FastAPI()
    settings = build_settings(SERVE_SPA=False)

    configure_web_routes(app, settings, frontend_dist=tmp_path / "dist")

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "spa-root" not in response.text


def test_serve_spa_enabled_without_a_build_falls_back_to_jinja(tmp_path: Path) -> None:
    app = FastAPI()
    settings = build_settings(SERVE_SPA=True)

    configure_web_routes(app, settings, frontend_dist=tmp_path / "dist")

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "spa-root" not in response.text


def test_serve_spa_enabled_with_a_build_serves_index_html_at_root(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    write_fake_spa_build(dist_dir)
    app = FastAPI()
    settings = build_settings(SERVE_SPA=True)

    configure_web_routes(app, settings, frontend_dist=dist_dir)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "spa-root" in response.text


def test_serve_spa_falls_back_to_index_html_for_unknown_client_routes(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    write_fake_spa_build(dist_dir)
    app = FastAPI()
    settings = build_settings(SERVE_SPA=True)

    configure_web_routes(app, settings, frontend_dist=dist_dir)

    with TestClient(app) as client:
        response = client.get("/models")

    assert response.status_code == 200
    assert "spa-root" in response.text


def test_serve_spa_exposes_built_assets_under_assets_path(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    write_fake_spa_build(dist_dir)
    app = FastAPI()
    settings = build_settings(SERVE_SPA=True)

    configure_web_routes(app, settings, frontend_dist=dist_dir)

    with TestClient(app) as client:
        response = client.get("/assets/app.js")

    assert response.status_code == 200
    assert "console.log" in response.text


def test_spa_catch_all_does_not_shadow_api_router(tmp_path: Path) -> None:
    """Mirrors main.py's registration order (api_router before the SPA
    catch-all) and asserts /api/v1/* still resolves to the real API endpoint
    with SERVE_SPA on, instead of being swallowed by the SPA fallback.
    """
    dist_dir = tmp_path / "dist"
    write_fake_spa_build(dist_dir)
    app = FastAPI()
    app.state.job_manager = SimpleNamespace(queue_depth=lambda: 0)
    app.state.video_job_manager = SimpleNamespace(queue_depth=lambda: 0)
    app.include_router(api_router)
    settings = build_settings(SERVE_SPA=True)

    configure_web_routes(app, settings, frontend_dist=dist_dir)

    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "spa-root" not in response.text
