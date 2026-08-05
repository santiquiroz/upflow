from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

# ---------------------------------------------------------------------------
# El limite de subida solo existia del lado del servidor, y se aplica MIENTRAS
# recibe el archivo (`storage.save_upload`). Con un limite de 2 GB para video,
# eso significa que alguien podia arrastrar un archivo de 3 GB, esperar la
# subida entera y recien ahi enterarse de que no entraba.
#
# Publicarlo en /engine es lo que le permite al navegador avisar antes de subir
# un solo byte. Se publica el numero del servidor, no una copia escrita a mano
# en el frontend: una copia se desincroniza en cuanto alguien cambia el .env.
# ---------------------------------------------------------------------------


def test_engine_info_publishes_both_upload_limits() -> None:
    with TestClient(app) as client:
        payload = client.get("/api/v1/engine").json()

    settings = get_settings()
    assert payload["maxUploadMb"] == settings.max_upload_mb
    assert payload["maxVideoUploadMb"] == settings.max_video_upload_mb


def test_the_published_limits_are_usable_numbers() -> None:
    """Un cero o un negativo dejaria al navegador rechazando todo."""
    with TestClient(app) as client:
        payload = client.get("/api/v1/engine").json()

    assert payload["maxUploadMb"] > 0
    assert payload["maxVideoUploadMb"] >= payload["maxUploadMb"]
