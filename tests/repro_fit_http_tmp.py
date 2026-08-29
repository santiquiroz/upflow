from __future__ import annotations
import io, json, re
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app
from app.config import get_settings


def test_http_fit_devuelve_ruta_absoluta(tmp_path):
    get_settings.cache_clear()
    with TestClient(app) as c:
        # token propio, registrado a mi nombre por el flujo normal
        r = c.post("/api/v1/print/parts", json={"width": 10, "height": 10, "depth": 10})
        print("parts:", r.status_code, r.text[:300])
        token = None
        if r.status_code < 300:
            token = r.json().get("token")
        if not token:
            # registro directo, equivalente a lo que hace /print/parts
            from app.api.routes import _register_print_token
            class Req:
                app = c.app
                state = type("S", (), {"current_user": None})()
            token = "b" * 32
            _register_print_token(Req(), token)
        r = c.post(
            f"/api/v1/model3d/fit/{token}",
            files={"file": ("malla.glb", io.BytesIO(b"glTF fake"), "model/gltf-binary")},
            data={"heightMeters": "1.7"},
        )
        print("STATUS:", r.status_code)
        print("BODY:", r.text)
        assert not re.search(r"[A-Za-z]:\\?", r.text), "FILTRA RUTA ABSOLUTA WINDOWS"


def test_http_generate_motor_ausente(tmp_path, monkeypatch):
    monkeypatch.setenv("MESH_ENGINES_DIR", str(tmp_path / "motores-ausentes"))
    get_settings.cache_clear()
    with TestClient(app) as c:
        r = c.post(
            "/api/v1/model3d/generate",
            files={"file": ("img.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
            data={"engine": "triposg"},
        )
        print("STATUS:", r.status_code)
        print("BODY:", r.text)
        assert not re.search(r"[A-Za-z]:\\?", r.text), "FILTRA RUTA ABSOLUTA WINDOWS"
