from __future__ import annotations

import io

from fastapi.testclient import TestClient

from app.main import app


def test_unknown_engine_http_detail():
    with TestClient(app) as client:
        r = client.post(
            "/api/v1/model3d/generate",
            files={"file": ("x.png", io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 64), "image/png")},
            data={"engine": "hunyuan3d"},
        )
        print("STATUS", r.status_code)
        print("BODY", r.text)
        assert "triposg" in r.text, "el detalle NO enumera los motores conocidos"
