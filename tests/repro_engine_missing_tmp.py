from pathlib import Path
import io, re
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import Settings, get_settings
from app.services import mesh_engine_service as m


def test_missing_lleva_ruta_absoluta(tmp_path, monkeypatch):
    monkeypatch.setenv("MESH_ENGINES_ROOT", str(tmp_path / "sin-motores"))
    get_settings.cache_clear()
    s = get_settings()
    b = m.build_for(s, "triposg")
    print("ready:", b.ready)
    print("MISSING:", b.missing)
    with pytest.raises(m.MeshEngineError) as ei:
        m.run_engine(s, "triposg", {"image": "x", "output": "y"})
    print("EXC:", str(ei.value))
    with TestClient(app) as c:
        r = c.post("/api/v1/model3d/generate",
                   files={"file": ("img.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")},
                   data={"engine": "triposg"})
        print("HTTP:", r.status_code, r.text)
    assert not re.search(r"[A-Za-z]:\\", str(ei.value)), "leak"
