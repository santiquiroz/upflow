from __future__ import annotations

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def isolated_runtime_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Points RUNTIME_DIR at a per-test tmp directory so no test can ever
    read from or write into the repo's real runtime/ folder.

    Without this, any test that builds the app via `TestClient(app)` picks up
    the real .env's RUNTIME_DIR and the live retention sweeper, which runs
    immediately on startup, ends up deleting real uploads/outputs.
    """
    monkeypatch.setenv("RUNTIME_DIR", str(tmp_path / "runtime"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
