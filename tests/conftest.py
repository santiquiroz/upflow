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
    # Los packs opcionales se resuelven contra vendor/ del REPO, no contra
    # RUNTIME_DIR: sin esto, un test que mira las capacidades da distinto en una
    # maquina con el pack bajado que en una limpia (paso de verdad con el motor
    # de borrado). Los tests que SI quieren probarlo pasan la ruta explicita.
    monkeypatch.setenv("MIGAN_MODEL", str(tmp_path / "pack-no-instalado.onnx"))
    # Mismo motivo para el lane Vulkan: con el pack bajado, las capacidades
    # incluyen los checkpoints reales del disco y los tests de "no hay modelos"
    # fallan solo en la maquina del que lo instalo.
    monkeypatch.setenv("SDCPP_MODEL", str(tmp_path / "pack-no-instalado.safetensors"))
    monkeypatch.setenv("SDCPP_MODELS_DIR", str(tmp_path / "vulkan-no-instalado"))
    # Y lo mismo con los plugins EP: desde que el default apunta a
    # vendor/ep-plugins, una maquina con el acelerador de Intel bajado hacia
    # fallar tests que en una limpia pasaban.
    monkeypatch.setenv("EP_PLUGINS_DIR", str(tmp_path / "ep-plugins-no-instalado"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def reset_ep_registry():
    """El ep_registry guarda estado de proceso (plugins registrados, errores
    de sesión); sin reset, un test contaminaría la resolución de EP del
    siguiente."""
    from app.services import ep_registry

    ep_registry.reset()
    yield
    ep_registry.reset()
