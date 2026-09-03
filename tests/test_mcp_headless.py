from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import headless
from app.mcp import headless_tools


def test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(headless_tools.MODE_ENV, raising=False)
    assert headless_tools.mode() == "auto"
    monkeypatch.setenv(headless_tools.MODE_ENV, "INPROCESS")
    assert headless_tools.mode() == "inprocess"


async def test_use_server_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(headless_tools.MODE_ENV, "server")
    monkeypatch.setattr(headless_tools, "server_reachable", lambda: pytest.fail("called"))
    assert await headless_tools.use_server()
    monkeypatch.setenv(headless_tools.MODE_ENV, "inprocess")
    assert not await headless_tools.use_server()
    monkeypatch.setenv(headless_tools.MODE_ENV, "auto")
    monkeypatch.setattr(headless_tools, "server_reachable", lambda: _false())
    assert not await headless_tools.use_server()


async def _false() -> bool:
    return False


async def test_health_inprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(headless_tools.MODE_ENV, "inprocess")
    monkeypatch.setattr(headless_tools, "get_context", lambda: object())
    monkeypatch.setattr(
        headless_tools.headless,
        "health",
        lambda ctx: {"version": "1.2.3", "devices": []},
    )
    result = json.loads(await headless_tools.upflow_health())
    assert result["mode"] == "inprocess"
    assert result["version"] == "1.2.3"


async def test_install_requires_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def install(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(headless_tools.headless, "install_model", install)
    result = await headless_tools.upflow_install_upscaler("org/model")
    assert result.startswith("Error:")
    assert not called


async def test_headless_upscale_passes_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    async def upscale(ctx, source, destination, **kwargs):
        seen.update(kwargs)
        seen["destination"] = destination
        return {"ok": True, "output": "o.png"}

    monkeypatch.setattr(headless_tools, "get_context", lambda: object())
    monkeypatch.setattr(headless_tools.headless, "upscale_image", upscale)
    result = await headless_tools.upflow_upscale_image_headless("C:/x/in.png")
    assert json.loads(result)["ok"]
    assert Path(seen["destination"]).name == "in-upscaled.png"
    assert seen["tile_size"] is None
    await headless_tools.upflow_upscale_image_headless("C:/x/in.png", tile_size=0)
    assert seen["tile_size"] == 0

    async def missing(*args, **kwargs):
        raise headless.ModelNotInstalledError("nope")

    monkeypatch.setattr(headless_tools.headless, "upscale_image", missing)
    error = json.loads(await headless_tools.upflow_upscale_image_headless("C:/x/in.png"))
    assert error == {"ok": False, "error": "nope", "code": 3}


async def test_autostart_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    records: list[tuple[list[str], dict[str, object]]] = []

    async def reachable() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(headless_tools, "server_reachable", reachable)
    monkeypatch.setattr(
        headless_tools.subprocess,
        "Popen",
        lambda command, **kwargs: records.append((command, kwargs)),
    )
    monkeypatch.setattr(headless_tools.asyncio, "sleep", _no_sleep)
    assert await headless_tools.autostart_server(8123)
    assert len(records) == 1
    assert "uvicorn" in records[0][0]
    assert "8123" in records[0][0]


async def test_autostart_server_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(headless_tools, "AUTOSTART_TIMEOUT_SECONDS", 2)
    monkeypatch.setattr(headless_tools, "server_reachable", _false)
    monkeypatch.setattr(headless_tools.subprocess, "Popen", lambda *args, **kwargs: None)
    ticks = iter((0.0, 3.0))
    monkeypatch.setattr(headless_tools, "_clock", lambda: next(ticks))
    monkeypatch.setattr(headless_tools.asyncio, "sleep", _no_sleep)
    assert not await headless_tools.autostart_server(8123)


async def _no_sleep(delay: float) -> None:
    return None
