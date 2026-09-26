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


async def test_headless_restore_photo_builds_the_spec_and_default_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    async def restore(ctx, output, spec, **kwargs):
        seen.update(kwargs, output=output, spec=spec)
        return {"ok": True, "output": str(output)}

    monkeypatch.setattr(headless_tools, "get_context", lambda: object())
    monkeypatch.setattr(headless_tools.headless, "restore_image", restore)
    result = json.loads(
        await headless_tools.upflow_restore_photo_headless(
            file_path="C:/x/abuela.jpg", steps=["tone"], options={"tone": {"strength": 0.4}}, scale=2, device="cpu"
        )
    )
    assert result["ok"] is True
    assert Path(seen["output"]).name == "abuela-restored.png"
    assert Path(seen["source"]).name == "abuela.jpg"
    assert seen["token"] is None
    spec = seen["spec"]
    assert spec.steps == ("tone",)
    assert spec.options == {"tone": {"strength": 0.4}}
    assert spec.scale == 2
    assert spec.device == "cpu"
    assert spec.output_format is None


async def test_headless_restore_photo_with_token_needs_a_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(headless_tools, "get_context", lambda: object())
    monkeypatch.setattr(headless_tools.headless, "restore_image", lambda *args, **kwargs: pytest.fail("must not run"))
    error = json.loads(await headless_tools.upflow_restore_photo_headless(token="t" * 32, steps=["tone"]))
    assert error["ok"] is False
    assert error["code"] == headless.EXIT_USAGE
    assert "destination_path" in error["error"]


async def test_headless_restore_photo_reports_missing_pack(monkeypatch: pytest.MonkeyPatch) -> None:
    async def missing(*args, **kwargs):
        raise headless.ModelNotInstalledError("Falta el pack")

    monkeypatch.setattr(headless_tools, "get_context", lambda: object())
    monkeypatch.setattr(headless_tools.headless, "restore_image", missing)
    error = json.loads(await headless_tools.upflow_restore_photo_headless(file_path="C:/x/a.png", steps=["repair"]))
    assert error == {"ok": False, "error": "Falta el pack", "code": 3}


async def test_headless_restore_analyze(monkeypatch: pytest.MonkeyPatch) -> None:
    async def analyze(ctx, source):
        return {"ok": True, "token": "t", "originalName": Path(source).name}

    monkeypatch.setattr(headless_tools, "get_context", lambda: object())
    monkeypatch.setattr(headless_tools.headless, "analyze_photo", analyze)
    result = json.loads(await headless_tools.upflow_restore_analyze_headless("C:/x/a.png"))
    assert result["originalName"] == "a.png"


async def test_headless_restore_recompose_points_to_the_server() -> None:
    error = json.loads(await headless_tools.upflow_restore_recompose_headless("cli-1", {"0": {"blend": 0.2}}))
    assert error["ok"] is False
    assert error["code"] == headless.EXIT_USAGE
    assert "server" in error["error"]
