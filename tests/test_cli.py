from __future__ import annotations

import json

import pytest

from app import cli, headless


def test_upscale_json_passes_tile_zero(monkeypatch, capsys, tmp_path):
    calls = []
    payload = {
        "ok": True,
        "output": "x.png",
        "width": 8,
        "height": 4,
        "model": "m",
        "device": "dml:0",
        "tile": {"size": 0, "overlap": 10, "meaning": "auto"},
        "seconds": 1.0,
        "scale": 2,
        "nativeScale": 4,
        "resized": True,
        "format": "png",
        "engine": "x",
        "engineModel": "m",
    }
    monkeypatch.setattr(headless, "build_context", lambda: object())

    async def fake(*args, **kwargs):
        calls.append(kwargs)
        return payload

    monkeypatch.setattr(headless, "upscale_image", fake)
    assert cli.main(["upscale", "--in", "a.png", "--out", "x.png", "--scale", "2", "--tile", "0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == payload
    assert calls[0]["tile_size"] == 0

    assert cli.main(["upscale", "--in", "a.png", "--out", "x.png", "--json"]) == 0
    assert calls[1]["tile_size"] is None


def test_upscale_json_reports_headless_error(monkeypatch, capsys):
    monkeypatch.setattr(headless, "build_context", lambda: object())

    async def fake(*args, **kwargs):
        raise headless.ModelNotInstalledError("nope")

    monkeypatch.setattr(headless, "upscale_image", fake)
    assert cli.main(["upscale", "--in", "a.png", "--out", "x.png", "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["code"] == 3


def test_install_requires_confirmation(monkeypatch, capsys):
    monkeypatch.setattr(headless, "install_model", lambda *_: pytest.fail("must not download"))
    assert cli.main(["install", "--repo", "a/b"]) == 2
    assert capsys.readouterr().err.strip() == "error: downloads need --yes"


def test_models_json(monkeypatch, capsys):
    payload = {"models": [{"id": "m", "kind": "builtin_ncnn", "scales": [4]}]}
    monkeypatch.setattr(headless, "build_context", lambda: object())
    monkeypatch.setattr(headless, "list_models", lambda ctx: payload)
    assert cli.main(["models", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == payload


def test_unknown_subcommand():
    with pytest.raises(SystemExit) as error:
        cli.main(["unknown"])
    assert error.value.code == 2
