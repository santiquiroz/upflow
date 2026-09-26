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


def _capture_restore(monkeypatch, payload=None):
    calls = []
    monkeypatch.setattr(headless, "build_context", lambda: object())

    async def fake(ctx, output, spec, **kwargs):
        calls.append({"output": output, "spec": spec, **kwargs})
        return payload or {"ok": True, "output": str(output), "width": 8, "height": 4, "steps": list(spec.steps)}

    monkeypatch.setattr(headless, "restore_image", fake)
    return calls


def test_restore_json_builds_the_spec_from_flags(monkeypatch, capsys):
    calls = _capture_restore(monkeypatch)
    argv = [
        "restore", "--in", "a.jpg", "--out", "b.png", "--steps", "denoise, tone", "--preset", "gentle",
        "--scale", "2", "--upscale", "classic", "--face-blend", "0.5", "--rotate", "90",
        "--crop", "1,2,30,40", "--device", "cpu", "--json",
    ]
    assert cli.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["steps"] == ["denoise", "tone"]
    call = calls[0]
    spec = call["spec"]
    assert str(call["source"]) == "a.jpg"
    assert str(call["output"]) == "b.png"
    assert spec.steps == ("denoise", "tone")
    assert spec.preset == "gentle"
    assert spec.scale == 2
    assert spec.device == "cpu"
    assert spec.options == {
        "geometry": {"rotate90": 1, "crop": [1, 2, 30, 40]},
        "upscale_mode": "classic",
        "faces": {"blend": 0.5},
    }


def test_restore_without_flags_lets_the_analysis_choose(monkeypatch, capsys):
    calls = _capture_restore(monkeypatch)
    assert cli.main(["restore", "--in", "a.jpg", "--out", "b.png"]) == 0
    spec = calls[0]["spec"]
    assert spec.steps == ()
    assert spec.preset is None
    assert spec.options == {}
    assert spec.scale == 1
    assert capsys.readouterr().out.startswith("wrote b.png (8x4)")


def test_restore_bad_crop_or_rotation_exits_2(monkeypatch, capsys):
    _capture_restore(monkeypatch)
    assert cli.main(["restore", "--in", "a.jpg", "--out", "b.png", "--crop", "1,2,3", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["code"] == 2
    with pytest.raises(SystemExit) as error:
        cli.main(["restore", "--in", "a.jpg", "--out", "b.png", "--rotate", "45"])
    assert error.value.code == 2


def test_restore_missing_pack_exits_3(monkeypatch, capsys):
    monkeypatch.setattr(headless, "build_context", lambda: object())

    async def missing(*args, **kwargs):
        raise headless.ModelNotInstalledError("Falta el pack de restauración.")

    monkeypatch.setattr(headless, "restore_image", missing)
    assert cli.main(["restore", "--in", "a.jpg", "--out", "b.png", "--steps", "repair", "--json"]) == 3
    assert json.loads(capsys.readouterr().out) == {"ok": False, "error": "Falta el pack de restauración.", "code": 3}
