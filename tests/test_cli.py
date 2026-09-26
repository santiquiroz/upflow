from __future__ import annotations

import json
from pathlib import Path

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


# ---------------------------------------------------------------- cctv


def forbid_cctv_work(monkeypatch) -> None:
    monkeypatch.setattr(headless, "build_context", lambda: pytest.fail("must fail before building the context"))


@pytest.mark.parametrize(
    "osd_flags",
    [[], ["--osd", "0,0,96,24"]],
)
def test_cctv_clarify_without_an_osd_decision_exits_2_with_the_osd_message(monkeypatch, capsys, osd_flags):
    forbid_cctv_work(monkeypatch)

    code = cli.main(["cctv", "clarify", "--in", "clip.mp4", "--out-dir", "caso", *osd_flags, "--json"])

    error = json.loads(capsys.readouterr().out)
    assert code == 2 and error["code"] == 2 and error["key"] == "cctv.error.osdUnconfirmed"
    assert "Confirm the on-screen text boxes" in error["error"] and "--no-osd" in error["error"]


def test_cctv_clarify_osd_confirmed_and_no_osd_are_exclusive(monkeypatch):
    forbid_cctv_work(monkeypatch)

    with pytest.raises(SystemExit) as error:
        cli.main(["cctv", "clarify", "--in", "c.mp4", "--out-dir", "o", "--osd-confirmed", "--no-osd"])

    assert error.value.code == 2


@pytest.mark.parametrize("flag", [["--trim", "10"], ["--osd", "0,0,96"], ["--frames", "5,x"], ["--preset", "sunny"]])
def test_cctv_clarify_malformed_flags_exit_2(monkeypatch, flag):
    forbid_cctv_work(monkeypatch)

    with pytest.raises(SystemExit) as error:
        cli.main(["cctv", "clarify", "--in", "c.mp4", "--out-dir", "o", "--no-osd", *flag])

    assert error.value.code == 2


def test_cctv_clarify_passes_every_choice_to_headless(monkeypatch, capsys):
    calls = []
    payload = {"ok": True, "jobId": "j1", "outputDir": "caso/j1.cctv", "framesIn": 38, "framesOut": 38}
    monkeypatch.setattr(headless, "build_context", lambda: "ctx")

    async def fake(ctx, source, out_dir, choices):
        calls.append((ctx, source, out_dir, choices))
        return payload

    monkeypatch.setattr(headless, "cctv_clarify_file", fake)
    argv = [
        "cctv", "clarify", "--in", "clip.mp4", "--out-dir", "caso", "--preset", "night_ir",
        "--osd", "0,0,96,24", "--osd", "200,220,96,20", "--osd-confirmed",
        "--trim", "3:40", "--frames", "5,30", "--json",
    ]  # fmt: skip

    assert cli.main(argv) == 0

    assert json.loads(capsys.readouterr().out) == payload
    ctx, source, out_dir, choices = calls[0]
    assert (ctx, source, out_dir) == ("ctx", Path("clip.mp4"), Path("caso"))
    assert choices == headless.CctvClarifyChoices(
        preset="night_ir",
        osd_boxes=((0, 0, 96, 24), (200, 220, 96, 20)),
        osd_confirmed=True,
        trim=(3, 40),
        still_frames=(5, 30),
    )


def test_cctv_probe_prints_a_readable_line(monkeypatch, capsys):
    monkeypatch.setattr(headless, "build_context", lambda: "ctx")
    seen = []

    async def fake(ctx, source):
        seen.append(source)
        return {"ok": True, "sourceSha256": "ab" * 32, "container": {"label": "Hikvision (IMKH)"}, "suggestedPreset": "day", "decodeFailed": False, "warnings": []}

    monkeypatch.setattr(headless, "cctv_probe", fake)

    assert cli.main(["cctv", "probe", "--in", "clip.mp4"]) == 0

    out = capsys.readouterr().out
    assert seen == [Path("clip.mp4")]
    assert "sha256=" + "ab" * 32 in out and "container=Hikvision (IMKH)" in out and "suggestedPreset=day" in out


def test_cctv_probe_reports_the_error_key(monkeypatch, capsys):
    monkeypatch.setattr(headless, "build_context", lambda: "ctx")

    async def fake(ctx, source):
        raise headless.ModelNotInstalledError("ffmpeg is not available for CCTV mode.", key="cctv.error.ffmpegUnavailable")

    monkeypatch.setattr(headless, "cctv_probe", fake)

    assert cli.main(["cctv", "probe", "--in", "clip.mp4"]) == 3
    assert capsys.readouterr().err.strip() == "error: ffmpeg is not available for CCTV mode. [cctv.error.ffmpegUnavailable]"


def test_cctv_verify_exits_5_when_a_file_changed(monkeypatch, capsys):
    result = {"directory": "caso/j1.cctv", "ok": False, "checked": 4, "mismatches": ["report.json"], "missing": []}
    monkeypatch.setattr(headless, "cctv_check_unchanged", lambda directory: result)

    assert cli.main(["cctv", "verify", "--dir", "caso/j1.cctv"]) == 5
    assert capsys.readouterr().out.strip() == "CHANGED: report.json MISSING: -"

    monkeypatch.setattr(headless, "cctv_check_unchanged", lambda directory: {**result, "ok": True, "mismatches": []})
    assert cli.main(["cctv", "verify", "--dir", "caso/j1.cctv"]) == 0
    assert capsys.readouterr().out.strip() == "unchanged: 4 files in caso/j1.cctv"
