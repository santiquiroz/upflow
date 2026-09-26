from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import cli, headless
from app.models import RoiFusionRequest


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


def test_json_stdout_stays_parseable_when_a_library_prints_during_the_command(monkeypatch, capsys):
    # onnxruntime avisa con print() en stdout cuando una sesion DML cae a CPU (visto con DDColor en P1-GPU-smoke).
    payload = {"ok": True, "output": "b.png", "width": 8, "height": 4, "steps": ["colorize"]}

    async def noisy(ctx, output, spec, **kwargs):
        print("*************** EP Error ***************")
        return payload

    monkeypatch.setattr(headless, "build_context", lambda: object())
    monkeypatch.setattr(headless, "restore_image", noisy)
    assert cli.main(["restore", "--in", "a.jpg", "--out", "b.png", "--steps", "colorize", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == payload
    assert "EP Error" in captured.err


def test_json_error_stays_parseable_when_a_library_prints_before_failing(monkeypatch, capsys):
    async def noisy_failure(ctx, output, spec, **kwargs):
        print("EP Error: falling back")
        raise RuntimeError("boom")

    monkeypatch.setattr(headless, "build_context", lambda: object())
    monkeypatch.setattr(headless, "restore_image", noisy_failure)
    assert cli.main(["restore", "--in", "a.jpg", "--out", "b.png", "--steps", "denoise", "--json"]) == 5
    assert json.loads(capsys.readouterr().out) == {"ok": False, "error": "boom", "code": 5}


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


# ---------------------------------------------------------------- cctv roi


ROI_ARGV = ["cctv", "roi", "--in", "clip.mp4", "--out-dir", "caso", "--frames", "10:40", "--ref", "22"]


def test_cctv_roi_passes_every_choice_to_headless(monkeypatch, capsys):
    calls = []
    payload = {"ok": True, "jobId": "j2", "outputDir": "caso/j2.cctv", "roi": {"framesUsed": 23}}
    monkeypatch.setattr(headless, "build_context", lambda: "ctx")

    async def fake(ctx, source, out_dir, choices):
        calls.append((ctx, source, out_dir, choices))
        return payload

    monkeypatch.setattr(headless, "cctv_roi_file", fake)
    argv = [
        *ROI_ARGV, "--box", "100,80,64,24", "--kind", "plate", "--scale", "3",
        "--method", "trimmed_mean", "--preset", "night_ir", "--json",
    ]  # fmt: skip

    assert cli.main(argv) == 0

    assert json.loads(capsys.readouterr().out) == payload
    ctx, source, out_dir, choices = calls[0]
    assert (ctx, source, out_dir) == ("ctx", Path("clip.mp4"), Path("caso"))
    assert choices == headless.CctvRoiChoices(
        roi=RoiFusionRequest(10, 40, 22, (100, 80, 64, 24), "plate", 3, "trimmed_mean"), preset="night_ir"
    )


def test_cctv_roi_defaults_to_2x_median_and_the_suggested_preset(monkeypatch):
    calls = []
    monkeypatch.setattr(headless, "build_context", lambda: "ctx")

    async def fake(ctx, source, out_dir, choices):
        calls.append(choices)
        return {"ok": True}

    monkeypatch.setattr(headless, "cctv_roi_file", fake)

    assert cli.main([*ROI_ARGV, "--box", "0,0,40,40", "--kind", "face_or_object", "--json"]) == 0

    assert calls == [headless.CctvRoiChoices(roi=RoiFusionRequest(10, 40, 22, (0, 0, 40, 40), "face_or_object"))]


def test_cctv_roi_without_ref_asks_for_the_suggested_reference(monkeypatch):
    calls = []
    monkeypatch.setattr(headless, "build_context", lambda: "ctx")

    async def fake(ctx, source, out_dir, choices):
        calls.append(choices)
        return {"ok": True}

    monkeypatch.setattr(headless, "cctv_roi_file", fake)
    argv = ["cctv", "roi", "--in", "c.mp4", "--out-dir", "o", "--frames", "10:40", "--box", "0,0,40,40", "--kind", "plate"]

    assert cli.main([*argv, "--json"]) == 0

    (choices,) = calls
    assert choices.suggest_reference is True
    assert choices.roi.reference_frame == 10


@pytest.mark.parametrize(
    "flags",
    [
        ["--box", "0,0,40,40", "--kind", "car"],
        ["--box", "0,0,40,40", "--kind", "plate", "--scale", "5"],
        ["--box", "0,0,40", "--kind", "plate"],
        ["--box", "0,0,40,40", "--kind", "plate", "--method", "mean"],
        ["--box", "0,0,40,40"],
        ["--kind", "plate"],
    ],
)
def test_cctv_roi_malformed_flags_exit_2(monkeypatch, flags):
    forbid_cctv_work(monkeypatch)

    with pytest.raises(SystemExit) as error:
        cli.main([*ROI_ARGV, *flags])

    assert error.value.code == 2


def test_cctv_roi_a_reference_outside_the_range_exits_2_before_touching_the_clip(monkeypatch, capsys):
    forbid_cctv_work(monkeypatch)
    argv = ["cctv", "roi", "--in", "c.mp4", "--out-dir", "o", "--frames", "10:40", "--ref", "41"]

    code = cli.main([*argv, "--box", "0,0,40,40", "--kind", "plate", "--json"])

    error = json.loads(capsys.readouterr().out)
    assert code == 2 and error["key"] == "cctv.error.roiFrames"


def test_cctv_roi_prints_a_readable_line(monkeypatch, capsys):
    monkeypatch.setattr(headless, "build_context", lambda: "ctx")
    roi = {
        "kind": "plate", "scale": 3, "framesTotal": 30, "framesUsed": 23, "effectiveSamples": 6,
        "nearCopies": False, "notices": [{"key": "cctv.roi.densityPlate", "params": {"px": 14}}],
    }  # fmt: skip

    async def fake(ctx, source, out_dir, choices):
        return {"ok": True, "jobId": "j2", "outputDir": "caso/j2.cctv", "roi": roi, "seconds": 3.5}

    monkeypatch.setattr(headless, "cctv_roi_file", fake)

    assert cli.main([*ROI_ARGV, "--box", "0,0,40,40", "--kind", "plate"]) == 0

    assert capsys.readouterr().out.strip() == (
        "wrote caso/j2.cctv job=j2 kind=plate scale=3x framesUsed=23/30 effective=6 nearCopies=False "
        "notices=cctv.roi.densityPlate 3.5s"
    )
