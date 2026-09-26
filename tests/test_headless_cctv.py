from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from app import cli, headless
from app.config import Settings
from app.models import RoiFusionRequest
from app.services.cctv_analysis import FFMPEG_UNAVAILABLE, CctvAnalysisError
from app.services.cctv_chain import CctvChainError
from app.services.cctv_session import JOB_DIR_SUFFIX
from app.services.frame_export import StillFrameError
from ffmpeg_support import needs_ffmpeg

INTERLACED_LITE = {
    "quality": {"interlace": {"interlaced": True}},
    "video": {"lite": {"sar": "4:3"}},
    "suggestedPreset": "night_ir",
}


def step_ids(steps) -> list[str]:
    return [step["id"] for step in steps]


# --- Pasos del preset


@pytest.mark.parametrize(("sar", "expected"), [("4:3", (4, 3)), (None, None), ("0:1", None), ("a:b", None), ("4", None)])
def test_cctv_sample_aspect_is_parsed_like_the_ui(sar, expected) -> None:
    assert headless.parse_sample_aspect(sar) == expected


def test_cctv_preset_steps_follow_the_diagnosis_of_the_clip() -> None:
    steps = headless.classic_preset_steps(INTERLACED_LITE)
    plain = headless.classic_preset_steps({"quality": None, "video": {"lite": None}})

    assert set(steps) == {"day", "night_ir", "analog", "low_res"}
    assert step_ids(steps["day"])[:2] == ["aspect", "deinterlace"]
    assert steps["day"][0]["params"] == {"num": 4, "den": 3}
    assert "deinterlace" not in step_ids(plain["day"]) and "aspect" not in step_ids(plain["day"])
    assert "deinterlace" in step_ids(plain["analog"])
    assert all("ai" not in step_id for preset in steps.values() for step_id in step_ids(preset))


def test_cctv_choices_take_the_suggested_preset_when_none_is_given() -> None:
    chosen = headless.choices_with_preset_steps(INTERLACED_LITE, headless.CctvClarifyChoices(no_osd=True))
    explicit = headless.CctvClarifyChoices(preset="day", steps=(), no_osd=True)

    assert chosen.preset == "night_ir" and "gray" in step_ids(chosen.steps)
    assert headless.choices_with_preset_steps(INTERLACED_LITE, explicit) is explicit
    assert headless.choices_with_preset_steps({}, headless.CctvClarifyChoices()).steps == ()


def test_cctv_an_unknown_preset_is_a_usage_error() -> None:
    with pytest.raises(headless.UsageError) as error:
        headless.choices_with_preset_steps({}, headless.CctvClarifyChoices(preset="sunny"))

    assert error.value.key == "cctv.error.unknownPreset" and error.value.exit_code == 2


def test_cctv_a_preset_by_token_needs_its_steps() -> None:
    with pytest.raises(headless.UsageError, match="presetSteps"):
        headless.require_steps_for_preset(headless.CctvClarifyChoices(preset="day"))

    headless.require_steps_for_preset(headless.CctvClarifyChoices(preset="day", steps=()))
    headless.require_steps_for_preset(headless.CctvClarifyChoices())


# --- Opciones del job


@pytest.mark.parametrize("raw", ["denoise", {"params": {}}, {"id": "denoise", "params": [1]}, {"id": 3}])
def test_cctv_a_malformed_step_is_a_usage_error(raw) -> None:
    with pytest.raises(headless.UsageError):
        headless.normalized_step(raw)


def test_cctv_options_carry_every_choice() -> None:
    choices = headless.CctvClarifyChoices(
        preset="day",
        steps=({"id": "denoise", "params": {"filter": "hqdn3d"}},),
        osd_boxes=((0, 0, 96, 24),),
        osd_confirmed=True,
        trim=(3, 40),
        still_frames=(5,),
        acquisition={"recorderMake": "HiLook"},
    )

    options = headless.cctv_options_of("tok", choices)

    assert (options.task, options.session_token, options.preset) == ("clarify", "tok", "day")
    assert options.steps[0].id == "denoise" and dict(options.steps[0].params) == {"filter": "hqdn3d"}
    assert options.osd_boxes == ((0, 0, 96, 24),) and options.osd_boxes_confirmed and not options.no_osd
    assert options.trim == (3, 40) and options.still_frames == (5,)
    assert dict(options.acquisition) == {"recorderMake": "HiLook"}


# --- Errores con codigo estable


@pytest.mark.parametrize(
    ("exc", "code", "key"),
    [
        (CctvChainError("cctv.error.trimOutOfRange", "bad trim"), 2, "cctv.error.trimOutOfRange"),
        (StillFrameError("cctv.error.stillOutsideTrim", "bad still"), 2, "cctv.error.stillOutsideTrim"),
        (CctvAnalysisError(FFMPEG_UNAVAILABLE, "no ffmpeg", 503), 3, FFMPEG_UNAVAILABLE),
        (CctvAnalysisError("cctv.error.analysisFailed", "boom", 500), 5, "cctv.error.analysisFailed"),
        (CctvAnalysisError("cctv.error.noVideoStream", "no video"), 2, "cctv.error.noVideoStream"),
        (RuntimeError("ffmpeg died"), 5, None),
    ],
)
def test_cctv_errors_map_to_the_cli_exit_codes(exc, code, key) -> None:
    mapped = headless.cctv_error(exc)

    assert (mapped.exit_code, mapped.key, str(mapped)) == (code, key, str(exc))


def test_cctv_a_build_without_cctv_support_is_a_missing_pack() -> None:
    with pytest.raises(headless.ModelNotInstalledError) as error:
        headless.require_cctv_mode({"modeAvailable": False, "modeUnavailableReason": "capability.setup.ffmpegBuildLacksCctv"})

    assert error.value.key == "capability.setup.ffmpegBuildLacksCctv"
    headless.require_cctv_mode({"modeAvailable": True})


@pytest.mark.parametrize("job_id", ["../x", "a/b", "", "x" * 65])
def test_cctv_a_job_id_cannot_leave_the_outputs_folder(tmp_path: Path, job_id: str) -> None:
    with pytest.raises(headless.UsageError):
        headless.checked_job_id(job_id)


# --- Antes de tocar el archivo


async def test_cctv_clarify_file_checks_the_osd_decision_before_probing(monkeypatch, tmp_path: Path) -> None:
    async def probe(*args, **kwargs):
        pytest.fail("the probe must not run without an OSD decision")

    monkeypatch.setattr(headless, "cctv_probe", probe)

    with pytest.raises(headless.UsageError) as error:
        await headless.cctv_clarify_file(object(), tmp_path / "clip.mp4", tmp_path / "out", headless.CctvClarifyChoices())

    assert error.value.key == "cctv.error.osdUnconfirmed"


async def test_cctv_clarify_file_rejects_an_output_path_that_is_a_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(headless, "cctv_probe", lambda *args, **kwargs: pytest.fail("must not probe"))
    taken = tmp_path / "out"
    taken.write_text("x", encoding="utf-8")

    with pytest.raises(headless.UsageError, match="output folder"):
        await headless.cctv_clarify_file(object(), tmp_path / "c.mp4", taken, headless.CctvClarifyChoices(no_osd=True))


async def test_cctv_probe_of_a_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    with pytest.raises(headless.UsageError, match="not found"):
        await headless.cctv_probe(object(), tmp_path / "nope.mp4")


def test_cctv_check_unchanged_needs_the_result_folder(tmp_path: Path) -> None:
    with pytest.raises(headless.UsageError, match="not found"):
        headless.cctv_check_unchanged(tmp_path / "missing.cctv")


# --- De punta a punta con ffmpeg real, por la CLI


def make_clip(path: Path, settings: Settings) -> Path:
    command = [
        str(settings.ffmpeg_binary_path), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25,noise=alls=12:allf=t",
        "-f", "lavfi", "-i", "sine=sample_rate=8000",
        "-t", "2", "-threads", "1", "-c:v", "libx264", "-x264-params", "threads=1", "-b:v", "200k", "-g", "25",
        "-c:a", "pcm_alaw", "-shortest", str(path),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return path


def run_cli(capsys, argv: list[str]) -> tuple[int, dict]:
    code = cli.main([*argv, "--json"])
    return code, json.loads(capsys.readouterr().out)


def leftovers(settings: Settings) -> list[Path]:
    sessions = list(settings.video_work_path.glob("*")) if settings.video_work_path.exists() else []
    return sessions + list(settings.outputs_path.glob(f"*{JOB_DIR_SUFFIX}"))


@needs_ffmpeg
def test_cctv_cli_probe_clarify_and_verify_on_a_real_clip(monkeypatch, capsys, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    build = headless.build_context
    monkeypatch.setattr(headless, "build_context", lambda: build(settings))
    clip = make_clip(tmp_path / "cámara 1.mkv", settings)
    clip_sha = hashlib.sha256(clip.read_bytes()).hexdigest()

    code, probe = run_cli(capsys, ["cctv", "probe", "--in", str(clip)])
    assert code == 0 and probe["sourceSha256"] == clip_sha and probe["token"] is None
    assert "day" in probe["presetSteps"] and leftovers(settings) == []

    out_dir = tmp_path / "caso"
    argv = [
        "cctv", "clarify", "--in", str(clip), "--out-dir", str(out_dir), "--preset", "day",
        "--osd", "0,0,96,24", "--osd-confirmed", "--trim", "3:40", "--frames", "5,30",
    ]  # fmt: skip
    code, result = run_cli(capsys, argv)
    assert code == 0, result
    result_dir = Path(result["outputDir"])
    assert result_dir.parent == out_dir.resolve() and result_dir.name == f"{result['jobId']}{JOB_DIR_SUFFIX}"
    assert result["sourceSha256"] == clip_sha and result["framesOut"] == 38 and result["preset"] == "day"
    assert (result_dir / result["outputs"]["package"]).is_file() and Path(result["report"]).is_file()
    assert len(result["outputs"]["stills"]) == 2
    assert leftovers(settings) == [] and hashlib.sha256(clip.read_bytes()).hexdigest() == clip_sha

    code, verified = run_cli(capsys, ["cctv", "verify", "--dir", str(result_dir)])
    assert code == 0 and verified["ok"] and verified["checked"] > 0

    (result_dir / "report.json").write_text("{}", encoding="utf-8")
    code, tampered = run_cli(capsys, ["cctv", "verify", "--dir", str(result_dir)])
    assert code == 5 and tampered["mismatches"] == ["report.json"]


# --- Foto multi-cuadro de una ROI


def roi_request(**overrides) -> RoiFusionRequest:
    fields = {"first_frame": 10, "last_frame": 40, "reference_frame": 22, "box": (0, 0, 40, 40), "kind": "plate"}
    return RoiFusionRequest(**{**fields, **overrides})


def test_cctv_roi_takes_only_deinterlace_and_deblock_from_the_preset() -> None:
    chosen = headless.roi_choices_with_steps(INTERLACED_LITE, headless.CctvRoiChoices(roi=roi_request()))
    explicit = headless.CctvRoiChoices(roi=roi_request(), steps=())

    assert chosen.preset == "night_ir" and step_ids(chosen.steps) == ["deinterlace", "deblock"]
    assert headless.roi_choices_with_steps(INTERLACED_LITE, explicit) is explicit
    assert headless.roi_choices_with_steps({}, headless.CctvRoiChoices(roi=roi_request())).steps == ()


def test_cctv_roi_options_carry_the_region() -> None:
    choices = headless.CctvRoiChoices(
        roi=roi_request(), preset="day", steps=({"id": "deblock"},), acquisition={"recorderMake": "HiLook"}
    )

    options = headless.cctv_roi_options_of("tok", choices)

    assert (options.task, options.session_token, options.preset) == ("roi_fusion", "tok", "day")
    assert options.roi == roi_request() and [step.id for step in options.steps] == ["deblock"]
    assert options.osd_boxes == () and options.trim is None and options.still_frames == ()
    assert dict(options.acquisition) == {"recorderMake": "HiLook"}


@pytest.mark.parametrize(
    ("overrides", "key"),
    [
        ({"reference_frame": 9}, "cctv.error.roiFrames"),
        ({"reference_frame": 41}, "cctv.error.roiFrames"),
        ({"first_frame": -1, "reference_frame": 0}, "cctv.error.roiFrames"),
        ({"kind": "car"}, "cctv.error.roiInvalid"),
        ({"scale": 5}, "cctv.error.roiInvalid"),
        ({"method": "mean"}, "cctv.error.roiInvalid"),
    ],
)
def test_cctv_roi_bad_choices_are_usage_errors(overrides, key) -> None:
    with pytest.raises(headless.UsageError) as error:
        headless.check_roi_request(roi_request(**overrides))

    assert error.value.key == key and error.value.exit_code == 2


async def test_cctv_roi_file_checks_the_region_before_probing(monkeypatch, tmp_path: Path) -> None:
    async def probe(*args, **kwargs):
        pytest.fail("the probe must not run with an invalid region")

    monkeypatch.setattr(headless, "cctv_probe", probe)
    choices = headless.CctvRoiChoices(roi=roi_request(reference_frame=50))

    with pytest.raises(headless.UsageError) as error:
        await headless.cctv_roi_file(object(), tmp_path / "clip.mp4", tmp_path / "out", choices)

    assert error.value.key == "cctv.error.roiFrames"


@needs_ffmpeg
def test_cctv_cli_roi_and_verify_on_a_real_clip(monkeypatch, capsys, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    build = headless.build_context
    monkeypatch.setattr(headless, "build_context", lambda: build(settings))
    clip = make_clip(tmp_path / "cámara 1.mkv", settings)
    clip_sha = hashlib.sha256(clip.read_bytes()).hexdigest()
    out_dir = tmp_path / "caso"
    argv = [
        "cctv", "roi", "--in", str(clip), "--out-dir", str(out_dir), "--frames", "5:20", "--ref", "10",
        "--box", "120,80,64,48", "--kind", "face_or_object",
    ]  # fmt: skip

    code, result = run_cli(capsys, argv)

    assert code == 0, result
    result_dir = Path(result["outputDir"])
    assert result_dir.parent == out_dir.resolve() and result_dir.name == f"{result['jobId']}{JOB_DIR_SUFFIX}"
    assert (result["task"], result["lane"], result["sourceSha256"]) == ("roi_fusion", "classic", clip_sha)
    roi = result["roi"]
    assert (roi["kind"], roi["scale"], roi["method"], roi["referenceFrame"]) == ("face_or_object", 2, "median", 10)
    assert roi["framesTotal"] == 16 and 1 <= roi["effectiveSamples"] <= 16
    assert all((result_dir / path).is_file() for path in result["outputs"]["roi"].values())
    assert Path(result["reportJson"]).is_file()
    assert leftovers(settings) == [] and hashlib.sha256(clip.read_bytes()).hexdigest() == clip_sha

    code, verified = run_cli(capsys, ["cctv", "verify", "--dir", str(result_dir)])
    assert code == 0 and verified["ok"] and verified["checked"] > 0


@needs_ffmpeg
def test_cctv_cli_roi_without_ref_uses_the_suggested_reference(monkeypatch, capsys, tmp_path: Path) -> None:
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    build = headless.build_context
    monkeypatch.setattr(headless, "build_context", lambda: build(settings))
    clip = make_clip(tmp_path / "clip.mkv", settings)
    argv = [
        "cctv", "roi", "--in", str(clip), "--out-dir", str(tmp_path / "caso"), "--frames", "5:20",
        "--box", "120,80,64,48", "--kind", "face_or_object",
    ]  # fmt: skip

    code, result = run_cli(capsys, argv)

    assert code == 0, result
    assert 5 <= result["roi"]["referenceFrame"] <= 20
    assert leftovers(settings) == []
