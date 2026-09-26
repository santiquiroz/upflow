from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from app.services.cctv_chain import GEOMETRY_CHANGED, NEW_PIXEL_VALUES, step_spec, steps_from_request
from app.services.cctv_clarify_runner import CLIPPING_INCREASED_KEY, ClippingReport
from app.services.cctv_frame_index import FrameEntry, summarize_frame_index
from app.services.cctv_ingest import (
    DecodeCheck,
    FrameIndex,
    ReceivedAt,
    RemuxAttempt,
    SourceRecord,
    VerifiedCopy,
    VideoStream,
    WorkingCopy,
    WorkingIngest,
)
from app.services.cctv_report import (
    AI_DETAIL,
    CLOCK_OFFSET,
    DISCLAIMER_TEXT,
    GENERATIVE_UPSCALE,
    GUIDELINES_TEXT,
    HASH_SCOPE,
    LOSSY_COPIES,
    REMUXED_PROPRIETARY,
    SAME_BUILD,
    SHA256SUMS_NAME,
    HostFacts,
    OutputArtifact,
    ProcessRun,
    ReportParts,
    ReportTools,
    StepRun,
    build_report,
    host_facts,
    load_report,
    parse_sha256sums,
    report_json_text,
    write_report,
    write_sha256sums,
)
from app.services.cctv_report_html import render_report_html
from app.services.cctv_report_model import (
    SCHEMA_PATH,
    Acquisition,
    CaseInfo,
    ReproductionOf,
    CctvReportV1,
    EngineInfo,
    report_schema,
    report_schema_text,
)
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.frame_export import StillFrame, StillPair
from app.services.media_signature import HIKVISION_PS, MP4
from app.services.osd_check import OsdSelection, osd_report
from app.services.video_analysis import CctvDiagnosis

FFMPEG_SHA = "f" * 64
SOURCE_SHA = "a" * 64
WORK_SHA = "b" * 64
CSV_SHA = "c" * 64
FRAME_SHA = "d" * 64
START = datetime(2026, 9, 25, 14, 0, 0, tzinfo=timezone.utc)
LOCAL_TZ = timezone(timedelta(hours=-5))
FPS = 25

CAPS = FfmpegCapabilities(
    binary_sha256=FFMPEG_SHA,
    version="ffmpeg version N-123588-g9c63742425-20260323 Copyright (c) 2000-2026 the FFmpeg developers",
    configuration=("--enable-gpl", "--enable-libx264"),
    filters=frozenset({"hqdn3d", "bwdif"}),
    encoders=frozenset({"ffv1", "libx264"}),
    cpu_extensions=("avx2", "avx512"),
)
HOST = HostFacts(os="Windows 11 (build 26200)", cpu_model="AMD Ryzen 9 7900X3D", python="3.11.7")

CLASSIC_REQUEST = [
    {"id": "trim", "params": {"start_frame": 2, "end_frame": 5}},
    {"id": "deblock", "params": {"filter": "deblock", "filter_type": "weak", "block": 8}},
    {"id": "denoise", "params": {"filter": "hqdn3d", "luma_spatial": 3.0}},
]


def frames(count: int = 10) -> tuple[FrameEntry, ...]:
    return tuple(FrameEntry(n, n / FPS, n % 5 == 0, "I" if n % 5 == 0 else "P", 1000 + n) for n in range(count))


def make_files(base: Path) -> dict[str, Path]:
    paths = {
        "original": base / "01_original" / "clip.mp4",
        "analysis": base / "analysis.mkv",
        "viewing": base / "viewing.mp4",
        "comparison": base / "comparison.mp4",
        "still_o": base / "stills" / "original_f3.png",
        "still_p": base / "stills" / "processed_f3.png",
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{name} bytes".encode())
    return paths


def make_ingest(base: Path) -> WorkingIngest:
    entries = frames()
    working = WorkingCopy(
        path=base / "work.mkv",
        sha256=WORK_SHA,
        method="auto",
        attempts=(RemuxAttempt("auto", (), True, "ok"),),
        copytb=False,
    )
    index = FrameIndex(entries, summarize_frame_index(entries), base / "frame_index.csv", CSV_SHA)
    return WorkingIngest(
        source_probe={"format": {"format_name": "mov,mp4"}, "streams": [{"codec_type": "video"}]},
        working_copy=working,
        video=VideoStream("h264", 64, 48, "25/1"),
        audio=(),
        decode=DecodeCheck(frames_decoded=10, error_lines=0, decode_failed=False),
        index=index,
        lite=None,
    )


def make_source(container=MP4) -> SourceRecord:
    return SourceRecord(
        original_name="clip.mp4",
        size_bytes=1234,
        modified_at="2026-09-24T10:00:00.000Z",
        sha256=SOURCE_SHA,
        received_at=ReceivedAt(utc="2026-09-25T14:00:00.000Z", local="2026-09-25T09:00:00.000-05:00"),
        container=container,
    )


def still_pair(paths: dict[str, Path]) -> StillPair:
    original = StillFrame("original", 3, paths["still_o"], 3 / FPS, FRAME_SHA)
    processed = StillFrame("processed", 3, paths["still_p"], 3 / FPS, FRAME_SHA)
    return StillPair(3, original, processed, approximate=False)


def make_parts(base: Path, request=CLASSIC_REQUEST, **overrides) -> ReportParts:
    paths = make_files(base)
    chain = steps_from_request(request, "classic")
    run = ProcessRun("analysis copy (FFV1)", ("ffmpeg", "-i", "work.mkv", "analysis.mkv"), START,
                     START + timedelta(seconds=3), 0, 4, 4)
    viewing = ProcessRun("viewing copy (H.264)", ("ffmpeg", "-i", "analysis.mkv", "viewing.mp4"),
                         START + timedelta(seconds=3), START + timedelta(seconds=4), 0, 4, 4)
    parts = ReportParts(
        mode="classic",
        app_version="0.80.0",
        commit="abc1234",
        case=CaseInfo(case_label="Case 7", operator_name="Santiago"),
        acquisition=Acquisition(recorder_make="HiLook"),
        caps=CAPS,
        host=HOST,
        source=make_source(),
        verified_copy=VerifiedCopy(paths["original"], SOURCE_SHA),
        ingest=make_ingest(base),
        diagnosis=None,
        osd=osd_report(OsdSelection((), False, True), (), chain),
        chain=chain,
        chain_run=run,
        processes=(run, viewing),
        outputs=(
            OutputArtifact(paths["analysis"], "analysis-lossless", {"format": {"format_name": "matroska,webm"}}),
            OutputArtifact(paths["viewing"], "viewing-copy"),
            OutputArtifact(paths["comparison"], "comparison"),
            OutputArtifact(paths["still_o"], "still"),
            OutputArtifact(paths["still_p"], "still"),
        ),
        base_dir=base,
        stills=(still_pair(paths),),
        clipping=ClippingReport(1.0, 1.2),
    )
    return replace(parts, **overrides)


def report_for(base: Path, **overrides) -> CctvReportV1:
    return build_report(make_parts(base, **overrides), ReportTools(now=lambda: START, local_tz=LOCAL_TZ))


def limitation_keys(report: CctvReportV1) -> list[str]:
    return [item.key for item in report.limitations]


# --- report.json y el esquema ---


def test_report_json_round_trips_through_the_model(tmp_path: Path) -> None:
    report = report_for(tmp_path)
    text = report_json_text(report)
    assert load_report(text) == report
    payload = json.loads(text)
    assert payload["schemaVersion"] == 1
    assert payload["mode"] == "classic"
    assert payload["aiUsed"] is False


def test_report_json_validates_against_the_published_schema(tmp_path: Path) -> None:
    payload = json.loads(report_json_text(report_for(tmp_path)))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(payload)


def test_published_schema_matches_the_model() -> None:
    # Si falla: python -c "from app.services.cctv_report_model import write_report_schema; write_report_schema()"
    assert SCHEMA_PATH.read_text(encoding="utf-8") == report_schema_text()


def test_schema_uses_the_json_field_names() -> None:
    properties = report_schema()["properties"]
    assert {"schemaVersion", "aiUsed", "inputs", "steps", "limitations", "guidelines", "disclaimer"} <= set(
        properties
    )


def test_report_records_received_at_hashes_and_the_ffmpeg_build(tmp_path: Path) -> None:
    payload = json.loads(report_json_text(report_for(tmp_path)))
    source = payload["inputs"][0]
    assert source["sha256"] == SOURCE_SHA
    assert source["verifiedCopySha256"] == SOURCE_SHA
    assert source["receivedAt"] == {"utc": "2026-09-25T14:00:00.000Z", "local": "2026-09-25T09:00:00.000-05:00"}
    assert source["verifiedCopyPath"] == "01_original/clip.mp4"
    assert source["workingCopy"]["sha256"] == WORK_SHA
    assert source["frameIndex"]["csvSha256"] == CSV_SHA
    environment = payload["environment"]
    assert environment["ffmpegSha256"] == FFMPEG_SHA
    assert environment["ffmpeg"]["version"].startswith("ffmpeg version N-123588")
    assert environment["cpu"]["extensions"] == ["avx2", "avx512"]
    assert payload["generatedAt"] == {"utc": "2026-09-25T14:00:00.000Z", "local": "2026-09-25T09:00:00.000-05:00"}


def test_outputs_carry_relative_paths_and_real_hashes(tmp_path: Path) -> None:
    report = report_for(tmp_path)
    analysis = report.outputs[0]
    assert analysis.path == "analysis.mkv"
    assert analysis.sha256 == hashlib.sha256(b"analysis bytes").hexdigest()
    assert analysis.ffprobe == {"format_name": "matroska,webm", "streams": []}
    assert [output.path for output in report.outputs[3:]] == ["stills/original_f3.png", "stills/processed_f3.png"]


def test_trim_records_frames_and_audio_times_from_the_index(tmp_path: Path) -> None:
    trim = report_for(tmp_path).trim
    assert trim is not None
    assert (trim.start_frame, trim.end_frame) == (2, 5)
    assert trim.audio_start_seconds == pytest.approx(2 / FPS)
    assert trim.audio_end_seconds == pytest.approx(6 / FPS)


def test_stills_keep_pts_timecode_and_framehash(tmp_path: Path) -> None:
    pair = report_for(tmp_path).stills[0]
    assert pair.original.file == "stills/original_f3.png"
    assert pair.original.timecode == "00:00:00.120"
    assert pair.processed.framehash_sha256 == FRAME_SHA


# --- Pasos en lenguaje llano ---


def test_steps_use_the_plain_language_catalog_descriptions(tmp_path: Path) -> None:
    report = report_for(tmp_path)
    assert [step.id for step in report.steps] == ["trim", "deblock", "denoise"]
    for step in report.steps:
        catalog = next(spec for spec in step_spec(step.id).filters if spec.name == step.filter)
        assert step.description == catalog.description
        assert step.doc_url == catalog.doc_url
        assert step.doc_url.startswith("https://ffmpeg.org/ffmpeg-filters.html#")
        assert step.category == "classic"


def test_steps_list_every_parameter_with_explicit_defaults(tmp_path: Path) -> None:
    denoise = report_for(tmp_path).steps[2]
    assert denoise.parameters == {"luma_spatial": 3.0, "chroma_spatial": 3.0, "luma_tmp": 6.0, "chroma_tmp": 4.5}


def test_classic_steps_carry_the_exact_argv_and_times_of_their_process(tmp_path: Path) -> None:
    report = report_for(tmp_path)
    for step in report.steps:
        assert step.argv == ["ffmpeg", "-i", "work.mkv", "analysis.mkv"]
        assert step.exit_code == 0
        assert step.started_at.utc == "2026-09-25T14:00:00.000Z"
        assert step.ended_at.local == "2026-09-25T09:00:03.000-05:00"
        assert step.engine.name == "ffmpeg"
        assert step.engine.version == CAPS.version
    assert [process.label for process in report.processes] == ["analysis copy (FFV1)", "viewing copy (H.264)"]


# --- Limitaciones ---


def test_classic_report_always_states_hash_scope_build_and_lossy_copies(tmp_path: Path) -> None:
    keys = limitation_keys(report_for(tmp_path))
    assert {HASH_SCOPE.key, SAME_BUILD.key, LOSSY_COPIES.key} <= set(keys)
    assert AI_DETAIL.key not in keys
    assert REMUXED_PROPRIETARY.key not in keys
    assert CLOCK_OFFSET.key not in keys
    assert len(keys) == len(set(keys))


def test_geometry_and_new_pixel_limitations_follow_the_steps(tmp_path: Path) -> None:
    request = [
        *CLASSIC_REQUEST,
        {"id": "crop", "params": {"w": 32, "h": 24, "x": 0, "y": 0}},
        {"id": "scale", "params": {"factor": 2, "flags": "lanczos"}},
    ]
    keys = limitation_keys(report_for(tmp_path / "lanczos", request=request))
    assert GEOMETRY_CHANGED.key in keys and NEW_PIXEL_VALUES.key in keys

    neighbor = [*CLASSIC_REQUEST, {"id": "scale", "params": {"factor": 2, "flags": "neighbor"}}]
    keys = limitation_keys(report_for(tmp_path / "neighbor", request=neighbor))
    assert GEOMETRY_CHANGED.key in keys and NEW_PIXEL_VALUES.key not in keys

    assert GEOMETRY_CHANGED.key not in limitation_keys(report_for(tmp_path / "none"))


def test_proprietary_container_adds_the_remux_limitation(tmp_path: Path) -> None:
    report = report_for(tmp_path, source=make_source(HIKVISION_PS))
    assert REMUXED_PROPRIETARY.key in limitation_keys(report)


def test_user_reported_clock_offset_is_declared(tmp_path: Path) -> None:
    acquisition = Acquisition(clock_offset_seconds=-42.5, clock_offset_method="Compared with a phone clock")
    report = report_for(tmp_path, acquisition=acquisition)
    assert CLOCK_OFFSET.key in limitation_keys(report)
    assert report.acquisition.clock_offset_seconds == -42.5


EXPORT_FACTS = {"recorderSerial": "L12345678", "exportMethod": "USB export from the recorder menu",
                "exportDate": "2026-09-20"}


def test_acquisition_records_the_recorder_serial_and_how_and_when_it_was_exported(tmp_path: Path) -> None:
    report = report_for(tmp_path, acquisition=Acquisition.model_validate(EXPORT_FACTS))
    data = json.loads(report_json_text(report))
    assert {key: data["acquisition"][key] for key in EXPORT_FACTS} == EXPORT_FACTS


def test_acquisition_export_fields_are_optional() -> None:
    acquisition = Acquisition()
    assert (acquisition.recorder_serial, acquisition.export_method, acquisition.export_date) == (None, None, None)


@pytest.mark.parametrize("value", ["20/09/2026", "2026-9-20", "2026-02-30", "2026-09-20T10:00", " 2026-09-20", ""])
def test_acquisition_rejects_an_export_date_that_is_not_a_calendar_date(value: str) -> None:
    with pytest.raises(ValidationError):
        Acquisition(export_date=value)


def test_acquisition_caps_the_export_text_like_the_other_recorder_fields() -> None:
    with pytest.raises(ValidationError):
        Acquisition(recorder_serial="x" * 201)
    with pytest.raises(ValidationError):
        Acquisition(export_method="x" * 201)


def test_acquisition_export_fields_are_in_the_published_schema() -> None:
    properties = report_schema()["$defs"]["Acquisition"]["properties"]
    assert {"recorderSerial", "exportMethod", "exportDate"} <= set(properties)


def test_increased_clipping_adds_limitation_and_warning(tmp_path: Path) -> None:
    report = report_for(tmp_path, clipping=ClippingReport(1.0, 2.0))
    texts = [item.text for item in report.limitations]
    assert "Processing increased clipped pixels by 1.00%" in texts
    assert CLIPPING_INCREASED_KEY in report.warnings
    assert report.clipping.increased is True

    quiet = report_for(tmp_path / "quiet")
    assert CLIPPING_INCREASED_KEY not in limitation_keys(quiet)
    assert CLIPPING_INCREASED_KEY not in quiet.warnings


def test_report_ships_the_fixed_guidelines_and_disclaimer(tmp_path: Path) -> None:
    report = report_for(tmp_path)
    assert report.guidelines == GUIDELINES_TEXT
    assert report.disclaimer == DISCLAIMER_TEXT
    assert "not a certified forensic tool" in report.disclaimer


def test_warnings_collect_the_osd_and_ingest_warnings_once(tmp_path: Path) -> None:
    osd = {**osd_report(OsdSelection((), False, True), (), ()), "warnings": ["cctv.osd.notText", "cctv.osd.notText"]}
    report = report_for(tmp_path, osd=osd, warnings=("cctv.lite",))
    assert report.warnings == ["cctv.osd.notText", "cctv.lite"]


def test_diagnosis_goes_into_the_input_and_its_warnings_into_the_report(tmp_path: Path) -> None:
    diagnosis = CctvDiagnosis(
        interlace=None, blocking=None, blur=None, freezes=[], frame_stats=None,
        suggested_preset="night_ir", warnings=["cctv.diag.lowFps"],
    )
    report = report_for(tmp_path, diagnosis=diagnosis)
    assert report.inputs[0].diagnosis["suggestedPreset"] == "night_ir"
    assert "cctv.diag.lowFps" in report.warnings


# --- Validacion del modelo ---


def test_model_rejects_a_verified_copy_hash_that_differs(tmp_path: Path) -> None:
    payload = json.loads(report_json_text(report_for(tmp_path)))
    payload["inputs"][0]["verifiedCopySha256"] = "e" * 64
    with pytest.raises(ValidationError, match="verifiedCopySha256"):
        CctvReportV1.model_validate(payload)


def test_model_rejects_ai_flag_that_disagrees_with_the_mode(tmp_path: Path) -> None:
    payload = json.loads(report_json_text(report_for(tmp_path)))
    payload["aiUsed"] = True
    with pytest.raises(ValidationError, match="aiUsed"):
        CctvReportV1.model_validate(payload)


def test_model_rejects_ai_steps_in_the_classic_lane(tmp_path: Path) -> None:
    payload = json.loads(report_json_text(report_for(tmp_path)))
    payload["steps"][0]["category"] = "ai"
    with pytest.raises(ValidationError, match="classic lane"):
        CctvReportV1.model_validate(payload)


@pytest.mark.parametrize("path", ["../escape.mkv", "/abs.mkv", "C:/x.mkv", "a\\b.mkv", "a//b.mkv", "./a.mkv", ""])
def test_model_rejects_unsafe_output_paths(tmp_path: Path, path: str) -> None:
    payload = json.loads(report_json_text(report_for(tmp_path)))
    payload["outputs"][0]["path"] = path
    with pytest.raises(ValidationError):
        CctvReportV1.model_validate(payload)


def test_model_rejects_unknown_fields_and_bad_hashes(tmp_path: Path) -> None:
    payload = json.loads(report_json_text(report_for(tmp_path)))
    with pytest.raises(ValidationError):
        CctvReportV1.model_validate({**payload, "extra": 1})
    payload["environment"]["ffmpegSha256"] = "F" * 64
    with pytest.raises(ValidationError):
        CctvReportV1.model_validate(payload)


def test_host_facts_describe_this_machine() -> None:
    host = host_facts()
    assert host.os and host.cpu_model and host.python.count(".") == 2
    assert host.gpus == () and host.onnx_runtime is None


def test_case_fields_are_bounded(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        CaseInfo(case_label="x" * 201)


# --- report.html ---


def test_html_escapes_user_text(tmp_path: Path) -> None:
    case = CaseInfo(case_label='<script>alert("x")</script>', operator_name="O'Brien & <b>", notes="</td><img src=x>")
    html_text = render_report_html(report_for(tmp_path, case=case))
    assert "<script" not in html_text
    assert "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;" in html_text
    assert "O&#x27;Brien &amp; &lt;b&gt;" in html_text
    assert "<img src=x>" not in html_text


REPRODUCED = ReproductionOf.model_validate(
    {"generatedAt": {"utc": "2026-09-01T10:00:00Z", "local": "2026-09-01 05:00:00 -05:00"},
     "reportSha256": "ab" * 32, "upflowVersion": "0.80.0"}
)  # fmt: skip


def test_a_reproduction_says_which_report_it_reproduces_in_its_json(tmp_path: Path) -> None:
    case = CaseInfo(case_label="Case 7", operator_name="Ana", reproduction_of=REPRODUCED)
    text = report_json_text(report_for(tmp_path, case=case))

    block = json.loads(text)["case"]["reproductionOf"]
    assert block == {"generatedAt": REPRODUCED.generated_at.model_dump(), "reportSha256": "ab" * 32,
                     "upflowVersion": "0.80.0"}  # fmt: skip
    jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))).validate(json.loads(text))
    assert load_report(text).case.reproduction_of == REPRODUCED


def test_a_report_that_is_not_a_reproduction_is_written_as_before(tmp_path: Path) -> None:
    # Sin la clave, un informe viejo re-serializado da los mismos bytes: su SHA-256 sigue siendo el de SHA256SUMS.
    text = report_json_text(report_for(tmp_path))

    assert "reproductionOf" not in text


def test_html_says_up_front_that_the_run_reproduces_an_earlier_report(tmp_path: Path) -> None:
    case = CaseInfo(operator_name="Ana", reproduction_of=REPRODUCED)
    html_text = render_report_html(report_for(tmp_path, case=case))

    header = html_text.split("</header>")[0]
    assert "Reproduction of an earlier report" in header
    assert "ab" * 32 in html_text and "2026-09-01T10:00:00Z" in html_text and "0.80.0" in html_text


def test_html_of_an_original_run_does_not_mention_a_reproduction(tmp_path: Path) -> None:
    assert "Reproduction of" not in render_report_html(report_for(tmp_path))


def test_html_shows_the_acquisition_export_fields_escaped(tmp_path: Path) -> None:
    acquisition = Acquisition.model_validate({**EXPORT_FACTS, "exportMethod": "Web client <v2> & USB"})
    html_text = render_report_html(report_for(tmp_path, acquisition=acquisition))
    assert "Recorder serial number" in html_text and "L12345678" in html_text
    assert "Export method" in html_text and "Web client &lt;v2&gt; &amp; USB" in html_text
    assert "Export date" in html_text and "2026-09-20" in html_text


def test_html_escapes_the_original_file_name(tmp_path: Path) -> None:
    source = replace(make_source(), original_name='cam"1<2>.mp4')
    html_text = render_report_html(report_for(tmp_path, source=source))
    assert 'cam"1<2>' not in html_text
    assert "cam&quot;1&lt;2&gt;.mp4" in html_text


def test_html_has_no_external_resources(tmp_path: Path) -> None:
    html_text = render_report_html(report_for(tmp_path)).lower()
    assert "<script" not in html_text
    assert "<link" not in html_text
    assert "@import" not in html_text
    assert 'src="http' not in html_text
    assert "url(" not in html_text


def test_html_leads_with_the_original_hash_and_ai_verdict(tmp_path: Path) -> None:
    html_text = render_report_html(report_for(tmp_path))
    header = html_text.split("</header>")[0]
    assert "clip.mp4" in header
    assert SOURCE_SHA in header
    assert "Classic filters (no AI)" in header
    assert "AI used: NO" in header


def test_html_shows_plain_steps_with_folded_argv_outputs_stills_and_footer(tmp_path: Path) -> None:
    report = report_for(tmp_path)
    html_text = render_report_html(report)
    for step in report.steps:
        assert step.description.replace("'", "&#x27;") in html_text
        assert f'href="{step.doc_url}"' in html_text
    assert "<details><summary>Command: analysis copy (FFV1)" in html_text
    assert "<pre>ffmpeg -i work.mkv analysis.mkv</pre>" in html_text
    assert report.outputs[0].sha256 in html_text
    assert '<img src="stills/original_f3.png"' in html_text
    assert HASH_SCOPE.text in html_text
    footer = html_text.split("<footer>")[1]
    assert GUIDELINES_TEXT in footer
    assert DISCLAIMER_TEXT in footer


def test_write_report_writes_json_and_html(tmp_path: Path) -> None:
    report = report_for(tmp_path)
    json_path, html_path = write_report(report, tmp_path, render_report_html)
    assert load_report(json_path.read_text(encoding="utf-8")) == report
    assert html_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


# --- SHA256SUMS.txt ---


def test_sha256sums_verifies_with_hashlib(tmp_path: Path) -> None:
    paths = make_files(tmp_path)
    relative = [path.relative_to(tmp_path).as_posix() for path in paths.values()]
    sums = write_sha256sums(tmp_path, relative)
    assert sums.name == SHA256SUMS_NAME
    text = sums.read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"{hashlib.sha256(b'original bytes').hexdigest()} *01_original/clip.mp4"
    entries = parse_sha256sums(text)
    assert [entry.path for entry in entries] == relative
    for entry in entries:
        assert hashlib.sha256((tmp_path / entry.path).read_bytes()).hexdigest() == entry.sha256


def test_sha256sums_lists_each_file_once(tmp_path: Path) -> None:
    make_files(tmp_path)
    sums = write_sha256sums(tmp_path, ["analysis.mkv", "analysis.mkv"])
    assert len(sums.read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "C:/Windows/win.ini", "a\\b"])
def test_sha256sums_refuses_paths_outside_the_package(tmp_path: Path, path: str) -> None:
    with pytest.raises(ValueError):
        write_sha256sums(tmp_path, [path])


def test_parse_sha256sums_rejects_malformed_lines() -> None:
    with pytest.raises(ValueError):
        parse_sha256sums("not a checksum line\n")
    with pytest.raises(ValueError):
        parse_sha256sums(f"{'a' * 63} *file.bin\n")


# --- Carril IA: cada paso con su proceso y su motor ---

AI_REQUEST = [
    {"id": "ai_deblock", "params": {"strength": 40}},
    {"id": "levels", "params": {"filter": "eq", "gamma": 1.2}},
]
DRUNET = EngineInfo(name="onnxruntime", version="1.24.4", model_name="DRUNet", model_sha256="e" * 64, model_license="MIT")


def ai_parts(base: Path, **overrides) -> ReportParts:
    frames_run = ProcessRun("AI lane frames", (), START, START + timedelta(seconds=9), 0, 4, 4)
    encode_run = ProcessRun("AI lane encode", ("ffmpeg", "-vf", "LABEL", "enhanced.mp4"), START,
                            START + timedelta(seconds=9), 0, 4, 4)
    viewing = base / "viewing.mp4"
    return make_parts(
        base,
        mode="ai-visual",
        chain=steps_from_request(AI_REQUEST, "ai"),
        chain_run=encode_run,
        processes=(frames_run, encode_run),
        outputs=(OutputArtifact(viewing, "ai-visualization", ai_applied=True, visible_label="AI-ENHANCED"),),
        stills=(),
        clipping=None,
        step_runs={"ai_deblock": StepRun(frames_run, DRUNET)},
        **overrides,
    )


def test_ai_steps_carry_the_model_engine_and_the_process_that_ran_them(tmp_path: Path) -> None:
    report = build_report(ai_parts(tmp_path), ReportTools(now=lambda: START, local_tz=LOCAL_TZ))

    deblock, levels, label = report.steps
    assert (deblock.id, deblock.category, deblock.argv) == ("ai_deblock", "ai", [])
    assert (deblock.engine.model_sha256, deblock.engine.model_license) == ("e" * 64, "MIT")
    assert (levels.engine.name, levels.argv) == ("ffmpeg", ["ffmpeg", "-vf", "LABEL", "enhanced.mp4"])
    assert (label.id, label.category, label.argv) == ("ai_label", "label", levels.argv)


def test_ai_report_marks_the_visualization_and_a_generative_upscaler(tmp_path: Path) -> None:
    parts = ai_parts(tmp_path, extra_limitations=(GENERATIVE_UPSCALE,))

    report = build_report(parts, ReportTools(now=lambda: START, local_tz=LOCAL_TZ))

    (output,) = report.outputs
    assert (output.role, output.ai_applied, output.visible_label) == ("ai-visualization", True, "AI-ENHANCED")
    keys = limitation_keys(report)
    assert AI_DETAIL.key in keys and GENERATIVE_UPSCALE.key in keys
    assert LOSSY_COPIES.key not in keys and SAME_BUILD.key not in keys
    jsonschema.validate(json.loads(report_json_text(report)), report_schema())
