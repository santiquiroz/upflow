from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas_cctv import CctvJobRequest
from app.services.cctv_chain import CctvChainError
from app.services.cctv_report_model import CctvReportV1
from app.services.cctv_reproduce import (
    NOT_CLASSIC,
    OTHER_BUILD,
    OTHER_CPU,
    OTHER_VERSION,
    REPORT_INVALID,
    SOURCE_MISMATCH,
    STEPS_MISMATCH,
    UNSUPPORTED,
    VERSION_IN_FILES,
    check_reproducible,
    check_same_source,
    compare_reproduction,
    expected_facts,
    job_body_from_report,
    parse_untrusted_report,
    preflight_warnings,
)
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from test_cctv_report import CAPS, FFMPEG_SHA, FRAME_SHA, SOURCE_SHA, report_for

TOKEN = "session0token1"


def report_json(base: Path, **overrides) -> dict:
    return report_for(base, **overrides).model_dump(mode="json", by_alias=True)


def keyed(call) -> str:
    with pytest.raises(CctvChainError) as caught:
        call()
    return caught.value.code


def with_steps(payload: dict, steps: list[dict]) -> dict:
    return {**payload, "steps": steps}


def changed_step(payload: dict, step_id: str, **params) -> list[dict]:
    return [
        {**step, "parameters": {**step["parameters"], **params}} if step["id"] == step_id else step
        for step in payload["steps"]
    ]


# --- Informe no confiable ---


def test_a_reproduce_report_round_trips_through_the_untrusted_parser(tmp_path: Path) -> None:
    report = parse_untrusted_report(report_json(tmp_path))

    assert isinstance(report, CctvReportV1) and report.inputs[0].sha256 == SOURCE_SHA


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: {**payload, "schemaVersion": 2},
        lambda payload: {**payload, "shellCommand": "calc.exe"},
        lambda payload: {**payload, "steps": "denoise"},
        lambda payload: {**payload, "inputs": []},
    ],
)
def test_a_reproduce_report_that_is_not_an_upflow_report_is_rejected(tmp_path: Path, mutate) -> None:
    payload = mutate(report_json(tmp_path))

    assert keyed(lambda: parse_untrusted_report(payload)) == REPORT_INVALID


def test_reproduce_only_takes_classic_reports(tmp_path: Path) -> None:
    payload = report_json(tmp_path)
    ai_report = CctvReportV1.model_construct(**{**parse_untrusted_report(payload).__dict__, "mode": "ai-visual"})

    assert keyed(lambda: check_reproducible(ai_report)) == NOT_CLASSIC


def test_reproduce_rejects_reports_with_more_than_one_original(tmp_path: Path) -> None:
    payload = report_json(tmp_path)
    report = parse_untrusted_report({**payload, "inputs": payload["inputs"] * 2})

    assert keyed(lambda: check_reproducible(report)) == UNSUPPORTED


def test_reproduce_needs_the_same_original_file(tmp_path: Path) -> None:
    report = parse_untrusted_report(report_json(tmp_path))

    check_same_source(report, SOURCE_SHA)
    assert keyed(lambda: check_same_source(report, "0" * 64)) == SOURCE_MISMATCH


# --- Lista blanca de pasos ---


def test_reproduce_turns_the_report_into_an_ordinary_clarify_request(tmp_path: Path) -> None:
    report = parse_untrusted_report(report_json(tmp_path))

    body = job_body_from_report(report, TOKEN)

    request = CctvJobRequest.model_validate(body)
    assert request.task == "clarify" and request.token == TOKEN and request.trim == (2, 5)
    assert [step.id for step in request.steps] == ["deblock", "denoise"]
    assert request.steps[1].params == {"filter": "hqdn3d", "luma_spatial": 3.0, **{
        key: value for key, value in report.steps[2].parameters.items() if key != "luma_spatial"
    }}  # fmt: skip
    assert request.still_frames == [3] and request.no_osd and request.osd_boxes == []
    assert request.acquisition == {"recorderMake": "HiLook"} and request.case_label == "Case 7"


def test_reproduce_never_carries_the_report_argv_into_the_request(tmp_path: Path) -> None:
    payload = report_json(tmp_path)
    payload["steps"][1]["argv"] = ["cmd.exe", "/c", "calc.exe"]

    body = job_body_from_report(parse_untrusted_report(payload), TOKEN)

    assert "calc.exe" not in repr(body)


@pytest.mark.parametrize(
    "steps",
    [
        lambda payload: changed_step(payload, "denoise", luma_spatial=999.0),
        lambda payload: changed_step(payload, "denoise", filter_graph="movie=secret.txt"),
        lambda payload: [{**payload["steps"][1], "filter": "nlmeans_opencl"}],
        lambda payload: [{**payload["steps"][1], "id": "minterpolate"}],
        lambda payload: [{**payload["steps"][1], "category": "ai", "id": "ai_deblock"}],
    ],
)
def test_reproduce_rejects_steps_outside_the_whitelist(tmp_path: Path, steps) -> None:
    payload = report_json(tmp_path)
    untrusted = with_steps(payload, steps(payload))

    assert keyed(lambda: job_body_from_report(parse_untrusted_report(untrusted), TOKEN)).startswith("cctv.error.")


def test_reproduce_rejects_steps_with_missing_parameters(tmp_path: Path) -> None:
    payload = report_json(tmp_path)
    denoise = next(step for step in payload["steps"] if step["id"] == "denoise")
    trimmed = {**denoise, "parameters": {"luma_spatial": 3.0}}
    report = parse_untrusted_report(with_steps(payload, [trimmed]))

    assert keyed(lambda: job_body_from_report(report, TOKEN)) == STEPS_MISMATCH


def test_reproduce_rejects_an_osd_step_without_boxes(tmp_path: Path) -> None:
    payload = report_json(tmp_path)
    osd = {**payload["steps"][0], "id": "osd_protect", "filter": "osd_restore", "parameters": {}}
    report = parse_untrusted_report(with_steps(payload, [*payload["steps"], osd]))

    assert keyed(lambda: job_body_from_report(report, TOKEN)) == STEPS_MISMATCH


# --- Esperado y avisos previos ---


def test_reproduce_expects_the_files_reproduce_cmd_checks_and_the_frame_hashes(tmp_path: Path) -> None:
    expected = expected_facts(parse_untrusted_report(report_json(tmp_path)))

    assert set(expected["outputs"]) == {"analysis-lossless", "viewing-copy"}
    assert expected["stills"] == [{"frame": 3, "original": FRAME_SHA, "processed": FRAME_SHA}]
    assert expected["sourceSha256"] == SOURCE_SHA and expected["ffmpegSha256"] == FFMPEG_SHA
    assert expected["upflowVersion"] == "0.80.0" and expected["cpuExtensions"] == ["avx2", "avx512"]


def test_reproduce_warns_before_running_on_another_version_build_or_cpu(tmp_path: Path) -> None:
    expected = expected_facts(parse_untrusted_report(report_json(tmp_path)))
    other = FfmpegCapabilities("e" * 64, CAPS.version, CAPS.configuration, CAPS.filters, CAPS.encoders, ("sse4",))

    assert preflight_warnings(expected, CAPS, "0.80.0") == []
    assert preflight_warnings(expected, other, "0.81.0") == [OTHER_VERSION, OTHER_BUILD, OTHER_CPU]


# --- Comparacion ---


def test_reproduce_of_an_identical_run_matches_every_check(tmp_path: Path) -> None:
    report = parse_untrusted_report(report_json(tmp_path))

    comparison = compare_reproduction(expected_facts(report), report)

    assert comparison.identical and comparison.frames_identical and comparison.notes() == []
    kinds = {check.kind for check in comparison.checks}
    assert kinds == {"file", "version", "build", "framehash", "output"}


def test_reproduce_on_another_version_keeps_the_frames_and_explains_the_files(tmp_path: Path) -> None:
    expected = expected_facts(parse_untrusted_report(report_json(tmp_path / "a")))
    expected["outputs"] = {**expected["outputs"], "viewing-copy": "9" * 64}
    produced = report_for(tmp_path / "b", app_version="0.81.0")

    result = compare_reproduction(expected, produced).to_json()

    failed = {(check["kind"], check["name"]) for check in result["checks"] if not check["match"]}
    assert failed == {("version", "upflow"), ("output", "viewing-copy")} and not result["identical"]
    assert result["framesIdentical"] is True and result["notes"] == [VERSION_IN_FILES]


def test_reproduce_reports_a_changed_frame_and_a_missing_still(tmp_path: Path) -> None:
    report = parse_untrusted_report(report_json(tmp_path))
    expected = expected_facts(report)
    expected["stills"] = [
        {"frame": 3, "original": FRAME_SHA, "processed": "9" * 64},
        {"frame": 8, "original": FRAME_SHA, "processed": FRAME_SHA},
    ]

    comparison = compare_reproduction(expected, report)

    framehashes = {check.name: check for check in comparison.of_kind("framehash")}
    assert framehashes["frame 3 original"].match and not framehashes["frame 3 processed"].match
    assert framehashes["frame 8 processed"].actual is None
    assert comparison.frames_identical is False and not comparison.identical


def test_reproduce_notes_another_build(tmp_path: Path) -> None:
    report = parse_untrusted_report(report_json(tmp_path))
    expected = {**expected_facts(report), "ffmpegSha256": "e" * 64}

    comparison = compare_reproduction(expected, report)

    assert comparison.notes() == [OTHER_BUILD] and not comparison.identical
