"""Reproduce dentro de la app (spec §4.13, P4-REPRODUCE).

El `report.json` que sube el usuario es un input no confiable: se valida contra
el modelo del informe y sus pasos vuelven a pasar por la lista blanca del
catalogo (`steps_from_request`); de el solo se toman ids, nombres de filtro y
parametros, nunca los `argv`. Con eso se arma el pedido de un job `clarify`
comun, y al terminar se compara el informe nuevo con lo esperado: el archivo de
entrada, el `framehash` de los cuadros exportados, los archivos que promete
`reproduce.cmd`, la build de ffmpeg y la version de Upflow.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from app.models import CctvOptions, ReproductionSource
from app.schemas_cctv import CctvJobRequest, CctvReproduceRequest, cctv_options
from app.services.cctv_artifacts import resolve_artifact
from app.services.cctv_chain import FILTER_KEY, CctvChainError, ResolvedStep, steps_from_request
from app.services.cctv_report import load_report, report_json_text
from app.services.cctv_report_model import CctvReportV1, ReportStep
from app.services.cctv_session import load_session
from app.services.ffmpeg_capabilities import FfmpegCapabilities

REPORT_INVALID = "cctv.error.reproduceReportInvalid"
NOT_CLASSIC = "cctv.error.reproduceNotClassic"
UNSUPPORTED = "cctv.error.reproduceUnsupported"
STEPS_MISMATCH = "cctv.error.reproduceStepsMismatch"
SOURCE_MISMATCH = "cctv.error.reproduceSourceMismatch"
NOT_A_REPRODUCTION = "cctv.error.notAReproduction"

OTHER_VERSION = "cctv.reproduce.otherVersion"
OTHER_BUILD = "cctv.reproduce.otherBuild"
OTHER_CPU = "cctv.reproduce.otherCpu"
VERSION_IN_FILES = "cctv.reproduce.versionInFiles"

REPRODUCE_TASK = "clarify"
OSD_STEP = "osd_protect"
TRIM_STEP = "trim"
# Los mismos archivos que compara reproduce.cmd: el comparativo y los PNG no se prometen bit a bit.
REPRODUCIBLE_ROLES = ("stabilization-motion", "analysis-lossless", "viewing-copy")
METADATA_KEY = "reproduce"

CheckKind = Literal["file", "framehash", "output", "build", "version"]


# --- Informe no confiable ---


def parse_untrusted_report(payload: Mapping[str, Any]) -> CctvReportV1:
    try:
        return CctvReportV1.model_validate(dict(payload))
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "report"
        raise CctvChainError(REPORT_INVALID, f"The report is not a valid Upflow report: {where}: {first['msg']}") from exc


def check_reproducible(report: CctvReportV1) -> None:
    if report.mode != "classic":
        raise CctvChainError(NOT_CLASSIC, "Only classic-filter reports can be reproduced; AI results are not repeatable.")
    if report.roi is not None:
        raise CctvChainError(UNSUPPORTED, "Multi-frame still reports can't be reproduced here yet.")
    if len(report.inputs) != 1:
        raise CctvChainError(UNSUPPORTED, "The report must name exactly one original file.")


def source_sha256(report: CctvReportV1) -> str:
    return report.inputs[0].sha256


def check_same_source(report: CctvReportV1, loaded_sha256: str) -> None:
    if loaded_sha256 != source_sha256(report):
        raise CctvChainError(
            SOURCE_MISMATCH,
            f"The loaded video is not the original in the report (sha256 {loaded_sha256}, "
            f"expected {source_sha256(report)}).",
        )


# --- Pasos: lista blanca ---


def raw_report_step(step: ReportStep) -> dict[str, Any]:
    return {"id": step.id, "params": {**step.parameters, FILTER_KEY: step.filter}}


def step_facts(steps: Sequence[ResolvedStep]) -> list[tuple[str, str, dict[str, Any]]]:
    return [(step.id, step.filter, dict(step.params)) for step in steps]


def recorded_facts(steps: Sequence[ReportStep]) -> list[tuple[str, str, dict[str, Any]]]:
    return [(step.id, step.filter, dict(step.parameters)) for step in steps]


def resolved_report_steps(report: CctvReportV1) -> tuple[ResolvedStep, ...]:
    return steps_from_request([raw_report_step(step) for step in report.steps], "classic")


def check_osd_step(report: CctvReportV1) -> None:
    has_step = any(step.id == OSD_STEP for step in report.steps)
    if has_step != bool(report.osd.boxes):
        raise CctvChainError(STEPS_MISMATCH, "The OSD protection step and the OSD boxes of the report disagree.")


def check_report_steps(report: CctvReportV1) -> tuple[ResolvedStep, ...]:
    # Resolver de nuevo debe dar exactamente lo que dice el informe: si no, el informe no es de esta cadena.
    resolved = resolved_report_steps(report)
    if step_facts(resolved) != recorded_facts(report.steps):
        raise CctvChainError(STEPS_MISMATCH, "The report's steps don't match what Upflow's filter catalog resolves.")
    check_osd_step(report)
    return resolved


# --- Pedido del job ---


def trim_of(steps: Sequence[ResolvedStep]) -> list[int] | None:
    trim = next((step for step in steps if step.id == TRIM_STEP), None)
    if trim is None:
        return None
    return [int(trim.params["start_frame"]), int(trim.params["end_frame"])]


def chosen_steps(steps: Sequence[ResolvedStep]) -> list[dict[str, Any]]:
    kept = (step for step in steps if step.id not in (TRIM_STEP, OSD_STEP))
    return [{"id": step.id, "params": {**step.params, FILTER_KEY: step.filter}} for step in kept]


def case_fields(report: CctvReportV1, operator_name: str | None) -> dict[str, Any]:
    # El caso es el mismo; el operador no: la reproduccion la firma quien la corre.
    fields = {"caseLabel": report.case.case_label, "operatorName": operator_name}
    return {name: value for name, value in fields.items() if value is not None}


def job_body_from_report(report: CctvReportV1, token: str, operator_name: str | None = None) -> dict[str, Any]:
    steps = check_report_steps(report)
    return {
        "token": token,
        "task": REPRODUCE_TASK,
        "steps": chosen_steps(steps),
        "osdBoxes": [list(box) for box in report.osd.boxes],
        "osdBoxesConfirmed": report.osd.osd_boxes_confirmed,
        "noOsd": report.osd.no_osd,
        "trim": trim_of(steps),
        "stillFrames": [pair.frame for pair in report.stills],
        "acquisition": report.acquisition.model_dump(by_alias=True, exclude_none=True),
        **case_fields(report, operator_name),
    }


def checked_job_request(body: dict[str, Any]) -> CctvJobRequest:
    try:
        return CctvJobRequest.model_validate(body)
    except ValidationError as exc:
        raise CctvChainError(REPORT_INVALID, f"The report asks for more than a job allows: {exc.errors()[0]['msg']}") from exc


def reproduction_source(report: CctvReportV1) -> ReproductionSource:
    written = report_json_text(report).encode("utf-8")
    return ReproductionSource(
        generated_at_utc=report.generated_at.utc,
        generated_at_local=report.generated_at.local,
        report_sha256=hashlib.sha256(written).hexdigest(),
        upflow_version=report.upflow.version,
    )


@dataclass(frozen=True, slots=True)
class ReproduceRequest:
    report: CctvReportV1
    job: CctvJobRequest

    def options(self) -> CctvOptions:
        return replace(cctv_options(self.job), reproduction_of=reproduction_source(self.report))


def reproduce_request(body: CctvReproduceRequest, work_root: Path) -> ReproduceRequest:
    report = parse_untrusted_report(body.report)
    check_reproducible(report)
    check_same_source(report, load_session(work_root, body.token).record.sha256)
    job_body = job_body_from_report(report, body.token, body.operator_name)
    return ReproduceRequest(report, checked_job_request(job_body))


# --- Lo esperado, guardado en el job ---


def expected_outputs(report: CctvReportV1) -> dict[str, str]:
    return {output.role: output.sha256 for output in report.outputs if output.role in REPRODUCIBLE_ROLES}


def expected_stills(report: CctvReportV1) -> list[dict[str, Any]]:
    return [
        {"frame": pair.frame, "original": pair.original.framehash_sha256, "processed": pair.processed.framehash_sha256}
        for pair in report.stills
    ]


def expected_facts(report: CctvReportV1) -> dict[str, Any]:
    environment = report.environment
    return {
        "reportGeneratedAt": report.generated_at.utc,
        "sourceSha256": source_sha256(report),
        "upflowVersion": report.upflow.version,
        "ffmpegVersion": environment.ffmpeg.version,
        "ffmpegSha256": environment.ffmpeg_sha256,
        "cpuExtensions": list(environment.cpu.extensions),
        "outputs": expected_outputs(report),
        "stills": expected_stills(report),
    }


def preflight_warnings(expected: Mapping[str, Any], caps: FfmpegCapabilities, app_version: str) -> list[str]:
    differences = (
        (expected["upflowVersion"] != app_version, OTHER_VERSION),
        (expected["ffmpegSha256"] != caps.binary_sha256, OTHER_BUILD),
        (list(expected["cpuExtensions"]) != list(caps.cpu_extensions), OTHER_CPU),
    )
    return [key for differs, key in differences if differs]


# --- Comparacion ---


@dataclass(frozen=True, slots=True)
class ReproduceCheck:
    kind: CheckKind
    name: str
    expected: str
    actual: str | None

    @property
    def match(self) -> bool:
        return self.expected == self.actual

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "expected": self.expected, "actual": self.actual,
                "match": self.match}  # fmt: skip


@dataclass(frozen=True, slots=True)
class ReproduceComparison:
    checks: tuple[ReproduceCheck, ...]

    def of_kind(self, kind: CheckKind) -> tuple[ReproduceCheck, ...]:
        return tuple(check for check in self.checks if check.kind == kind)

    @property
    def identical(self) -> bool:
        return all(check.match for check in self.checks)

    @property
    def frames_identical(self) -> bool | None:
        frames = self.of_kind("framehash")
        return all(check.match for check in frames) if frames else None

    def notes(self) -> list[str]:
        failed = {check.kind for check in self.checks if not check.match}
        notes = [VERSION_IN_FILES] if {"version", "output"} <= failed else []
        return [*notes, OTHER_BUILD] if "build" in failed else notes

    def to_json(self) -> dict[str, Any]:
        return {
            "identical": self.identical,
            "framesIdentical": self.frames_identical,
            "checks": [check.to_json() for check in self.checks],
            "notes": self.notes(),
        }


def environment_checks(expected: Mapping[str, Any], produced: CctvReportV1) -> list[ReproduceCheck]:
    environment = produced.environment
    return [
        ReproduceCheck("file", "original", expected["sourceSha256"], source_sha256(produced)),
        ReproduceCheck("version", "upflow", expected["upflowVersion"], produced.upflow.version),
        ReproduceCheck("build", "ffmpeg", expected["ffmpegSha256"], environment.ffmpeg_sha256),
        ReproduceCheck("build", "cpuExtensions", " ".join(expected["cpuExtensions"]), " ".join(environment.cpu.extensions)),
    ]


def produced_framehashes(produced: CctvReportV1) -> dict[tuple[int, str], str]:
    return {
        (pair.frame, still.role): still.framehash_sha256
        for pair in produced.stills
        for still in (pair.original, pair.processed)
    }


def framehash_checks(expected: Mapping[str, Any], produced: CctvReportV1) -> list[ReproduceCheck]:
    hashes = produced_framehashes(produced)
    return [
        ReproduceCheck("framehash", f"frame {still['frame']} {role}", still[role], hashes.get((still["frame"], role)))
        for still in expected["stills"]
        for role in ("original", "processed")
    ]


def output_checks(expected: Mapping[str, Any], produced: CctvReportV1) -> list[ReproduceCheck]:
    actual = expected_outputs(produced)
    return [ReproduceCheck("output", role, sha, actual.get(role)) for role, sha in expected["outputs"].items()]


def produced_report(job_dir: Path, outputs: Mapping[str, Any]) -> CctvReportV1:
    path = resolve_artifact(job_dir, "report_json", outputs)
    return load_report(path.read_text(encoding="utf-8"))


def compare_reproduction(expected: Mapping[str, Any], produced: CctvReportV1) -> ReproduceComparison:
    return ReproduceComparison(
        (
            *environment_checks(expected, produced),
            *framehash_checks(expected, produced),
            *output_checks(expected, produced),
        )
    )
