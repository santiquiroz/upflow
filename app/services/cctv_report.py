"""Arma el informe CCTV (`report.json`, `report.html`) y `SHA256SUMS.txt` (spec §4.11).

Todo entra por parametros: los hechos de la ingesta, la cadena resuelta, los
procesos que corrieron con sus `argv` y tiempos, y las salidas. Las
limitaciones se derivan de esos hechos con reglas puras, asi que el informe
no depende de que alguien se acuerde de agregarlas.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any

from app.models import utc_now
from app.services.cctv_chain import (
    CctvChainError,
    FilterSpec,
    Limitation,
    ResolvedStep,
    UNKNOWN_FILTER,
    limitations_for,
    step_spec,
)
from app.services.cctv_clarify_runner import (
    CLIPPING_INCREASED_KEY,
    ClippingReport,
    clipping_limitation,
    frame_time,
    trim_step,
)
from app.services.cctv_frame_index import FrameEntry
from app.services.cctv_ingest import SourceRecord, VerifiedCopy, WorkingIngest, received_at, sha256_file
from app.services.cctv_report_model import (
    Acquisition,
    CaseInfo,
    CctvReportV1,
    ClippingInfo,
    CpuInfo,
    EngineInfo,
    EnvironmentInfo,
    FfmpegBuild,
    FrameHashesInfo,
    GpuInfo,
    InputFile,
    LimitationInfo,
    OnnxRuntimeInfo,
    OsdInfo,
    OutputFile,
    OutputRole,
    ProcessInfo,
    ReportMode,
    ReportStep,
    StillPairInfo,
    Timestamp,
    TrimInfo,
    UpflowInfo,
    check_relative_path,
    check_sha256,
)
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.frame_export import StillPair
from app.services.video_analysis import CctvDiagnosis

REPORT_JSON_NAME = "report.json"
REPORT_HTML_NAME = "report.html"
SHA256SUMS_NAME = "SHA256SUMS.txt"

GUIDELINES_KEY = "cctv.report.guidelines"
GUIDELINES_TEXT = (
    "The report records what SWGDE and ENFSI guidelines ask a processing report to include: "
    "hashes, every step in order with its settings, and software versions."
)
DISCLAIMER_KEY = "cctv.disclaimer"
DISCLAIMER_TEXT = (
    "Upflow is not a certified forensic tool and has not been validated by a forensic laboratory. "
    "This report documents what Upflow did; it does not certify authenticity or admissibility. "
    "Always preserve and provide the original recording."
)

LANE_NAMES: Mapping[ReportMode, str] = {
    "classic": "Classic filters (no AI)",
    "ai-visual": "AI enhancement (visual only)",
}
PROPRIETARY_CONTAINERS = frozenset({"hikvision_ps", "dahua_dav"})
LOSSY_ROLES = frozenset({"viewing-copy", "comparison"})
FFMPEG_ENGINE = "ffmpeg"

REMUXED_PROPRIETARY = Limitation(
    "cctv.limitation.remuxedProprietary", "Source was remuxed from a proprietary container."
)
LOSSY_COPIES = Limitation(
    "cctv.limitation.lossyCopies",
    "Viewing and comparison copies are lossy (H.264); use analysis.mkv for examination.",
)
HASH_SCOPE = Limitation(
    "cctv.limitation.hashScope",
    "SHA-256 values were computed when Upflow received the file. They detect accidental changes "
    "after that point; they do not prove the recording is authentic or say what happened before it was loaded.",
)
SAME_BUILD = Limitation(
    "cctv.limitation.sameBuild",
    "Identical results need the same ffmpeg build (sha256 above) and the same CPU features.",
)
AI_DETAIL = Limitation("cctv.limitation.aiDetail", "AI model may add detail not present in the source.")
CLOCK_OFFSET = Limitation("cctv.limitation.clockOffsetUserReported", "Recorder clock offset is user-reported.")


# --- Hechos de entrada ---


@dataclass(frozen=True, slots=True)
class ProcessRun:
    label: str
    argv: tuple[str, ...]
    started_at: datetime
    ended_at: datetime
    exit_code: int
    frames_in: int | None = None
    frames_out: int | None = None


@dataclass(frozen=True, slots=True)
class OutputArtifact:
    path: Path
    role: OutputRole
    probe: Mapping[str, Any] | None = None
    ai_applied: bool = False
    visible_label: str | None = None


@dataclass(frozen=True, slots=True)
class HostFacts:
    os: str
    cpu_model: str
    python: str
    gpus: tuple[GpuInfo, ...] = ()
    onnx_runtime: OnnxRuntimeInfo | None = None


@dataclass(frozen=True, slots=True)
class ReportParts:
    mode: ReportMode
    app_version: str
    commit: str | None
    case: CaseInfo
    acquisition: Acquisition
    caps: FfmpegCapabilities
    host: HostFacts
    source: SourceRecord
    verified_copy: VerifiedCopy
    ingest: WorkingIngest
    diagnosis: CctvDiagnosis | None
    osd: Mapping[str, Any]
    chain: tuple[ResolvedStep, ...]
    chain_run: ProcessRun
    processes: tuple[ProcessRun, ...]
    outputs: tuple[OutputArtifact, ...]
    base_dir: Path
    stills: tuple[StillPair, ...] = ()
    clipping: ClippingReport | None = None
    frame_hashes: FrameHashesInfo | None = None
    extra_limitations: tuple[Limitation, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReportTools:
    hash_file: Callable[[Path], str] = sha256_file
    now: Callable[[], datetime] = utc_now
    local_tz: tzinfo | None = None


def host_facts(gpus: Sequence[GpuInfo] = (), onnx_runtime: OnnxRuntimeInfo | None = None) -> HostFacts:
    return HostFacts(
        os=f"{platform.system()} {platform.release()} (build {platform.version()})",
        cpu_model=platform.processor() or platform.machine(),
        python=sys.version.split()[0],
        gpus=tuple(gpus),
        onnx_runtime=onnx_runtime,
    )


# --- Piezas del informe ---


def timestamp(moment: datetime, local_tz: tzinfo | None = None) -> Timestamp:
    stamp = received_at(moment, local_tz)
    return Timestamp(utc=stamp.utc, local=stamp.local)


def relative_to_base(path: Path, base: Path) -> str:
    return check_relative_path(path.resolve().relative_to(base.resolve()).as_posix())


def environment_info(caps: FfmpegCapabilities, host: HostFacts) -> EnvironmentInfo:
    return EnvironmentInfo(
        os=host.os,
        cpu=CpuInfo(model=host.cpu_model, extensions=list(caps.cpu_extensions)),
        gpus=list(host.gpus),
        onnx_runtime=host.onnx_runtime,
        ffmpeg=FfmpegBuild(version=caps.version, configuration=list(caps.configuration), gpl=caps.is_gpl),
        ffmpeg_sha256=caps.binary_sha256,
        python=host.python,
    )


def input_file(
    source: SourceRecord,
    verified: VerifiedCopy,
    ingest: WorkingIngest,
    diagnosis: CctvDiagnosis | None,
    base: Path,
) -> InputFile:
    record = source.to_json()
    working = ingest.to_json()
    return InputFile(
        name=source.original_name,
        size_bytes=source.size_bytes,
        mtime=source.modified_at,
        sha256=source.sha256,
        received_at=Timestamp(**record["receivedAt"]),
        verified_copy_path=relative_to_base(verified.path, base),
        verified_copy_sha256=verified.sha256,
        container=record["container"],
        working_copy=working["workingCopy"],
        ffprobe=ingest.source_probe,
        frame_index=working["frameIndex"],
        video=working["video"],
        audio=working["audio"],
        decode=working["decode"],
        lite=working["lite"],
        diagnosis=None if diagnosis is None else diagnosis.model_dump(mode="json", by_alias=True),
        warnings=working["warnings"],
    )


def trim_info(chain: Sequence[ResolvedStep], frames: Sequence[FrameEntry]) -> TrimInfo | None:
    trim = trim_step(chain)
    if trim is None:
        return None
    start, end = int(trim.params["start_frame"]), int(trim.params["end_frame"])
    return TrimInfo(
        start_frame=start,
        end_frame=end,
        audio_start_seconds=frame_time(frames, start),
        audio_end_seconds=frame_time(frames, end + 1),
    )


def catalog_filter(step: ResolvedStep) -> FilterSpec:
    spec = step_spec(step.id)
    match = next((candidate for candidate in spec.filters if candidate.name == step.filter), None)
    if match is None:
        raise CctvChainError(UNKNOWN_FILTER, f"Step {step.id!r} has no filter {step.filter!r}.")
    return match


def ffmpeg_engine(caps: FfmpegCapabilities) -> EngineInfo:
    return EngineInfo(name=FFMPEG_ENGINE, version=caps.version)


def run_fields(run: ProcessRun, local_tz: tzinfo | None) -> dict[str, Any]:
    return {
        "argv": list(run.argv),
        "started_at": timestamp(run.started_at, local_tz),
        "ended_at": timestamp(run.ended_at, local_tz),
        "exit_code": run.exit_code,
        "frames_in": run.frames_in,
        "frames_out": run.frames_out,
    }


def report_step(
    index: int, step: ResolvedStep, run: ProcessRun, engine: EngineInfo, local_tz: tzinfo | None
) -> ReportStep:
    spec, filter_spec = step_spec(step.id), catalog_filter(step)
    return ReportStep(
        index=index,
        id=step.id,
        category=spec.category,
        label=spec.label,
        filter=step.filter,
        description=filter_spec.description,
        doc_url=filter_spec.doc_url,
        parameters=dict(step.params),
        engine=engine,
        **run_fields(run, local_tz),
    )


def chain_steps(
    chain: Sequence[ResolvedStep], run: ProcessRun, engine: EngineInfo, local_tz: tzinfo | None
) -> list[ReportStep]:
    # En el carril clasico toda la cadena corre en un solo ffmpeg: cada paso lleva ese argv.
    return [report_step(index, step, run, engine, local_tz) for index, step in enumerate(chain)]


def process_info(run: ProcessRun, local_tz: tzinfo | None) -> ProcessInfo:
    return ProcessInfo(label=run.label, **run_fields(run, local_tz))


def summarize_stream(stream: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("index", "codec_type", "codec_name", "width", "height", "pix_fmt", "sample_aspect_ratio",
            "avg_frame_rate", "nb_frames", "sample_rate", "channels")
    return {key: stream[key] for key in keys if stream.get(key) is not None}


def summarize_probe(probe: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if probe is None:
        return None
    fmt = probe.get("format", {})
    summary = {key: fmt[key] for key in ("format_name", "duration", "size", "bit_rate") if fmt.get(key) is not None}
    return {**summary, "streams": [summarize_stream(stream) for stream in probe.get("streams", [])]}


def output_file(artifact: OutputArtifact, base: Path, hash_file: Callable[[Path], str]) -> OutputFile:
    return OutputFile(
        path=relative_to_base(artifact.path, base),
        role=artifact.role,
        sha256=hash_file(artifact.path),
        ffprobe=summarize_probe(artifact.probe),
        ai_applied=artifact.ai_applied,
        visible_label=artifact.visible_label,
    )


def still_pair_info(pair: StillPair, base: Path) -> StillPairInfo:
    return StillPairInfo.model_validate(pair.to_json(base))


def clipping_info(clipping: ClippingReport | None) -> ClippingInfo | None:
    return None if clipping is None else ClippingInfo.model_validate(clipping.to_json())


# --- Limitaciones y avisos ---


def _unique_by_key(limitations: Iterable[Limitation]) -> tuple[Limitation, ...]:
    unique: dict[str, Limitation] = {}
    for limitation in limitations:
        unique.setdefault(limitation.key, limitation)
    return tuple(unique.values())


def source_limitations(source: SourceRecord) -> tuple[Limitation, ...]:
    return (REMUXED_PROPRIETARY,) if source.container.kind in PROPRIETARY_CONTAINERS else ()


def lane_limitations(mode: ReportMode) -> tuple[Limitation, ...]:
    return (AI_DETAIL,) if mode == "ai-visual" else (SAME_BUILD,)


def output_limitations(outputs: Sequence[OutputArtifact]) -> tuple[Limitation, ...]:
    return (LOSSY_COPIES,) if any(output.role in LOSSY_ROLES for output in outputs) else ()


def acquisition_limitations(acquisition: Acquisition) -> tuple[Limitation, ...]:
    return (CLOCK_OFFSET,) if acquisition.clock_offset_seconds is not None else ()


def clipping_limitations(clipping: ClippingReport | None) -> tuple[Limitation, ...]:
    limitation = None if clipping is None else clipping_limitation(clipping)
    return () if limitation is None else (limitation,)


def automatic_limitations(parts: ReportParts) -> tuple[Limitation, ...]:
    return _unique_by_key(
        (
            *source_limitations(parts.source),
            *limitations_for(parts.chain),
            *clipping_limitations(parts.clipping),
            *output_limitations(parts.outputs),
            *lane_limitations(parts.mode),
            HASH_SCOPE,
            *acquisition_limitations(parts.acquisition),
            *parts.extra_limitations,
        )
    )


def clipping_warnings(clipping: ClippingReport | None) -> tuple[str, ...]:
    return (CLIPPING_INCREASED_KEY,) if clipping is not None and clipping.increased else ()


def report_warnings(parts: ReportParts) -> list[str]:
    diagnosis = () if parts.diagnosis is None else tuple(parts.diagnosis.warnings)
    collected = (
        *parts.ingest.warnings,
        *diagnosis,
        *parts.osd.get("warnings", ()),
        *clipping_warnings(parts.clipping),
        *parts.warnings,
    )
    return list(dict.fromkeys(collected))


# --- Informe completo ---


def build_report(parts: ReportParts, tools: ReportTools = ReportTools()) -> CctvReportV1:
    tz, base = tools.local_tz, parts.base_dir
    frames = () if parts.ingest.index is None else parts.ingest.index.frames
    return CctvReportV1(
        generated_at=timestamp(tools.now(), tz),
        case=parts.case,
        upflow=UpflowInfo(version=parts.app_version, commit=parts.commit),
        mode=parts.mode,
        ai_used=parts.mode == "ai-visual",
        environment=environment_info(parts.caps, parts.host),
        inputs=[input_file(parts.source, parts.verified_copy, parts.ingest, parts.diagnosis, base)],
        acquisition=parts.acquisition,
        osd=OsdInfo.model_validate(parts.osd),
        trim=trim_info(parts.chain, frames),
        steps=chain_steps(parts.chain, parts.chain_run, ffmpeg_engine(parts.caps), tz),
        processes=[process_info(run, tz) for run in parts.processes],
        outputs=[output_file(output, base, tools.hash_file) for output in parts.outputs],
        stills=[still_pair_info(pair, base) for pair in parts.stills],
        clipping=clipping_info(parts.clipping),
        frame_hashes=parts.frame_hashes,
        limitations=[LimitationInfo(key=item.key, text=item.text) for item in automatic_limitations(parts)],
        warnings=report_warnings(parts),
        guidelines=GUIDELINES_TEXT,
        disclaimer=DISCLAIMER_TEXT,
    )


def report_json_text(report: CctvReportV1) -> str:
    payload = report.model_dump(mode="json", by_alias=True)
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def load_report(text: str) -> CctvReportV1:
    return CctvReportV1.model_validate_json(text)


def write_report(report: CctvReportV1, out_dir: Path, render_html: Callable[[CctvReportV1], str]) -> tuple[Path, Path]:
    json_path, html_path = out_dir / REPORT_JSON_NAME, out_dir / REPORT_HTML_NAME
    json_path.write_text(report_json_text(report), encoding="utf-8", newline="\n")
    html_path.write_text(render_html(report), encoding="utf-8", newline="\n")
    return json_path, html_path


# --- SHA256SUMS.txt ---


@dataclass(frozen=True, slots=True)
class ChecksumEntry:
    sha256: str
    path: str

    def line(self) -> str:
        return f"{self.sha256} *{self.path}"


def checksum_entries(
    base: Path, relative_paths: Sequence[str], hash_file: Callable[[Path], str] = sha256_file
) -> tuple[ChecksumEntry, ...]:
    checked = [check_relative_path(path) for path in relative_paths]
    return tuple(ChecksumEntry(hash_file(base / path), path) for path in dict.fromkeys(checked))


def sha256sums_text(entries: Sequence[ChecksumEntry]) -> str:
    return "".join(f"{entry.line()}\n" for entry in entries)


def write_sha256sums(
    base: Path, relative_paths: Sequence[str], hash_file: Callable[[Path], str] = sha256_file
) -> Path:
    path = base / SHA256SUMS_NAME
    path.write_text(sha256sums_text(checksum_entries(base, relative_paths, hash_file)), encoding="utf-8", newline="\n")
    return path


def parse_checksum_line(line: str) -> ChecksumEntry:
    digest, separator, path = line.partition(" *")
    if not separator:
        raise ValueError(f"Not a sha256sum line: {line[:200]!r}")
    return ChecksumEntry(check_sha256(digest.lower()), check_relative_path(path))


def parse_sha256sums(text: str) -> tuple[ChecksumEntry, ...]:
    return tuple(parse_checksum_line(line) for line in text.splitlines() if line.strip())
