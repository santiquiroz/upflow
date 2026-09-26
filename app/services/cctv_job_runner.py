"""Job "Clarify video" del carril clasico (spec §4.1, §4.6 a §4.13).

Arma, desde la copia verificada de `outputs/{id}.cctv/01_original`, la copia de
trabajo y su indice, el diagnostico, las copias procesadas, los cuadros
exportados, el comparativo, el informe y el paquete de entrega. Todo lo que
sobrevive al job queda en `outputs/{id}.cctv/`; los intermedios van al
`video-work/{id}` que el llamador borra al terminar.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings
from app.core.version import get_app_version
from app.models import CctvOptions, VideoUpscaleJob, utc_now
from app.services.cctv_chain import ResolvedStep
from app.services.cctv_clarify_runner import (
    ClarifyPlan,
    ClarifyResult,
    ClarifyThreads,
    ClarifyTools,
    PassTiming,
    StageProgress,
    clarify_threads,
    run_clarify,
    trim_step,
)
from app.services.cctv_frame_index import FrameEntry
from app.services.cctv_ingest import (
    FRAME_INDEX_NAME,
    UNDECODABLE_WARNING,
    MediaTools,
    SourceRecord,
    VerifiedCopy,
    VerifiedCopyMismatch,
    WorkingIngest,
    first_stream,
    ingest_working_copy,
    probe_media,
    read_source_record,
    sha256_file,
    source_facts,
    verified_copy_path,
)
from app.services.cctv_job_validation import osd_selection, parse_case_details, resolve_steps
from app.services.cctv_report import (
    HostFacts,
    OutputArtifact,
    ProcessRun,
    ReportParts,
    build_report,
    host_facts,
    write_report,
)
from app.services.cctv_report_html import render_report_html
from app.services.cctv_session import cctv_job_dir
from app.services.ffmpeg_capabilities import FfmpegCapabilities, cached_capabilities
from app.services.ffmpeg_filters import Box, FrameGeometry
from app.services.frame_export import StillPair, StillRequest, StillSource, export_still_pairs
from app.services.handover_package import (
    COMPARISONS_DIRNAME,
    STILLS_DIRNAME,
    PackageOutcome,
    ProcessedFiles,
    ReproducedOutput,
    ReproduceSources,
    build_handover_package,
    finalize_package_files,
    free_disk_bytes,
    package_reproduce_script,
    package_root_name,
    place_processed,
    readme_facts,
)
from app.services.osd_check import OsdBoxCheck, osd_report, run_osd_check
from app.services.side_by_side import ComparisonPlan, ComparisonResult, run_side_by_side
from app.services.video_analysis import CctvDiagnosis, analyze_video

STAGE_INGESTING = "ingesting"
STAGE_ANALYZING = "analyzing_video"
STAGE_EXPORTING = "exporting_frames"
STAGE_COMPARISON = "building_comparison"
STAGE_REPORTING = "reporting"
STAGE_PACKAGING = "packaging"

CLASSIC_LANE = "classic"
CLASSIC_MODE = "classic"
CLARIFY_DIRNAME = "clarify"
DEFAULT_PIX_FMT = "yuv420p"
ANALYSIS_LABEL = "analysis copy (FFV1)"
VIEWING_LABEL = "viewing copy (H.264)"
COMPARISON_LABEL = "side-by-side comparison (H.264)"
UNDECODABLE_MESSAGE = (
    "The video could not be decoded. The file may be encrypted, damaged or truncated; "
    "keep the original and do not convert it with other tools before handing it over."
)


class CctvTaskRunner(Protocol):
    async def run(self, job: VideoUpscaleJob, work_dir: Path, on_stage: StageProgress) -> Path: ...


class UndecodableVideo(RuntimeError):
    def __init__(self) -> None:
        super().__init__(UNDECODABLE_MESSAGE)
        self.key = UNDECODABLE_WARNING


@dataclass(frozen=True, slots=True)
class CctvRunnerConfig:
    outputs_root: Path
    media: MediaTools
    clarify: ClarifyTools
    threads: ClarifyThreads
    app_version: str
    capabilities: Callable[[], FfmpegCapabilities]
    host: Callable[[], HostFacts] = host_facts
    free_bytes: Callable[[Path], int] = free_disk_bytes
    today: Callable[[], date] = date.today
    now: Callable[[], datetime] = utc_now


@dataclass(frozen=True, slots=True)
class ClarifyJob:
    job: VideoUpscaleJob
    steps: tuple[ResolvedStep, ...]
    job_dir: Path
    work_dir: Path
    on_stage: StageProgress

    @property
    def options(self) -> CctvOptions:
        return self.job.cctv

    @property
    def osd_boxes(self) -> tuple[Box, ...]:
        return tuple(tuple(box) for box in self.options.osd_boxes)


@dataclass(frozen=True, slots=True)
class IngestedSource:
    record: SourceRecord
    verified: VerifiedCopy
    ingest: WorkingIngest
    geometry: FrameGeometry
    pix_fmt: str

    @property
    def work(self) -> Path:
        return self.ingest.working_copy.path

    @property
    def frames(self) -> tuple[FrameEntry, ...]:
        return () if self.ingest.index is None else self.ingest.index.frames

    @property
    def frame_times(self) -> tuple[float, ...]:
        return tuple(frame.pts_time or 0.0 for frame in self.frames)


@dataclass(frozen=True, slots=True)
class ClarifyOutputs:
    result: ClarifyResult
    placed: ProcessedFiles
    stills: tuple[StillPair, ...]
    comparison: ComparisonResult
    comparison_timing: PassTiming


@dataclass(frozen=True, slots=True)
class ReportInputs:
    caps: FfmpegCapabilities
    diagnosis: CctvDiagnosis
    osd: dict[str, Any]
    outputs: ClarifyOutputs


# --- Ingesta dentro del job ---


def parse_sar(raw: Any) -> Fraction:
    try:
        num, den = (int(part) for part in str(raw).split(":"))
    except ValueError:
        return Fraction(1)
    return Fraction(num, den) if num > 0 and den > 0 else Fraction(1)


def video_fields(probe: dict[str, Any]) -> dict[str, Any]:
    return first_stream(probe, "video") or {}


def frame_geometry(probe: dict[str, Any]) -> FrameGeometry:
    stream = video_fields(probe)
    return FrameGeometry(
        int(stream.get("width") or 0), int(stream.get("height") or 0), parse_sar(stream.get("sample_aspect_ratio"))
    )


def pix_fmt_of(probe: dict[str, Any]) -> str:
    return str(video_fields(probe).get("pix_fmt") or DEFAULT_PIX_FMT)


def job_verified_copy(job_dir: Path) -> tuple[SourceRecord, VerifiedCopy]:
    # Se re-hashea al empezar: el informe declara ese hash y tiene que ser el del archivo que se proceso.
    record = read_source_record(job_dir)
    path = verified_copy_path(job_dir, record.original_name)
    digest = sha256_file(path)
    if digest != record.sha256:
        raise VerifiedCopyMismatch(record.sha256, digest)
    return record, VerifiedCopy(path, digest)


def require_decodable(ingest: WorkingIngest) -> None:
    if ingest.decode.decode_failed or ingest.index is None:
        raise UndecodableVideo()


def keep_frame_index(ingest: WorkingIngest, job_dir: Path) -> None:
    shutil.copyfile(ingest.index.csv_path, job_dir / FRAME_INDEX_NAME)


async def ingest_job_source(media: MediaTools, job_dir: Path, work_dir: Path, on_stage: StageProgress) -> IngestedSource:
    on_stage(STAGE_INGESTING, 0.0)
    record, verified = await asyncio.to_thread(job_verified_copy, job_dir)
    ingest = await ingest_working_copy(media, verified.path, work_dir, record.container)
    require_decodable(ingest)
    keep_frame_index(ingest, job_dir)
    probe = await probe_media(media, ingest.working_copy.path)
    on_stage(STAGE_INGESTING, 1.0)
    return IngestedSource(record, verified, ingest, frame_geometry(probe), pix_fmt_of(probe))


# --- Planes de cada etapa ---


def trim_start_of(steps: Sequence[ResolvedStep]) -> int:
    trim = trim_step(steps)
    return 0 if trim is None else int(trim.params["start_frame"])


def clarify_plan(clarify: ClarifyJob, source: IngestedSource, app_version: str) -> ClarifyPlan:
    return ClarifyPlan(
        work=source.work,
        steps=clarify.steps,
        geometry=source.geometry,
        source_pix_fmt=source.pix_fmt,
        frames=source.frames,
        has_audio=bool(source.ingest.audio),
        app_version=app_version,
        osd_boxes=clarify.osd_boxes,
    )


def still_request(clarify: ClarifyJob, source: IngestedSource, analysis: Path) -> StillRequest:
    times = source.frame_times
    frames = tuple(sorted(set(clarify.options.still_frames)))
    # Carril clasico: sin re-temporizar, el cuadro procesado es el original menos el comienzo del recorte.
    return StillRequest(
        StillSource(source.work, times),
        StillSource(analysis, times),
        frames,
        clarify.job_dir / STILLS_DIRNAME,
        trim_start_of(clarify.steps),
    )


def comparison_plan(clarify: ClarifyJob, source: IngestedSource, result: ClarifyResult, analysis: Path) -> ComparisonPlan:
    return ComparisonPlan(
        source.work, analysis, clarify.steps, result.output_geometry, CLASSIC_LANE, result.output_frames
    )


# --- Informe y metadata ---


def process_run(label: str, argv: Sequence[str], timing: PassTiming, frames: tuple[int | None, int | None]) -> ProcessRun:
    return ProcessRun(label, tuple(argv), timing.started_at, timing.ended_at, 0, frames_in=frames[0], frames_out=frames[1])


def clarify_runs(outputs: ClarifyOutputs) -> tuple[ProcessRun, ProcessRun, ProcessRun]:
    result = outputs.result
    return (
        process_run(ANALYSIS_LABEL, result.analysis_command, result.analysis_timing, (result.expected_frames, result.output_frames)),
        process_run(VIEWING_LABEL, result.viewing_command, result.viewing_timing, (result.output_frames, result.output_frames)),
        process_run(COMPARISON_LABEL, outputs.comparison.command, outputs.comparison_timing, (None, result.output_frames)),
    )


def still_artifacts(stills: Sequence[StillPair]) -> tuple[OutputArtifact, ...]:
    return tuple(OutputArtifact(still.path, "still") for pair in stills for still in (pair.original, pair.processed))


def output_artifacts(outputs: ClarifyOutputs) -> tuple[OutputArtifact, ...]:
    return (
        OutputArtifact(outputs.placed.analysis, "analysis-lossless"),
        OutputArtifact(outputs.placed.viewing, "viewing-copy"),
        OutputArtifact(outputs.comparison.path, "comparison"),
        *still_artifacts(outputs.stills),
    )


def report_parts(
    clarify: ClarifyJob, source: IngestedSource, inputs: ReportInputs, app_version: str, host: HostFacts
) -> ReportParts:
    acquisition, case = parse_case_details(clarify.options)
    outputs = inputs.outputs
    runs = clarify_runs(outputs)
    return ReportParts(
        mode=CLASSIC_MODE,
        app_version=app_version,
        commit=None,
        case=case,
        acquisition=acquisition,
        caps=inputs.caps,
        host=host,
        source=source.record,
        verified_copy=source.verified,
        ingest=source.ingest,
        diagnosis=inputs.diagnosis,
        osd=inputs.osd,
        chain=clarify.steps,
        chain_run=runs[0],
        processes=runs,
        outputs=output_artifacts(outputs),
        base_dir=clarify.job_dir,
        stills=outputs.stills,
        clipping=outputs.result.clipping,
        extra_limitations=outputs.result.limitations,
        warnings=outputs.comparison.warnings,
    )


def reproduced_outputs(outputs: ClarifyOutputs, job_dir: Path) -> tuple[ReproducedOutput, ...]:
    placed, result = outputs.placed, outputs.result
    return (
        ReproducedOutput(result.analysis, placed.analysis.relative_to(job_dir).as_posix()),
        ReproducedOutput(result.viewing, placed.viewing.relative_to(job_dir).as_posix()),
    )


def relative(path: Path | None, base: Path) -> str | None:
    return None if path is None else path.relative_to(base).as_posix()


def cctv_metadata(
    clarify: ClarifyJob, source: IngestedSource, outputs: ClarifyOutputs, package: PackageOutcome
) -> dict[str, Any]:
    job_dir = clarify.job_dir
    warnings = [*source.ingest.warnings, *outputs.comparison.warnings, *package.warnings]
    return {
        "task": clarify.options.task,
        "lane": CLASSIC_LANE,
        "sourceSha256": source.record.sha256,
        "receivedAt": {"utc": source.record.received_at.utc, "local": source.record.received_at.local},
        "framesIn": outputs.result.expected_frames,
        "framesOut": outputs.result.output_frames,
        "outputs": {
            "analysis": relative(outputs.placed.analysis, job_dir),
            "viewing": relative(outputs.placed.viewing, job_dir),
            "comparison": relative(outputs.comparison.path, job_dir),
            "stills": [pair.to_json(job_dir) for pair in outputs.stills],
            "package": relative(package.path, job_dir),
        },
        "warnings": list(dict.fromkeys(warnings)),
    }


# --- Orquestacion ---


class CctvClarifyRunner:
    def __init__(self, config: CctvRunnerConfig) -> None:
        self.config = config

    async def run(self, job: VideoUpscaleJob, work_dir: Path, on_stage: StageProgress) -> Path:
        steps = resolve_steps(job.cctv, CLASSIC_LANE)
        clarify = ClarifyJob(job, steps, cctv_job_dir(self.config.outputs_root, job.id), work_dir, on_stage)
        caps = await asyncio.to_thread(self.config.capabilities)
        source = await self._ingest(clarify)
        diagnosis, osd = await self._analyze(clarify, source)
        outputs = await self._clarify(clarify, source)
        await self._report(clarify, source, ReportInputs(caps, diagnosis, osd, outputs))
        package = await self._package(clarify, source, caps, outputs)
        job.metadata["cctv"] = cctv_metadata(clarify, source, outputs, package)
        return outputs.placed.viewing

    async def _ingest(self, clarify: ClarifyJob) -> IngestedSource:
        return await ingest_job_source(self.config.media, clarify.job_dir, clarify.work_dir, clarify.on_stage)

    async def _analyze(self, clarify: ClarifyJob, source: IngestedSource) -> tuple[CctvDiagnosis, dict[str, Any]]:
        clarify.on_stage(STAGE_ANALYZING, 0.0)
        media = self.config.media
        diagnosis = await analyze_video(media.ffmpeg, source.work, source_facts(source.ingest), run=media.run)
        checks = await self._osd_checks(clarify, source)
        clarify.on_stage(STAGE_ANALYZING, 1.0)
        return diagnosis, osd_report(osd_selection(clarify.options), checks, clarify.steps)

    async def _osd_checks(self, clarify: ClarifyJob, source: IngestedSource) -> tuple[OsdBoxCheck, ...]:
        if not clarify.osd_boxes:
            return ()
        geometry, media = source.geometry, self.config.media
        return await run_osd_check(
            media.ffmpeg, source.work, geometry.width, geometry.height, clarify.osd_boxes, run=media.run
        )

    async def _clarify(self, clarify: ClarifyJob, source: IngestedSource) -> ClarifyOutputs:
        result = await self._run_clarify(clarify, source)
        placed = place_processed(clarify.job_dir, source.record.original_name, clarify.job.id, result.analysis, result.viewing)
        stills = await self._export_stills(clarify, source, placed.analysis)
        comparison, timing = await self._compare(clarify, source, result, placed.analysis)
        return ClarifyOutputs(result, placed, stills, comparison, timing)

    async def _run_clarify(self, clarify: ClarifyJob, source: IngestedSource) -> ClarifyResult:
        config = self.config
        output_dir = clarify.work_dir / CLARIFY_DIRNAME
        output_dir.mkdir(parents=True, exist_ok=True)
        plan = clarify_plan(clarify, source, config.app_version)
        return await run_clarify(config.clarify, plan, output_dir, config.threads, clarify.on_stage, config.now)

    async def _export_stills(self, clarify: ClarifyJob, source: IngestedSource, analysis: Path) -> tuple[StillPair, ...]:
        if not clarify.options.still_frames:
            return ()
        clarify.on_stage(STAGE_EXPORTING, 0.0)
        request = still_request(clarify, source, analysis)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        pairs = await export_still_pairs(self.config.clarify, request)
        clarify.on_stage(STAGE_EXPORTING, 1.0)
        return pairs

    async def _compare(
        self, clarify: ClarifyJob, source: IngestedSource, result: ClarifyResult, analysis: Path
    ) -> tuple[ComparisonResult, PassTiming]:
        output_dir = clarify.job_dir / COMPARISONS_DIRNAME
        output_dir.mkdir(parents=True, exist_ok=True)
        config = self.config
        started = config.now()
        comparison = await run_side_by_side(
            config.clarify,
            comparison_plan(clarify, source, result, analysis),
            output_dir,
            clarify.work_dir,
            config.threads,
            lambda fraction: clarify.on_stage(STAGE_COMPARISON, fraction),
        )
        return comparison, PassTiming(started, config.now())

    async def _report(self, clarify: ClarifyJob, source: IngestedSource, inputs: ReportInputs) -> None:
        clarify.on_stage(STAGE_REPORTING, 0.0)
        parts = report_parts(clarify, source, inputs, self.config.app_version, self.config.host())
        report = await asyncio.to_thread(build_report, parts)
        await asyncio.to_thread(write_report, report, clarify.job_dir, render_report_html)
        clarify.on_stage(STAGE_REPORTING, 1.0)

    async def _package(
        self, clarify: ClarifyJob, source: IngestedSource, caps: FfmpegCapabilities, outputs: ClarifyOutputs
    ) -> PackageOutcome:
        clarify.on_stage(STAGE_PACKAGING, 0.0)
        job_dir = clarify.job_dir
        sources = ReproduceSources(source.verified.path, source.ingest.working_copy)
        script = package_reproduce_script(sources, outputs.result.reproduce_steps(), reproduced_outputs(outputs, job_dir), caps)
        root = package_root_name(clarify.options.case_label, source.record.original_name, self.config.today())
        facts = readme_facts(source.record, CLASSIC_MODE, caps)
        await asyncio.to_thread(finalize_package_files, job_dir, root, facts, script)
        outcome = await asyncio.to_thread(
            build_handover_package, job_dir, job_dir / f"{root}.zip", root, self.config.free_bytes
        )
        clarify.on_stage(STAGE_PACKAGING, 1.0)
        return outcome


def clarify_runner_config(settings: Settings) -> CctvRunnerConfig:
    ffmpeg, ffprobe = settings.ffmpeg_binary_path, settings.ffprobe_binary_path
    return CctvRunnerConfig(
        outputs_root=settings.outputs_path,
        media=MediaTools(ffmpeg, ffprobe),
        clarify=ClarifyTools(ffmpeg, ffprobe),
        threads=clarify_threads(settings.cctv_x264_threads, settings.cctv_ffv1_slices),
        app_version=get_app_version(settings.update_package_name),
        capabilities=lambda: cached_capabilities(ffmpeg),
    )


def build_cctv_runners(settings: Settings) -> dict[str, CctvTaskRunner]:
    # Import diferido: roi_fusion_runner reusa la ingesta de este modulo. "enhance" lo registra VideoUpscaler
    # (CctvEnhanceRunner) cuando tiene la etapa compuesta: necesita su encoder y su raw-pipe.
    from app.services.roi_fusion_runner import RoiFusionRunner

    config = clarify_runner_config(settings)
    return {"clarify": CctvClarifyRunner(config), "roi_fusion": RoiFusionRunner(config, settings.cctv_roi_ecc_min)}
