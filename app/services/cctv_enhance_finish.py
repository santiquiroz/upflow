"""Cierre de un job del carril IA de CCTV: cuadros exportados, informe y `SHA256SUMS.txt`.

Corre despues del stream, con el video rotulado ya en `02_processed/`. Todas
las dependencias (ffmpeg, capacidades de la build, hechos del host, hash) entran
por `EnhanceFinishTools`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models import CctvOptions
from app.services.cctv_chain import AiLanePlan, ResolvedStep
from app.services.cctv_clarify_runner import ClarifyTools
from app.services.cctv_enhance_outputs import (
    EnhanceLabel,
    EnhanceReportFacts,
    EnhanceRuns,
    ModelFile,
    enhance_report_parts,
    enhance_still_request,
    enhanced_artifact,
    output_times,
    still_artifacts,
)
from app.services.cctv_enhance_plan import EnhancePlan
from app.services.cctv_ingest import MediaTools, probe_media, sha256_file
from app.services.cctv_job_runner import STAGE_EXPORTING, STAGE_REPORTING, IngestedSource, StageProgress
from app.services.cctv_job_validation import osd_selection
from app.services.cctv_report import HostFacts, OutputArtifact, ReportTools, build_report, write_report, write_sha256sums
from app.services.cctv_report_html import render_report_html
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.frame_export import StillPair, StillSource, export_still_pairs
from app.services.handover_package import checksum_paths
from app.services.label_band import still_xmp_packet
from app.services.osd_check import OsdBoxCheck, osd_report, run_osd_check


@dataclass(frozen=True, slots=True)
class EnhanceFinishTools:
    media: MediaTools
    stills: ClarifyTools
    capabilities: Callable[[], FfmpegCapabilities]
    host: Callable[[], HostFacts]
    hash_file: Callable[[Path], str] = sha256_file


@dataclass(frozen=True, slots=True)
class EnhanceContext:
    job_id: str
    options: CctvOptions
    lane: AiLanePlan
    plan: EnhancePlan
    source: IngestedSource
    job_dir: Path
    enhanced: Path
    frames_out: int
    label: EnhanceLabel
    runs: EnhanceRuns
    models: Mapping[str, ModelFile]
    generative: bool
    app_version: str


@dataclass(frozen=True, slots=True)
class EnhanceOutputs:
    stills: tuple[StillPair, ...]
    report_json: Path
    report_html: Path
    checksums: Path


# --- Cuadros exportados ---


def enhanced_still_source(context: EnhanceContext) -> StillSource:
    return StillSource(context.enhanced, output_times(context.plan.rate, context.frames_out))


async def export_enhance_stills(
    tools: EnhanceFinishTools, context: EnhanceContext, on_stage: StageProgress
) -> tuple[StillPair, ...]:
    if not context.options.still_frames:
        return ()
    on_stage(STAGE_EXPORTING, 0.0)
    packet = still_xmp_packet(context.app_version, context.job_id, context.generative)
    source = enhanced_still_source(context)
    request = enhance_still_request(context.options, context.lane, context.source, source, context.job_dir, packet)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = await export_still_pairs(tools.stills, request)
    on_stage(STAGE_EXPORTING, 1.0)
    return pairs


# --- Informe ---


async def osd_checks(tools: EnhanceFinishTools, context: EnhanceContext) -> tuple[OsdBoxCheck, ...]:
    boxes = tuple(tuple(box) for box in context.options.osd_boxes)
    if not boxes:
        return ()
    geometry, media = context.source.geometry, tools.media
    return await run_osd_check(media.ffmpeg, context.source.work, geometry.width, geometry.height, boxes, run=media.run)


def lane_steps(lane: AiLanePlan) -> tuple[ResolvedStep, ...]:
    return (*lane.decode, *lane.composite, *lane.encode)


def enhance_outputs(
    context: EnhanceContext, probe: Mapping[str, Any], stills: Sequence[StillPair]
) -> tuple[OutputArtifact, ...]:
    return (enhanced_artifact(context.enhanced, probe, context.label.text), *still_artifacts(stills, context.label.text))


async def report_facts(
    tools: EnhanceFinishTools, context: EnhanceContext, stills: tuple[StillPair, ...]
) -> EnhanceReportFacts:
    caps = await asyncio.to_thread(tools.capabilities)
    host = await asyncio.to_thread(tools.host)
    probe = await probe_media(tools.media, context.enhanced)
    checks = await osd_checks(tools, context)
    return EnhanceReportFacts(
        options=context.options,
        lane=context.lane,
        source=context.source,
        caps=caps,
        host=host,
        osd=osd_report(osd_selection(context.options), checks, lane_steps(context.lane)),
        runs=context.runs,
        models=context.models,
        outputs=enhance_outputs(context, probe, stills),
        stills=stills,
        generative=context.generative,
        app_version=context.app_version,
        base_dir=context.job_dir,
        warnings=tuple(context.source.ingest.warnings),
    )


def write_enhance_report(facts: EnhanceReportFacts, hash_file: Callable[[Path], str]) -> tuple[Path, Path, Path]:
    report = build_report(enhance_report_parts(facts, hash_file), ReportTools(hash_file=hash_file))
    json_path, html_path = write_report(report, facts.base_dir, render_report_html)
    checksums = write_sha256sums(facts.base_dir, checksum_paths(facts.base_dir), hash_file)
    return json_path, html_path, checksums


async def finish_enhance(
    tools: EnhanceFinishTools, context: EnhanceContext, on_stage: StageProgress
) -> EnhanceOutputs:
    stills = await export_enhance_stills(tools, context, on_stage)
    on_stage(STAGE_REPORTING, 0.0)
    facts = await report_facts(tools, context, stills)
    json_path, html_path, checksums = await asyncio.to_thread(write_enhance_report, facts, tools.hash_file)
    on_stage(STAGE_REPORTING, 1.0)
    return EnhanceOutputs(stills, json_path, html_path, checksums)


# --- metadata.cctv ---


def relative_to(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def finished_json(context: EnhanceContext, outputs: EnhanceOutputs) -> dict[str, Any]:
    base = context.job_dir
    return {
        "generative": context.generative,
        "label": {"text": context.label.text, "bandHeight": context.label.assets.band_height},
        "outputs": {
            "stills": [pair.to_json(base) for pair in outputs.stills],
            "report": relative_to(outputs.report_json, base),
            "reportHtml": relative_to(outputs.report_html, base),
            "checksums": relative_to(outputs.checksums, base),
        },
    }


def with_finished(
    metadata: Mapping[str, Any], finished: Mapping[str, Any], upscale: Mapping[str, Any] | None
) -> dict[str, Any]:
    outputs = {**metadata["outputs"], **finished["outputs"]}
    return {**metadata, **finished, "aiUpscale": upscale, "outputs": outputs}
