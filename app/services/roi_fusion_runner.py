"""Job "Plate or face still (multi-frame)" (spec §4.9).

Reusa la ingesta del job CCTV (copia verificada re-hasheada, `work.mkv` e
indice), decodifica `[a, b]` con solo desentrelazado y desbloqueo clasicos,
registra la ROI, la fusiona y deja en `04_stills/` la imagen fusionada de 16
bits, la referencia ampliada con *nearest neighbor*, el mapa de acuerdo, la
pila alineada en FFV1 y `roi_samples.csv`. El informe lleva la matriz y la
correlacion ECC de cada cuadro, y `SHA256SUMS.txt` cubre todo lo que quedo.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.models import CctvOptions, RoiFusionRequest, VideoUpscaleJob
from app.services.cctv_chain import Limitation, ResolvedStep
from app.services.cctv_clarify_runner import FFV1_ARGS, OUTPUT_BITEXACT, StageProgress
from app.services.cctv_job_runner import (
    CLASSIC_LANE,
    CLASSIC_MODE,
    CctvRunnerConfig,
    IngestedSource,
    ingest_job_source,
    relative,
)
from app.services.cctv_job_validation import parse_case_details, resolve_steps
from app.services.cctv_report import (
    HostFacts,
    OutputArtifact,
    ProcessRun,
    ReportParts,
    build_report,
    write_report,
    write_sha256sums,
)
from app.services.cctv_report_html import render_report_html
from app.services.cctv_session import cctv_job_dir
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.ffmpeg_filters import build_filter
from app.services.handover_package import STILLS_DIRNAME, checksum_paths
from app.services.osd_check import OsdSelection, osd_report
from app.services.roi_frames import build_roi_decode_command, decode_roi_frames, roi_frames_from
from app.services.roi_fusion import (
    FRAMES_USED,
    FusionSettings,
    RoiDensity,
    RoiFusion,
    RoiNotice,
    clipped_frames_pct,
    frames_used,
    fuse_roi,
    fusion_notices,
    output_names,
    png_bytes,
    roi_density,
    samples_csv_text,
    stack_gray16le,
    to_uint16,
)
from app.services.roi_registration import (
    RegistrationResult,
    RegistrationSettings,
    RoiBox,
    RoiFrame,
    motion_for,
    register_roi_frames,
)
from app.services.video_analysis import run_pass

STAGE_REGISTERING = "roi_registering"
STAGE_FUSING = "roi_fusing"
STAGE_REPORTING = "reporting"

DECODE_LABEL = "ROI frames decoded to gray (rawvideo)"
STACK_LABEL = "aligned ROI stack (FFV1)"
STACK_PASS = "roi-stack"
STACK_RAW_NAME = "roi_stack.gray16le"
STACK_PIX_FMT = "gray16le"
STACK_TIMEOUT_SECONDS = 600.0

ROI_NO_NEW_DETAIL = Limitation(
    "cctv.limitation.roiNoNewDetail",
    "Combining frames can reduce noise, but it can't create detail finer than what was recorded.",
)
NO_OSD_DECISION = OsdSelection((), False, False)
OUTPUT_ROLES = {"fused": "roi-fused", "reference": "roi-reference", "agreement": "roi-agreement", "stack": "roi-stack"}


@dataclass(frozen=True, slots=True)
class RoiJob:
    job: VideoUpscaleJob
    steps: tuple[ResolvedStep, ...]
    job_dir: Path
    work_dir: Path
    on_stage: StageProgress

    @property
    def options(self) -> CctvOptions:
        return self.job.cctv

    @property
    def request(self) -> RoiFusionRequest:
        return self.options.roi

    @property
    def box(self) -> RoiBox:
        return RoiBox(*self.request.box)

    @property
    def stills_dir(self) -> Path:
        return self.job_dir / STILLS_DIRNAME


@dataclass(frozen=True, slots=True, eq=False)
class DecodedRange:
    frames: tuple[RoiFrame, ...]
    run: ProcessRun


@dataclass(frozen=True, slots=True, eq=False)
class FusionOutcome:
    registration: RegistrationResult
    fusion: RoiFusion
    density: RoiDensity
    clipped_pct: float
    notices: tuple[RoiNotice, ...]
    files: dict[str, Path]
    stack_run: ProcessRun


@dataclass(frozen=True, slots=True)
class RoiReportFacts:
    caps: FfmpegCapabilities
    host: HostFacts
    app_version: str
    roi: dict[str, Any]


# --- Comandos ---


def roi_prefilters(steps: tuple[ResolvedStep, ...]) -> tuple[str, ...]:
    return tuple(build_filter(step) for step in steps)


def build_stack_command(ffmpeg: Path, raw: Path, width: int, height: int, output: Path) -> list[str]:
    return [
        str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", STACK_PIX_FMT, "-video_size", f"{width}x{height}", "-framerate", "1",
        "-i", str(raw), "-map", "0:v:0", "-threads", "1", *FFV1_ARGS,
        *OUTPUT_BITEXACT, str(output),
    ]  # fmt: skip


# --- Informe y metadata ---


def library_versions() -> dict[str, str]:
    return {"opencv": cv2.__version__, "numpy": np.__version__}


def roi_report_json(
    request: RoiFusionRequest, outcome: FusionOutcome, settings: FusionSettings, ecc_min: float
) -> dict[str, Any]:
    registration = outcome.registration
    return {
        "kind": request.kind,
        "box": list(request.box),
        "firstFrame": request.first_frame,
        "lastFrame": request.last_frame,
        "referenceFrame": request.reference_frame,
        "scale": settings.scale,
        "method": settings.method,
        "motion": registration.motion,
        "eccMin": ecc_min,
        "noiseFloor": registration.noise_floor,
        "copyThreshold": registration.copy_threshold,
        "framesTotal": len(registration.samples),
        "framesUsed": frames_used(registration.samples),
        "effectiveSamples": registration.effective_samples,
        "rejectedFrames": list(registration.rejected_frames),
        "nearCopies": registration.near_copies,
        "spread": registration.spread.to_json(),
        "density": outcome.density.to_json(),
        "clippedFramesPct": outcome.clipped_pct,
        "trimmedShare": settings.trimmed_share,
        "agreementFullScale": settings.agreement_full_scale,
        "libraries": library_versions(),
        "samples": [sample.to_json() for sample in registration.samples],
        "notices": [notice.to_json() for notice in outcome.notices],
    }


def report_warnings(notices: tuple[RoiNotice, ...]) -> tuple[str, ...]:
    # "Frames used" es un dato del resultado, no un aviso.
    return tuple(notice.key for notice in notices if notice.key != FRAMES_USED)


def output_artifacts(files: dict[str, Path]) -> tuple[OutputArtifact, ...]:
    return tuple(OutputArtifact(files[name], role) for name, role in OUTPUT_ROLES.items())


def report_parts(
    roi: RoiJob, source: IngestedSource, outcome: FusionOutcome, decoded: DecodedRange, facts: RoiReportFacts
) -> ReportParts:
    acquisition, case = parse_case_details(roi.options)
    return ReportParts(
        mode=CLASSIC_MODE,
        app_version=facts.app_version,
        commit=None,
        case=case,
        acquisition=acquisition,
        caps=facts.caps,
        host=facts.host,
        source=source.record,
        verified_copy=source.verified,
        ingest=source.ingest,
        diagnosis=None,
        osd=osd_report(NO_OSD_DECISION, (), roi.steps),
        chain=roi.steps,
        chain_run=decoded.run,
        processes=(decoded.run, outcome.stack_run),
        outputs=output_artifacts(outcome.files),
        base_dir=roi.job_dir,
        extra_limitations=(ROI_NO_NEW_DETAIL,),
        warnings=report_warnings(outcome.notices),
        roi=facts.roi,
    )


def roi_summary(roi_json: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "kind", "scale", "method", "motion", "referenceFrame", "framesTotal", "framesUsed",
        "effectiveSamples", "rejectedFrames", "nearCopies", "density", "clippedFramesPct", "notices",
    )  # fmt: skip
    return {key: roi_json[key] for key in keys}


def roi_metadata(
    roi: RoiJob, source: IngestedSource, outcome: FusionOutcome, roi_json: dict[str, Any]
) -> dict[str, Any]:
    return {
        "task": roi.options.task,
        "lane": CLASSIC_LANE,
        "sourceSha256": source.record.sha256,
        "receivedAt": {"utc": source.record.received_at.utc, "local": source.record.received_at.local},
        "framesIn": roi_json["framesTotal"],
        "framesOut": 1,
        "outputs": {
            "roi": {name: relative(path, roi.job_dir) for name, path in outcome.files.items()},
            "package": None,
        },
        "roi": roi_summary(roi_json),
        "warnings": list(dict.fromkeys(source.ingest.warnings)),
    }


# --- Archivos ---


def write_fusion_files(roi: RoiJob, registration: RegistrationResult, fusion: RoiFusion, scale: int) -> dict[str, Path]:
    names = output_names(scale)
    roi.stills_dir.mkdir(parents=True, exist_ok=True)
    files = {name: roi.stills_dir / file_name for name, file_name in names.items()}
    files["fused"].write_bytes(png_bytes(to_uint16(fusion.fused)))
    files["reference"].write_bytes(png_bytes(fusion.reference_nearest))
    files["agreement"].write_bytes(png_bytes(fusion.agreement))
    files["samples"].write_bytes(samples_csv_text(registration.samples).encode("utf-8"))
    return files


def finish_files(job_dir: Path) -> Path:
    return write_sha256sums(job_dir, checksum_paths(job_dir))


# --- Orquestacion ---


class RoiFusionRunner:
    def __init__(self, config: CctvRunnerConfig, ecc_min: float = RegistrationSettings().ecc_min) -> None:
        self.config = config
        self.ecc_min = ecc_min

    async def run(self, job: VideoUpscaleJob, work_dir: Path, on_stage: StageProgress) -> Path:
        steps = resolve_steps(job.cctv, CLASSIC_LANE)
        roi = RoiJob(job, steps, cctv_job_dir(self.config.outputs_root, job.id), work_dir, on_stage)
        caps = await asyncio.to_thread(self.config.capabilities)
        source = await ingest_job_source(self.config.media, roi.job_dir, work_dir, on_stage)
        decoded = await self._decode(roi, source)
        registration = await self._register(roi, decoded)
        settings = FusionSettings(scale=roi.request.scale, method=roi.request.method)
        outcome = await self._fuse(roi, source, decoded, registration, settings)
        roi_json = roi_report_json(roi.request, outcome, settings, self.ecc_min)
        facts = RoiReportFacts(caps, self.config.host(), self.config.app_version, roi_json)
        await self._report(roi, source, outcome, decoded, facts)
        job.metadata["cctv"] = roi_metadata(roi, source, outcome, roi_json)
        return outcome.files["fused"]

    async def _decode(self, roi: RoiJob, source: IngestedSource) -> DecodedRange:
        roi.on_stage(STAGE_REGISTERING, 0.0)
        request, geometry, ffmpeg = roi.request, source.geometry, self.config.clarify.ffmpeg
        prefilters = roi_prefilters(roi.steps)
        started = self.config.now()
        stack = await decode_roi_frames(
            ffmpeg, source.work, request.first_frame, request.last_frame, geometry.width, geometry.height,
            prefilters, run=self.config.media.run,
        )  # fmt: skip
        argv = build_roi_decode_command(ffmpeg, source.work, request.first_frame, request.last_frame, prefilters)
        frames = roi_frames_from(stack, request.first_frame, source.frames)
        return DecodedRange(frames, self._process(DECODE_LABEL, argv, started, len(frames)))

    async def _register(self, roi: RoiJob, decoded: DecodedRange) -> RegistrationResult:
        request, box = roi.request, roi.box
        settings = RegistrationSettings(ecc_min=self.ecc_min)
        motion = motion_for(request.kind, box)
        registration = await asyncio.to_thread(
            register_roi_frames, decoded.frames, request.reference_frame, box, motion, settings
        )
        roi.on_stage(STAGE_REGISTERING, 1.0)
        return registration

    async def _fuse(
        self,
        roi: RoiJob,
        source: IngestedSource,
        decoded: DecodedRange,
        registration: RegistrationResult,
        settings: FusionSettings,
    ) -> FusionOutcome:
        roi.on_stage(STAGE_FUSING, 0.0)
        fusion = await asyncio.to_thread(fuse_roi, decoded.frames, registration, roi.box, settings)
        files = await asyncio.to_thread(write_fusion_files, roi, registration, fusion, settings.scale)
        stack_run = await self._encode_stack(roi, fusion, files["stack"])
        density = roi_density(roi.request.kind, roi.box, source.geometry.sar)
        clipped = clipped_frames_pct(decoded.frames, roi.box)
        notices = fusion_notices(registration, density, clipped)
        roi.on_stage(STAGE_FUSING, 1.0)
        return FusionOutcome(registration, fusion, density, clipped, notices, files, stack_run)

    async def _encode_stack(self, roi: RoiJob, fusion: RoiFusion, output: Path) -> ProcessRun:
        raw = roi.work_dir / STACK_RAW_NAME
        await asyncio.to_thread(raw.write_bytes, stack_gray16le(fusion.stack))
        height, width = fusion.fused.shape
        argv = build_stack_command(self.config.clarify.ffmpeg, raw, width, height, output)
        started = self.config.now()
        await run_pass(STACK_PASS, argv, self.config.media.run, STACK_TIMEOUT_SECONDS)
        return self._process(STACK_LABEL, argv, started, len(fusion.stack))

    async def _report(
        self, roi: RoiJob, source: IngestedSource, outcome: FusionOutcome, decoded: DecodedRange, facts: RoiReportFacts
    ) -> None:
        roi.on_stage(STAGE_REPORTING, 0.0)
        parts = report_parts(roi, source, outcome, decoded, facts)
        report = await asyncio.to_thread(build_report, parts)
        await asyncio.to_thread(write_report, report, roi.job_dir, render_report_html)
        await asyncio.to_thread(finish_files, roi.job_dir)
        roi.on_stage(STAGE_REPORTING, 1.0)

    def _process(self, label: str, argv: list[str], started: datetime, frames: int) -> ProcessRun:
        return ProcessRun(label, tuple(argv), started, self.config.now(), 0, frames_in=frames, frames_out=frames)
