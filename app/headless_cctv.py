"""Modo CCTV en proceso: probe, clarify, foto multi-cuadro (roi) y verificacion, sin servidor."""

from __future__ import annotations

import asyncio
import re
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from app.config import Settings
from app.headless_base import (
    HeadlessContext,
    HeadlessError,
    InferenceError,
    ModelNotInstalledError,
    UsageError,
    existing_file,
)
from app.models import CctvOptions, CctvStep, RoiFusionRequest, VideoUpscaleJob
from app.services.cctv_analysis import (
    FFMPEG_UNAVAILABLE,
    AnalysisRequest,
    AnalysisTools,
    CctvAnalysisError,
    analysis_tools,
    discard_session,
    new_session,
    run_session_analysis,
    upload_destination,
)
from app.services.cctv_chain import CctvChainError
from app.services.cctv_job_validation import ROI_FRAMES, ROI_PREFILTER_STEPS, check_roi_choices
from app.services.cctv_ingest import MediaTools
from app.services.cctv_presets import CCTV_PRESETS, PresetContext, preset_steps
from app.services.cctv_preview import load_preview_source
from app.services.cctv_report import REPORT_HTML_NAME, REPORT_JSON_NAME
from app.services.cctv_session import cctv_job_dir
from app.services.cctv_session import session_dir as cctv_session_dir
from app.services.device_semaphores import DeviceSemaphores
from app.services.ffmpeg_capabilities import cached_capabilities
from app.services.ffmpeg_filters import Box
from app.services.frame_export import StillFrameError
from app.services.handover_package import check_files_unchanged
from app.services.osd_check import OsdSelection, validate_osd_decision
from app.services.roi_reference import (
    ReferenceRequest,
    check_reference_request,
    resolved_roi_steps,
    suggest_session_reference,
)

if TYPE_CHECKING:
    from app.services.video_job_manager import VideoJobManager

CCTV_CLARIFY_TASK = "clarify"
CCTV_ROI_TASK = "roi_fusion"
CCTV_CLASSIC_LANE = "classic"
CCTV_JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


@dataclass(frozen=True, slots=True)
class CctvClarifyChoices:
    preset: str | None = None
    steps: tuple[Mapping[str, Any], ...] | None = None
    osd_boxes: tuple[Box, ...] = ()
    osd_confirmed: bool = False
    no_osd: bool = False
    trim: tuple[int, int] | None = None
    still_frames: tuple[int, ...] = ()
    acquisition: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class CctvRoiChoices:
    roi: RoiFusionRequest
    preset: str | None = None
    steps: tuple[Mapping[str, Any], ...] | None = None
    acquisition: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    # Sin --ref: roi.reference_frame es un marcador y se pide "Suggest reference frame" con la sesion.
    suggest_reference: bool = False


AnyCctvChoices = CctvClarifyChoices | CctvRoiChoices
StepPicker = Callable[[Sequence[Mapping[str, Any]]], tuple[Mapping[str, Any], ...]]
CctvResultDescriber = Callable[[VideoUpscaleJob, Path, float], dict[str, Any]]


def checked_job_id(job_id: str) -> str:
    if not CCTV_JOB_ID.fullmatch(job_id):
        raise UsageError(f"invalid job id {job_id!r}")
    return job_id


def cctv_error(exc: BaseException) -> HeadlessError:
    if isinstance(exc, HeadlessError):
        return exc
    if isinstance(exc, CctvAnalysisError):
        return analysis_headless_error(exc)
    if isinstance(exc, CctvChainError):
        return UsageError(str(exc), key=exc.code)
    if isinstance(exc, StillFrameError):
        return UsageError(str(exc), key=exc.key)
    return InferenceError(str(exc), key=getattr(exc, "key", None))


def analysis_headless_error(exc: CctvAnalysisError) -> HeadlessError:
    if exc.key == FFMPEG_UNAVAILABLE:
        return ModelNotInstalledError(str(exc), key=exc.key)
    if exc.status >= 500:
        return InferenceError(str(exc), key=exc.key)
    return UsageError(str(exc), key=exc.key)


def check_osd_decision(choices: CctvClarifyChoices) -> None:
    try:
        validate_osd_decision(OsdSelection(choices.osd_boxes, choices.osd_confirmed, choices.no_osd))
    except CctvChainError as exc:
        raise UsageError(str(exc), key=exc.code) from exc


def check_roi_request(roi: RoiFusionRequest) -> None:
    try:
        check_roi_choices(roi)
    except CctvChainError as exc:
        raise UsageError(str(exc), key=exc.code) from exc
    if not 0 <= roi.first_frame <= roi.reference_frame <= roi.last_frame:
        raise UsageError(
            f"the reference frame {roi.reference_frame} must be inside the range {roi.first_frame}:{roi.last_frame}",
            key=ROI_FRAMES,
        )


def check_out_dir(out_dir: Path | None) -> None:
    if out_dir is not None and Path(out_dir).exists() and not Path(out_dir).is_dir():
        raise UsageError(f"the output folder is a file: {out_dir}")


# --- Pasos del preset, con el mismo contexto que arma la UI (cctvSteps.presetContextOf)


def parse_sample_aspect(sar: object) -> tuple[int, int] | None:
    parts = str(sar or "").split(":")
    if len(parts) != 2 or not all(part.isdigit() and int(part) > 0 for part in parts):
        return None
    return int(parts[0]), int(parts[1])


def preset_context_of(analysis: Mapping[str, Any]) -> PresetContext:
    interlace = (analysis.get("quality") or {}).get("interlace") or {}
    lite = (analysis.get("video") or {}).get("lite") or {}
    return PresetContext(interlaced=bool(interlace.get("interlaced")), sample_aspect=parse_sample_aspect(lite.get("sar")))


def classic_preset_steps(analysis: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    context = preset_context_of(analysis)
    return {preset.id: preset_steps(preset.id, CCTV_CLASSIC_LANE, context) for preset in CCTV_PRESETS}


def steps_for_preset(analysis: Mapping[str, Any], preset: str) -> tuple[dict[str, Any], ...]:
    try:
        return tuple(preset_steps(preset, CCTV_CLASSIC_LANE, preset_context_of(analysis)))
    except CctvChainError as exc:
        raise UsageError(str(exc), key=exc.code) from exc


def with_preset_steps(analysis: Mapping[str, Any]) -> dict[str, Any]:
    return {**analysis, "presetSteps": classic_preset_steps(analysis)}


def require_cctv_mode(analysis: Mapping[str, Any]) -> None:
    if analysis.get("modeAvailable") is False:
        reason = analysis.get("modeUnavailableReason")
        raise ModelNotInstalledError(f"this ffmpeg build cannot run CCTV mode ({reason})", key=reason)


def with_steps_of_preset(analysis: Mapping[str, Any], choices: AnyCctvChoices, pick: StepPicker) -> AnyCctvChoices:
    if choices.steps is not None:
        return choices
    preset = choices.preset or analysis.get("suggestedPreset")
    if preset is None:
        return replace(choices, steps=())
    return replace(choices, preset=preset, steps=pick(steps_for_preset(analysis, preset)))


def choices_with_preset_steps(analysis: Mapping[str, Any], choices: CctvClarifyChoices) -> CctvClarifyChoices:
    return with_steps_of_preset(analysis, choices, tuple)


def roi_prefilter_steps(steps: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(step for step in steps if step["id"] in ROI_PREFILTER_STEPS)


def roi_choices_with_steps(analysis: Mapping[str, Any], choices: CctvRoiChoices) -> CctvRoiChoices:
    return with_steps_of_preset(analysis, choices, roi_prefilter_steps)


def require_steps_for_preset(choices: AnyCctvChoices) -> None:
    if choices.steps is None and choices.preset:
        raise UsageError(
            f"pass the steps of preset {choices.preset!r}: copy presetSteps[{choices.preset!r}] from the CCTV probe"
        )


# --- Opciones del job


def normalized_step(raw: object) -> dict[str, Any]:
    step_id = raw.get("id") if isinstance(raw, Mapping) else None
    params = (raw.get("params") or {}) if isinstance(raw, Mapping) else None
    if not isinstance(step_id, str) or not isinstance(params, Mapping):
        raise UsageError(f"each CCTV step needs an 'id' and an optional 'params' object, got {raw!r}")
    return {"id": step_id, "params": dict(params)}


def cctv_step_of(raw: object) -> CctvStep:
    step = normalized_step(raw)
    return CctvStep(step["id"], MappingProxyType(step["params"]))


def cctv_options_of(token: str, choices: CctvClarifyChoices) -> CctvOptions:
    return CctvOptions(
        task=CCTV_CLARIFY_TASK,
        session_token=token,
        preset=choices.preset,
        steps=tuple(cctv_step_of(step) for step in choices.steps or ()),
        osd_boxes=tuple(tuple(box) for box in choices.osd_boxes),
        osd_boxes_confirmed=choices.osd_confirmed,
        no_osd=choices.no_osd,
        trim=choices.trim,
        still_frames=choices.still_frames,
        acquisition=MappingProxyType(dict(choices.acquisition)),
    )


def cctv_roi_options_of(token: str, choices: CctvRoiChoices) -> CctvOptions:
    return CctvOptions(
        task=CCTV_ROI_TASK,
        session_token=token,
        preset=choices.preset,
        steps=tuple(cctv_step_of(step) for step in choices.steps or ()),
        roi=choices.roi,
        acquisition=MappingProxyType(dict(choices.acquisition)),
    )


# --- Sesion, job y entrega


def cctv_analysis_tools(settings: Settings) -> AnalysisTools:
    ffmpeg = settings.ffmpeg_binary_path
    return analysis_tools(ffmpeg, settings.ffprobe_binary_path, lambda: cached_capabilities(ffmpeg))


def open_cctv_session(work_root: Path, source: Path) -> AnalysisRequest:
    token, directory = new_session(work_root)
    try:
        upload = upload_destination(directory, source.name)
        shutil.copy2(source, upload)
    except BaseException:
        discard_session(directory)
        raise
    return AnalysisRequest(token, directory, upload)


def cctv_job_manager(ctx: HeadlessContext) -> VideoJobManager:
    # Import diferido: el pipeline de video suma ~1,7 s al arranque de cada comando de la CLI.
    from app.services.cctv_job_runner import build_cctv_runners
    from app.services.media_tools import MediaTools as VideoMediaTools
    from app.services.video_job_manager import VideoJobManager
    from app.services.video_upscaler import VideoUpscaler

    settings = ctx.settings
    media = VideoMediaTools(settings)
    upscaler = VideoUpscaler(
        settings, ctx.ncnn_engine, media, devices=ctx.devices, cctv_runners=build_cctv_runners(settings)
    )
    semaphores = DeviceSemaphores(settings, resource_probes=ctx.probes)
    return VideoJobManager(settings, upscaler, media, semaphores, registry=ctx.registry, devices=ctx.devices)


def deliver_cctv_outputs(job_dir: Path, out_dir: Path | None) -> Path:
    if out_dir is None:
        return job_dir
    destination = Path(out_dir).expanduser().resolve() / job_dir.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(job_dir), str(destination))
    return destination


def describe_cctv_result(job: VideoUpscaleJob, result_dir: Path, seconds: float) -> dict[str, Any]:
    meta = job.metadata.get("cctv") or {}
    return {
        "ok": True,
        "jobId": job.id,
        "task": job.cctv.task,
        "lane": meta.get("lane"),
        "preset": job.cctv.preset,
        "outputDir": str(result_dir),
        "sourceSha256": meta.get("sourceSha256"),
        "receivedAt": meta.get("receivedAt"),
        "framesIn": meta.get("framesIn"),
        "framesOut": meta.get("framesOut"),
        "outputs": meta.get("outputs", {}),
        "report": str(result_dir / REPORT_HTML_NAME),
        "reportJson": str(result_dir / REPORT_JSON_NAME),
        "warnings": list(meta.get("warnings", [])),
        "seconds": round(seconds, 2),
    }


def describe_roi_result(job: VideoUpscaleJob, result_dir: Path, seconds: float) -> dict[str, Any]:
    return {**describe_cctv_result(job, result_dir, seconds), "roi": (job.metadata.get("cctv") or {}).get("roi")}


async def cctv_probe(ctx: HeadlessContext, source: Path, *, keep_session: bool = False) -> dict[str, Any]:
    source = existing_file(source)
    request = await asyncio.to_thread(open_cctv_session, ctx.settings.video_work_path, source)
    try:
        analysis = await run_session_analysis(request, cctv_analysis_tools(ctx.settings))
    except Exception as exc:  # noqa: BLE001 - run_session_analysis ya borro la sesion
        raise cctv_error(exc) from exc
    if not keep_session:
        await asyncio.to_thread(discard_session, request.directory)
        analysis = {**analysis, "token": None}
    return {"ok": True, **with_preset_steps(analysis)}


async def cctv_clarify(
    ctx: HeadlessContext, token: str, choices: CctvClarifyChoices, out_dir: Path | None = None
) -> dict[str, Any]:
    check_osd_decision(choices)
    require_steps_for_preset(choices)
    check_out_dir(out_dir)
    return await run_cctv_job(ctx, cctv_options_of(token, choices), out_dir, describe_cctv_result)


async def cctv_roi(
    ctx: HeadlessContext, token: str, choices: CctvRoiChoices, out_dir: Path | None = None
) -> dict[str, Any]:
    check_roi_request(choices.roi)
    require_steps_for_preset(choices)
    check_out_dir(out_dir)
    chosen = await with_suggested_reference(ctx, token, choices)
    return await run_cctv_job(ctx, cctv_roi_options_of(token, chosen), out_dir, describe_roi_result)


async def with_suggested_reference(ctx: HeadlessContext, token: str, choices: CctvRoiChoices) -> CctvRoiChoices:
    if not choices.suggest_reference:
        return choices
    try:
        frame = await suggested_reference(ctx.settings, token, choices)
    except Exception as exc:  # noqa: BLE001 - la CLI traduce cualquier fallo a un codigo estable
        raise cctv_error(exc) from exc
    return replace(choices, roi=replace(choices.roi, reference_frame=frame), suggest_reference=False)


async def suggested_reference(settings: Settings, token: str, choices: CctvRoiChoices) -> int:
    roi, ffmpeg = choices.roi, settings.ffmpeg_binary_path
    request = ReferenceRequest(roi.first_frame, roi.last_frame, tuple(roi.box), tuple(choices.steps or ()))
    source = await load_preview_source(settings.video_work_path, token, MediaTools(ffmpeg, settings.ffprobe_binary_path))
    check_reference_request(request, source, settings.cctv_roi_max_frames)
    steps = resolved_roi_steps(request.steps, await asyncio.to_thread(cached_capabilities, ffmpeg))
    return await suggest_session_reference(ffmpeg, source, request, steps)


async def run_cctv_job(
    ctx: HeadlessContext, options: CctvOptions, out_dir: Path | None, describe: CctvResultDescriber
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        job = await cctv_job_manager(ctx).run_cctv_inline(cctv=options)
    except Exception as exc:  # noqa: BLE001 - la CLI traduce cualquier fallo a un codigo estable
        raise cctv_error(exc) from exc
    job_dir = cctv_job_dir(ctx.settings.outputs_path, job.id)
    result_dir = await asyncio.to_thread(deliver_cctv_outputs, job_dir, out_dir)
    return describe(job, result_dir, time.perf_counter() - started)


async def on_probed_clip(
    ctx: HeadlessContext, source: Path, run: Callable[[Mapping[str, Any], str], Awaitable[dict[str, Any]]]
) -> dict[str, Any]:
    analysis = await cctv_probe(ctx, source, keep_session=True)
    token = analysis["token"]
    try:
        require_cctv_mode(analysis)
        return await run(analysis, token)
    finally:
        # Sin sweeper en modo headless: la sesion solo servia para este job.
        await asyncio.to_thread(discard_session, cctv_session_dir(ctx.settings.video_work_path, token))


async def cctv_clarify_file(
    ctx: HeadlessContext, source: Path, out_dir: Path, choices: CctvClarifyChoices
) -> dict[str, Any]:
    check_osd_decision(choices)
    check_out_dir(out_dir)

    async def clarify(analysis: Mapping[str, Any], token: str) -> dict[str, Any]:
        return await cctv_clarify(ctx, token, choices_with_preset_steps(analysis, choices), out_dir)

    return await on_probed_clip(ctx, source, clarify)


async def cctv_roi_file(ctx: HeadlessContext, source: Path, out_dir: Path, choices: CctvRoiChoices) -> dict[str, Any]:
    check_roi_request(choices.roi)
    check_out_dir(out_dir)

    async def fuse(analysis: Mapping[str, Any], token: str) -> dict[str, Any]:
        return await cctv_roi(ctx, token, roi_choices_with_steps(analysis, choices), out_dir)

    return await on_probed_clip(ctx, source, fuse)


def cctv_result_dir(ctx: HeadlessContext, job_id: str) -> Path:
    return cctv_job_dir(ctx.settings.outputs_path, checked_job_id(job_id))


def cctv_check_unchanged(directory: Path) -> dict[str, Any]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise UsageError(f"CCTV result folder not found: {directory}")
    return {"directory": str(directory), **check_files_unchanged(directory).to_json()}
