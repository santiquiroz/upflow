"""Validacion de un job CCTV antes de encolarlo (spec §5.3, `_validate_cctv`).

Funciones puras: reciben las opciones del pedido y los hechos de la sesion
(geometria del cuadro, conteo del indice, capacidades de la build de ffmpeg) y
devuelven la cadena resuelta o un `CctvChainError` con su clave, que la ruta
convierte en 400. El manager solo junta los hechos y llama a `plan_cctv_job`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from app.models import CctvOptions, RoiFusionRequest
from app.services.cctv_chain import CctvChainError, Lane, ResolvedStep, steps_from_request
from app.services.cctv_presets import preset_spec
from app.services.cctv_report_model import Acquisition, CaseInfo
from app.services.devices_service import CPU_DEVICE_ID
from app.services.ffmpeg_capabilities import (
    FILTER_UNAVAILABLE,
    FfmpegCapabilities,
    mode_unavailable_reason,
    unavailable_filters,
)
from app.services.ffmpeg_filters import FrameGeometry, output_dims_after
from app.services.frame_export import StillFrameError, checked_still_frames, TOO_MANY_STILL_FRAMES
from app.services.missing_pack import missing_pack_message
from app.services.osd_check import OsdSelection, validate_osd_selection

UNKNOWN_TASK = "cctv.error.unknownTask"
TASK_UNAVAILABLE = "cctv.error.taskUnavailable"
MODE_UNAVAILABLE = "cctv.error.modeUnavailable"
AI_CPU_BLOCKED = "cctv.ai.cpuBlocked"
AI_PACK_MISSING = "cctv.error.aiPackMissing"
TRIM_OUT_OF_RANGE = "cctv.error.trimOutOfRange"
INVALID_ACQUISITION = "cctv.error.invalidCaseDetails"
ROI_REQUIRED = "cctv.error.roiRequired"
ROI_OUTSIDE_FRAME = "cctv.error.roiOutsideFrame"
ROI_ODD = "cctv.error.roiOdd"
ROI_FRAMES = "cctv.error.roiFrames"
ROI_UNEXPECTED = "cctv.error.roiUnexpected"
ROI_INVALID = "cctv.error.roiInvalid"

TASK_LANES: Mapping[str, Lane] = {"clarify": "classic", "enhance": "ai", "roi_fusion": "classic"}
VIDEO_TASKS = frozenset({"clarify", "enhance"})
CPU_TASKS = frozenset({"clarify", "roi_fusion"})
OSD_STEP = "osd_protect"
TRIM_STEP = "trim"
AI_DEBLOCK_STEP = "ai_deblock"
AI_DEBLOCK_PACK = "restore-core"
ROI_SCALES = frozenset({2, 3, 4})
ROI_KINDS = frozenset({"plate", "face_or_object"})
ROI_METHODS = frozenset({"median", "trimmed_mean"})


@dataclass(frozen=True, slots=True)
class CctvJobFacts:
    geometry: FrameGeometry
    frame_count: int
    caps: FfmpegCapabilities
    restore_core_installed: bool
    task_available: bool
    max_still_frames: int
    max_roi_frames: int


@dataclass(frozen=True, slots=True)
class CctvJobPlan:
    lane: Lane
    steps: tuple[ResolvedStep, ...]
    first_frame: int
    last_frame: int
    still_frames: tuple[int, ...]
    acquisition: Acquisition
    case: CaseInfo
    device: str | None


# --- Tarea, carril y device ---


def lane_for_task(task: str) -> Lane:
    lane = TASK_LANES.get(task)
    if lane is None:
        raise CctvChainError(UNKNOWN_TASK, f"Unknown CCTV task {task!r}. Valid: {', '.join(TASK_LANES)}.")
    return lane


def device_for_task(task: str, device: str | None) -> str | None:
    # El carril clasico y la fusion de ROI son CPU puro: forzarlo evita que el job espere un permiso de GPU.
    return CPU_DEVICE_ID if task in CPU_TASKS else device


def check_preset(preset: str | None) -> None:
    if preset is not None:
        preset_spec(preset)


def check_task_available(task: str, available: bool) -> None:
    if not available:
        raise CctvChainError(TASK_UNAVAILABLE, f"The CCTV task {task!r} is not available in this version.")


# --- Pasos ---


def raw_step(step_id: str, params: Mapping[str, Any]) -> dict[str, Any]:
    return {"id": step_id, "params": dict(params)}


def trim_raw_step(trim: tuple[int, int] | None) -> list[dict[str, Any]]:
    if trim is None:
        return []
    return [raw_step(TRIM_STEP, {"start_frame": trim[0], "end_frame": trim[1]})]


def osd_raw_step(options: CctvOptions) -> list[dict[str, Any]]:
    # osd_protect existe si y solo si hay cajas: asi el informe nunca lista un paso que no hizo nada.
    return [raw_step(OSD_STEP, {})] if options.osd_boxes else []


def requested_raw_steps(options: CctvOptions) -> list[dict[str, Any]]:
    chosen = [raw_step(step.id, step.params) for step in options.steps if step.id != OSD_STEP]
    return [*chosen, *trim_raw_step(options.trim), *osd_raw_step(options)]


def resolve_steps(options: CctvOptions, lane: Lane) -> tuple[ResolvedStep, ...]:
    return steps_from_request(requested_raw_steps(options), lane)


def check_mode_available(caps: FfmpegCapabilities) -> None:
    reason = mode_unavailable_reason(caps)
    if reason is not None:
        raise CctvChainError(MODE_UNAVAILABLE, reason)


def check_filters_available(steps: Sequence[ResolvedStep], caps: FfmpegCapabilities) -> None:
    missing = {(item.step_id, item.filter): item for item in unavailable_filters(caps)}
    for step in steps:
        item = missing.get((step.id, step.filter))
        if item is not None:
            raise CctvChainError(FILTER_UNAVAILABLE, f"Step {step.id!r}: {item.message}")


def check_geometry(steps: Sequence[ResolvedStep], geometry: FrameGeometry) -> None:
    output_dims_after(steps, geometry.width, geometry.height, geometry.sar)


# --- Recorte y cuadros ---


def trim_range(steps: Sequence[ResolvedStep], frame_count: int) -> tuple[int, int]:
    trim = next((step for step in steps if step.id == TRIM_STEP), None)
    if trim is None:
        return 0, frame_count - 1
    first, last = int(trim.params["start_frame"]), int(trim.params["end_frame"])
    if last >= frame_count:
        raise CctvChainError(TRIM_OUT_OF_RANGE, f"The trim ends at frame {last} but the video has {frame_count} frames.")
    return first, last


def still_frames_in(requested: Sequence[int], first: int, last: int, max_frames: int) -> tuple[int, ...]:
    if len(set(requested)) > max_frames:
        raise StillFrameError(TOO_MANY_STILL_FRAMES, f"At most {max_frames} still frames per job")
    return checked_still_frames(requested, first, last)


# --- OSD ---


def osd_selection(options: CctvOptions) -> OsdSelection:
    return OsdSelection(tuple(tuple(box) for box in options.osd_boxes), options.osd_boxes_confirmed, options.no_osd)


def check_osd(options: CctvOptions, geometry: FrameGeometry) -> None:
    if options.task in VIDEO_TASKS:
        validate_osd_selection(osd_selection(options), geometry.width, geometry.height)


# --- ROI ---


def check_roi_box(box: Sequence[int], geometry: FrameGeometry) -> None:
    x, y, w, h = box
    if x < 0 or y < 0 or w < 2 or h < 2 or x + w > geometry.width or y + h > geometry.height:
        raise CctvChainError(ROI_OUTSIDE_FRAME, f"The region {tuple(box)!r} is outside the {geometry.width}x{geometry.height} frame.")
    if w % 2 or h % 2:
        raise CctvChainError(ROI_ODD, f"The region {tuple(box)!r} must have an even width and height.")


def check_roi_frames(roi: RoiFusionRequest, frame_count: int, max_frames: int) -> None:
    in_index = 0 <= roi.first_frame <= roi.reference_frame <= roi.last_frame < frame_count
    count = roi.last_frame - roi.first_frame + 1
    if not in_index or count > max_frames:
        raise CctvChainError(
            ROI_FRAMES,
            f"Pick at most {max_frames} frames inside the video (0-{frame_count - 1}) with the reference inside the range.",
        )


def check_roi_choices(roi: RoiFusionRequest) -> None:
    if roi.kind not in ROI_KINDS or roi.scale not in ROI_SCALES or roi.method not in ROI_METHODS:
        raise CctvChainError(ROI_INVALID, "Region type, scale (2x, 3x or 4x) and combine method must be valid choices.")


def check_roi(options: CctvOptions, facts: CctvJobFacts) -> None:
    roi = options.roi
    if options.task != "roi_fusion":
        if roi is not None:
            raise CctvChainError(ROI_UNEXPECTED, "A region of interest only applies to the multi-frame still task.")
        return
    if roi is None:
        raise CctvChainError(ROI_REQUIRED, "The multi-frame still task needs a region of interest.")
    check_roi_choices(roi)
    check_roi_frames(roi, facts.frame_count, facts.max_roi_frames)
    check_roi_box(roi.box, facts.geometry)


# --- Carril IA ---


def check_ai_device(device: str | None) -> None:
    if device is None or device == CPU_DEVICE_ID:
        raise CctvChainError(AI_CPU_BLOCKED, "AI enhancement needs a GPU; it is not available on the CPU.")


def check_ai_packs(steps: Sequence[ResolvedStep], restore_core_installed: bool) -> None:
    if any(step.id == AI_DEBLOCK_STEP for step in steps) and not restore_core_installed:
        detail = "Se pidio el desbloqueo con IA en el modo CCTV."
        raise CctvChainError(AI_PACK_MISSING, missing_pack_message(AI_DEBLOCK_PACK, detail=detail))


def check_ai_lane(lane: Lane, steps: Sequence[ResolvedStep], device: str | None, packs_ready: bool) -> None:
    if lane == "ai":
        check_ai_device(device)
        check_ai_packs(steps, packs_ready)


# --- Datos del caso ---


def parse_case_details(options: CctvOptions) -> tuple[Acquisition, CaseInfo]:
    try:
        acquisition = Acquisition.model_validate(dict(options.acquisition))
        case = CaseInfo(case_label=options.case_label, operator_name=options.operator_name)
    except ValidationError as exc:
        raise CctvChainError(INVALID_ACQUISITION, f"Case details are not valid: {exc.errors()[0]['msg']}") from exc
    return acquisition, case


# --- Plan completo ---


def plan_cctv_job(options: CctvOptions, facts: CctvJobFacts, device: str | None) -> CctvJobPlan:
    lane = lane_for_task(options.task)
    check_preset(options.preset)
    steps = resolve_steps(options, lane)
    resolved_device = device_for_task(options.task, device)
    check_ai_lane(lane, steps, resolved_device, facts.restore_core_installed)
    check_mode_available(facts.caps)
    check_filters_available(steps, facts.caps)
    check_osd(options, facts.geometry)
    check_geometry(steps, facts.geometry)
    check_roi(options, facts)
    first, last = trim_range(steps, facts.frame_count)
    stills = still_frames_in(options.still_frames, first, last, facts.max_still_frames)
    acquisition, case = parse_case_details(options)
    check_task_available(options.task, facts.task_available)
    return CctvJobPlan(lane, steps, first, last, stills, acquisition, case, resolved_device)
