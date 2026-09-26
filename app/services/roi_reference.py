"""Cuadro de referencia sugerido para la fusion de ROI (spec §4.9 paso 1, "Suggest reference frame").

Decodifica el rango de la sesion con los mismos prefiltros que el job (solo deinterlace y
deblock) y elige el cuadro con mas varianza del Laplaciano dentro de la caja entre los que
tienen menos del 5% de pixeles saturados. El usuario lo acepta o elige otro.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.cctv_chain import ResolvedStep, steps_from_request
from app.services.cctv_job_validation import check_filters_available, check_roi_box, check_roi_range, check_roi_steps
from app.services.cctv_preview import PreviewSource
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.ffmpeg_filters import build_filter
from app.services.process_runner import run_guarded_process
from app.services.roi_frames import decode_roi_frames, roi_frames_from
from app.services.roi_registration import RoiBox, suggest_reference_frame
from app.services.video_analysis import ProcessRunner

ROI_TASK = "roi_fusion"
ROI_LANE = "classic"


@dataclass(frozen=True, slots=True)
class ReferenceRequest:
    first_frame: int
    last_frame: int
    box: tuple[int, int, int, int]
    steps: tuple[Any, ...] = ()


def resolved_roi_steps(raw_steps: Sequence[Any], caps: FfmpegCapabilities) -> tuple[ResolvedStep, ...]:
    steps = steps_from_request(list(raw_steps), ROI_LANE)
    check_roi_steps(ROI_TASK, steps)
    check_filters_available(steps, caps)
    return steps


def check_reference_request(request: ReferenceRequest, source: PreviewSource, max_frames: int) -> None:
    first = request.first_frame
    check_roi_range(first, request.last_frame, first, source.frame_count, max_frames)
    check_roi_box(request.box, source.geometry)


async def suggest_session_reference(
    ffmpeg: Path,
    source: PreviewSource,
    request: ReferenceRequest,
    steps: Sequence[ResolvedStep],
    run: ProcessRunner = run_guarded_process,
) -> int:
    first, last, geometry = request.first_frame, request.last_frame, source.geometry
    prefilters = tuple(build_filter(step) for step in steps)
    stack = await decode_roi_frames(ffmpeg, source.work, first, last, geometry.width, geometry.height, prefilters, run=run)
    frames = roi_frames_from(stack, first, source.frames)
    return await asyncio.to_thread(suggest_reference_frame, frames, RoiBox(*request.box))
