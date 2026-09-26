"""Plan del carril IA de CCTV (spec §4.7): que decodifica el source, a que tamaño y a que tasa.

Todo sale de la cadena resuelta, de la geometria de `work.mkv` y de su indice de
cuadros, antes de abrir ningun proceso: el source lee `ancho*alto*3` bytes por
cuadro y con las dimensiones equivocadas el decode sale truncado.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.services.cctv_chain import AiLanePlan, ResolvedStep, in_catalog_order
from app.services.cctv_clarify_runner import atrim_filter, trim_step, trimmed_count
from app.services.cctv_frame_index import FrameEntry, FrameIndexSummary
from app.services.cctv_ingest import SourceRecord
from app.services.engines.frame_restorer import ComposedStageReport
from app.services.ffmpeg_filters import Box, FrameGeometry, build_filter, osd_boxes_after, output_dims_after
from app.services.media_tools import parse_fps_fraction

if TYPE_CHECKING:
    from app.services.cctv_job_runner import IngestedSource

AI_LANE = "ai"
ENHANCED_STEM = "enhanced"
# trim conserva los PTS del original: sin rebasarlos, -fps_mode cfr rellena desde 0 con copias del primer cuadro.
REBASE_PTS = "setpts=PTS-STARTPTS"
REBASE_AUDIO_PTS = "asetpts=PTS-STARTPTS"
FIELD_RATE_MODE = "send_field"
FIELDS_PER_FRAME = 2
# Matroska guarda los PTS en milisegundos: la mediana de los deltas no es mas fina que eso.
PTS_RESOLUTION_SECONDS = 0.001
MAX_RATE_DENOMINATOR = 1001
MIN_AI_UPSCALE = 2
NO_RATE_MESSAGE = "The frame rate of this video could not be measured and its header does not declare one."
UPSCALE_WITHOUT_SCALE_MESSAGE = "AI upscale needs a scale of 2 or more."


@dataclass(frozen=True, slots=True)
class EnhancePlan:
    prefilter_args: tuple[str, ...]
    decoded: FrameGeometry
    rate: Fraction
    frames_in: int
    strength: int | None
    upscale: int
    osd_boxes: tuple[Box, ...]

    @property
    def rate_text(self) -> str:
        return f"{self.rate.numerator}/{self.rate.denominator}"

    @property
    def output_size(self) -> tuple[int, int]:
        return self.decoded.width * self.upscale, self.decoded.height * self.upscale


@dataclass(frozen=True, slots=True)
class EnhanceSource:
    geometry: FrameGeometry
    index: FrameIndexSummary
    header_rate: str | None


def enhance_source(source: IngestedSource) -> EnhanceSource:
    index = source.ingest.index
    if index is None:
        raise RuntimeError("The working copy has no frame index.")
    return EnhanceSource(source.geometry, index.summary, source.ingest.video.header_rate)


def step_of(steps: Sequence[ResolvedStep], step_id: str) -> ResolvedStep | None:
    return next((step for step in steps if step.id == step_id), None)


def step_filters(step: ResolvedStep) -> tuple[str, ...]:
    built = build_filter(step)
    return (built, REBASE_PTS) if step.id == "trim" else (built,)


def decode_filters(steps: Sequence[ResolvedStep]) -> list[str]:
    return [text for step in in_catalog_order(steps) for text in step_filters(step)]


def prefilter_args(steps: Sequence[ResolvedStep]) -> tuple[str, ...]:
    filters = decode_filters(steps)
    return ("-vf", ",".join(filters)) if filters else ()


def fields_per_frame(steps: Sequence[ResolvedStep]) -> int:
    deinterlace = step_of(steps, "deinterlace")
    sends_fields = deinterlace is not None and deinterlace.params.get("mode") == FIELD_RATE_MODE
    return FIELDS_PER_FRAME if sends_fields else 1


def rate_tolerance(median_delta: float) -> float:
    return PTS_RESOLUTION_SECONDS / median_delta


def agrees_with_header(measured: float, median_delta: float, header: Fraction) -> bool:
    return abs(measured - float(header)) / float(header) <= rate_tolerance(median_delta)


def snapped_rate(measured: float, median_delta: float, header: Fraction | None) -> Fraction:
    # Dentro de la resolucion de los PTS el header es el valor exacto (30000/1001, no 30.303).
    if header is not None and agrees_with_header(measured, median_delta, header):
        return header
    return Fraction(measured).limit_denominator(MAX_RATE_DENOMINATOR)


def source_rate(index: FrameIndexSummary, header_rate: str | None) -> Fraction:
    header = parse_fps_fraction(header_rate)
    if index.measured_fps is not None and index.median_delta:
        return snapped_rate(index.measured_fps, index.median_delta, header)
    if header is None:
        raise RuntimeError(NO_RATE_MESSAGE)
    return header


def deblock_strength(composite: Sequence[ResolvedStep]) -> int | None:
    step = step_of(composite, "ai_deblock")
    return None if step is None else int(step.params["strength"])


def upscale_factor(composite: Sequence[ResolvedStep], scale: int) -> int:
    if step_of(composite, "ai_upscale") is None:
        return 1
    if scale < MIN_AI_UPSCALE:
        raise RuntimeError(UPSCALE_WITHOUT_SCALE_MESSAGE)
    return scale


def decoded_osd_boxes(lane: AiLanePlan, boxes: Sequence[Sequence[int]], geometry: FrameGeometry) -> tuple[Box, ...]:
    if step_of(lane.composite, "osd_protect") is None or not boxes:
        return ()
    return osd_boxes_after(lane.decode, [tuple(box) for box in boxes], geometry)


def decoded_frames(lane: AiLanePlan, frame_count: int) -> int:
    return trimmed_count(frame_count, trim_step(lane.decode)) * fields_per_frame(lane.decode)


def build_enhance_plan(
    lane: AiLanePlan, source: EnhanceSource, osd_boxes: Sequence[Sequence[int]], scale: int
) -> EnhancePlan:
    geometry = source.geometry
    return EnhancePlan(
        prefilter_args=prefilter_args(lane.decode),
        decoded=output_dims_after(lane.decode, geometry.width, geometry.height, geometry.sar),
        rate=source_rate(source.index, source.header_rate) * fields_per_frame(lane.decode),
        frames_in=decoded_frames(lane, source.index.frame_count),
        strength=deblock_strength(lane.composite),
        upscale=upscale_factor(lane.composite, scale),
        osd_boxes=decoded_osd_boxes(lane, osd_boxes, geometry),
    )


def enhanced_output_path(directory: Path, container: str) -> Path:
    return directory / f"{ENHANCED_STEM}.{container}"


def audio_trim(decode: Sequence[ResolvedStep], frames: Sequence[FrameEntry]) -> str | None:
    trim = trim_step(decode)
    return None if trim is None else atrim_filter(trim, frames)


def audio_trim_filter(atrim: str | None) -> list[str]:
    return [] if atrim is None else ["-af", f"{atrim},{REBASE_AUDIO_PTS}"]


def build_audio_command(ffmpeg: Path, work: Path, atrim: str | None, output: Path) -> list[str]:
    return [
        str(ffmpeg), "-y", "-v", "error", "-i", str(work), "-map", "0:a:0", "-vn",
        *audio_trim_filter(atrim), "-c:a", "aac", "-b:a", "192k", str(output),
    ]  # fmt: skip


def source_json(record: SourceRecord) -> dict[str, Any]:
    return {
        "sourceSha256": record.sha256,
        "receivedAt": {"utc": record.received_at.utc, "local": record.received_at.local},
    }


def restore_json(report: ComposedStageReport) -> dict[str, Any] | None:
    restore = report.restore
    if restore is None:
        return None
    return {
        "model": restore.model_id,
        "device": restore.device,
        "precision": restore.precision,
        "tile": restore.tile,
        "ioBinding": restore.io_binding,
    }


def stream_json(plan: EnhancePlan, report: ComposedStageReport, frames_out: int) -> dict[str, Any]:
    width, height = plan.output_size
    return {
        "framesIn": plan.frames_in,
        "framesOut": frames_out,
        "measuredFps": plan.rate_text,
        "cfrNormalized": True,
        "duplicatesReused": report.duplicates_reused,
        "prefilter": plan.prefilter_args[1] if plan.prefilter_args else None,
        "decodedSize": [plan.decoded.width, plan.decoded.height],
        "outputSize": [width, height],
        "upscaled": report.upscaled,
        "osdBoxes": report.osd_boxes,
        "restore": restore_json(report),
    }


def enhance_metadata(
    base: Mapping[str, Any], stream: Mapping[str, Any], enhanced: str, warnings: Sequence[str]
) -> dict[str, Any]:
    outputs = {**dict(base.get("outputs") or {}), "enhanced": enhanced}
    unique_warnings = list(dict.fromkeys(warnings))
    return {**dict(base), "lane": AI_LANE, **dict(stream), "outputs": outputs, "warnings": unique_warnings}
