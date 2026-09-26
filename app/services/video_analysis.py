"""Diagnostico de un video CCTV (spec §4.3).

Pasadas de solo lectura con ffmpeg (`run_guarded_process`, patron de
`scene_cuts.py`): `idet` sobre los primeros 500 cuadros, `blockdetect` y
`blurdetect` sobre 1 de cada 25 cuadros (hasta 200), `freezedetect` sobre todo
el video y 50 cuadros espaciados decodificados por pipe `rawvideo` a numpy para
luma, croma y *clipping*. Los parsers se prueban contra stderr grabado del
binario vendorizado (`tests/fixtures/ffmpeg/`).

Los umbrales son provisionales: se recalibran con los exports reales del HiLook
(P2-VAL).
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from app.services.process_runner import run_guarded_process

PresetId = Literal["day", "night_ir", "analog", "low_res"]
FieldOrder = Literal["tff", "bff"]
ProcessRunner = Callable[[list[str], float], Awaitable[tuple[bytes, bytes, int]]]

IDET_MAX_FRAMES = 500
SAMPLE_EVERY_N_FRAMES = 25
SAMPLE_MAX_FRAMES = 200
FREEZE_NOISE = "-60dB"
FREEZE_MIN_SECONDS = 0.2
STATS_FRAME_COUNT = 50
STATS_MAX_PIXELS = 1280 * 720
PASS_TIMEOUT_SECONDS = 1800.0

INTERLACED_MIN_RATIO = 0.5
INTERLACE_MIN_DECIDED_FRAMES = 10
MONOCHROME_MAX_CHROMA_DEVIATION = 3.0
NIGHT_MAX_LUMA_MEAN = 100.0
CLIP_HIGH_LEVEL = 250
CLIP_LOW_LEVEL = 5
CLIPPING_WARN_PCT = 1.0
HEAVY_BLOCKING_MEDIAN = 50.0
HEAVY_BLUR_MEDIAN = 10.0
LOW_FPS_BELOW = 20.0
LOW_RES_MAX_SIZE = (704, 576)
CHROMA_NEUTRAL = 128

BLOCK_KEY = "lavfi.block"
BLUR_KEY = "lavfi.blur"

_IDET_MULTI = re.compile(
    r"Multi frame detection:\s*TFF:\s*(\d+)\s+BFF:\s*(\d+)\s+Progressive:\s*(\d+)\s+Undetermined:\s*(\d+)"
)
_FREEZE_EVENT = re.compile(r"lavfi\.freezedetect\.freeze_(start|end):\s*([-+\d.eE]+)")


class VideoAnalysisError(RuntimeError):
    def __init__(self, pass_name: str, detail: str) -> None:
        super().__init__(f"Video analysis pass '{pass_name}' failed: {detail}")
        self.pass_name = pass_name


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class SourceFacts(_Frozen):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    frame_count: int | None = Field(default=None, ge=0)
    measured_fps: float | None = Field(default=None, gt=0)
    is_vfr: bool = False
    is_lite: bool = False
    probable_duplicates: int = Field(default=0, ge=0)


class InterlaceReport(_Frozen):
    interlaced: bool
    field_order: FieldOrder | None = Field(default=None, serialization_alias="fieldOrder")
    tff: int
    bff: int
    progressive: int
    undetermined: int


class MetricSummary(_Frozen):
    samples: int
    mean: float
    median: float
    p90: float


class FreezeInterval(_Frozen):
    start: float
    end: float | None = None


class FrameStats(_Frozen):
    samples: int
    luma_mean: float = Field(serialization_alias="lumaMean")
    chroma_deviation: float = Field(serialization_alias="chromaDeviation")
    clipped_high_pct: float = Field(serialization_alias="clippedHighPct")
    clipped_low_pct: float = Field(serialization_alias="clippedLowPct")
    monochrome: bool


class CctvDiagnosis(_Frozen):
    interlace: InterlaceReport | None
    blocking: MetricSummary | None
    blur: MetricSummary | None
    freezes: list[FreezeInterval]
    frame_stats: FrameStats | None = Field(serialization_alias="frameStats")
    suggested_preset: PresetId = Field(serialization_alias="suggestedPreset")
    warnings: list[str]


def _decode_pass(ffmpeg: Path, video: Path, filters: str, *extra: str) -> list[str]:
    return [
        str(ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-nostats",
        *extra,
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        filters,
        # Sin esto el muxer puede duplicar cuadros en VFR y el limite de
        # -frames:v contaria copias en vez de muestras.
        "-fps_mode",
        "passthrough",
    ]


def _read_only_pass(ffmpeg: Path, video: Path, filters: str, frame_limit: int | None) -> list[str]:
    limit = ["-frames:v", str(frame_limit)] if frame_limit is not None else []
    return [*_decode_pass(ffmpeg, video, filters), *limit, "-f", "null", "-"]


def every_nth_frame(step: int) -> str:
    return f"select=not(mod(n\\,{step}))"


def build_idet_command(ffmpeg: Path, video: Path) -> list[str]:
    return _read_only_pass(ffmpeg, video, "idet", IDET_MAX_FRAMES)


def _sampled_metric_filters(detector: str, key: str) -> str:
    return f"{every_nth_frame(SAMPLE_EVERY_N_FRAMES)},{detector},metadata=mode=print:key={key}"


def build_blockdetect_command(ffmpeg: Path, video: Path) -> list[str]:
    return _read_only_pass(ffmpeg, video, _sampled_metric_filters("blockdetect", BLOCK_KEY), SAMPLE_MAX_FRAMES)


def build_blurdetect_command(ffmpeg: Path, video: Path) -> list[str]:
    return _read_only_pass(ffmpeg, video, _sampled_metric_filters("blurdetect", BLUR_KEY), SAMPLE_MAX_FRAMES)


def build_freezedetect_command(ffmpeg: Path, video: Path) -> list[str]:
    filters = f"freezedetect=n={FREEZE_NOISE}:d={FREEZE_MIN_SECONDS}"
    return _read_only_pass(ffmpeg, video, filters, None)


def stats_step(frame_count: int | None) -> int:
    if not frame_count:
        return SAMPLE_EVERY_N_FRAMES
    return max(1, frame_count // STATS_FRAME_COUNT)


def _even_at_least_two(value: float) -> int:
    return max(2, int(value) // 2 * 2)


def stats_dimensions(width: int, height: int) -> tuple[int, int]:
    factor = min(1.0, math.sqrt(STATS_MAX_PIXELS / (width * height)))
    return _even_at_least_two(width * factor), _even_at_least_two(height * factor)


def build_frame_stats_command(ffmpeg: Path, video: Path, facts: SourceFacts) -> list[str]:
    width, height = stats_dimensions(facts.width, facts.height)
    # neighbor submuestrea sin inventar valores y out_range=full deja los
    # umbrales de clipping (>=250, <=5) en la misma escala para fuentes TV y PC.
    filters = (
        f"{every_nth_frame(stats_step(facts.frame_count))},"
        f"scale={width}:{height}:flags=neighbor:out_range=full,format=yuv420p"
    )
    return [
        *_decode_pass(ffmpeg, video, filters, "-v", "error"),
        "-frames:v",
        str(STATS_FRAME_COUNT),
        "-f",
        "rawvideo",
        "-",
    ]


def parse_idet_counts(stderr: str) -> tuple[int, int, int, int] | None:
    matches = _IDET_MULTI.findall(stderr)
    if not matches:
        return None
    # ffmpeg arma el grafo dos veces y el primer idet se despide con todo en cero.
    tff, bff, progressive, undetermined = (int(value) for value in matches[-1])
    return tff, bff, progressive, undetermined


def interlace_report(tff: int, bff: int, progressive: int, undetermined: int) -> InterlaceReport:
    decided = tff + bff + progressive
    interlaced = decided >= INTERLACE_MIN_DECIDED_FRAMES and (tff + bff) / decided >= INTERLACED_MIN_RATIO
    return InterlaceReport(
        interlaced=interlaced,
        field_order=field_order(tff, bff) if interlaced else None,
        tff=tff,
        bff=bff,
        progressive=progressive,
        undetermined=undetermined,
    )


def field_order(tff: int, bff: int) -> FieldOrder:
    return "tff" if tff >= bff else "bff"


def parse_interlace(stderr: str) -> InterlaceReport | None:
    counts = parse_idet_counts(stderr)
    return interlace_report(*counts) if counts is not None else None


def _metadata_pattern(key: str) -> re.Pattern[str]:
    return re.compile(rf"\]\s*{re.escape(key)}=(\S+)")


def _finite_or_none(raw: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def parse_metadata_values(stderr: str, key: str) -> list[float]:
    values = (_finite_or_none(raw) for raw in _metadata_pattern(key).findall(stderr))
    return [value for value in values if value is not None]


def summarize(values: Sequence[float]) -> MetricSummary | None:
    if not values:
        return None
    data = np.asarray(values, dtype=np.float64)
    return MetricSummary(
        samples=int(data.size),
        mean=float(data.mean()),
        median=float(np.median(data)),
        p90=float(np.percentile(data, 90)),
    )


def _apply_freeze_event(
    closed: tuple[FreezeInterval, ...], open_start: float | None, kind: str, seconds: float
) -> tuple[tuple[FreezeInterval, ...], float | None]:
    if kind == "start":
        return closed, seconds
    if open_start is None:
        return closed, None
    return (*closed, FreezeInterval(start=open_start, end=seconds)), None


def parse_freezes(stderr: str) -> list[FreezeInterval]:
    closed: tuple[FreezeInterval, ...] = ()
    open_start: float | None = None
    for kind, raw in _FREEZE_EVENT.findall(stderr):
        closed, open_start = _apply_freeze_event(closed, open_start, kind, float(raw))
    # freezedetect no imprime freeze_end si el video termina congelado.
    tail = (FreezeInterval(start=open_start),) if open_start is not None else ()
    return [*closed, *tail]


def yuv420p_frame_size(width: int, height: int) -> int:
    return width * height * 3 // 2


def split_yuv420p_frames(buffer: bytes, width: int, height: int) -> np.ndarray:
    frame_size = yuv420p_frame_size(width, height)
    count = len(buffer) // frame_size
    return np.frombuffer(buffer, dtype=np.uint8, count=count * frame_size).reshape(count, frame_size)


def frame_stats(frames: np.ndarray, width: int, height: int) -> FrameStats | None:
    if frames.shape[0] == 0:
        return None
    luma = frames[:, : width * height]
    chroma = frames[:, width * height :].astype(np.int16)
    chroma_deviation = float(np.abs(chroma - CHROMA_NEUTRAL).mean())
    return FrameStats(
        samples=int(frames.shape[0]),
        luma_mean=float(luma.mean()),
        chroma_deviation=chroma_deviation,
        clipped_high_pct=float((luma >= CLIP_HIGH_LEVEL).mean() * 100),
        clipped_low_pct=float((luma <= CLIP_LOW_LEVEL).mean() * 100),
        monochrome=chroma_deviation <= MONOCHROME_MAX_CHROMA_DEVIATION,
    )


def parse_frame_stats(stdout: bytes, facts: SourceFacts) -> FrameStats | None:
    width, height = stats_dimensions(facts.width, facts.height)
    return frame_stats(split_yuv420p_frames(stdout, width, height), width, height)


def is_night_ir(stats: FrameStats | None) -> bool:
    return stats is not None and stats.monochrome and stats.luma_mean <= NIGHT_MAX_LUMA_MEAN


def is_interlaced(report: InterlaceReport | None) -> bool:
    return report is not None and report.interlaced


def is_low_res(facts: SourceFacts) -> bool:
    max_width, max_height = LOW_RES_MAX_SIZE
    return facts.width <= max_width and facts.height <= max_height


def suggest_preset(facts: SourceFacts, interlace: InterlaceReport | None, stats: FrameStats | None) -> PresetId:
    if is_night_ir(stats):
        return "night_ir"
    if is_interlaced(interlace):
        return "analog"
    if is_low_res(facts):
        return "low_res"
    return "day"


def _median_at_least(summary: MetricSummary | None, threshold: float) -> bool:
    return summary is not None and summary.median >= threshold


def _is_low_fps(facts: SourceFacts) -> bool:
    return facts.measured_fps is not None and facts.measured_fps < LOW_FPS_BELOW


def _clips_highlights(stats: FrameStats | None) -> bool:
    return stats is not None and stats.clipped_high_pct >= CLIPPING_WARN_PCT


def _has_frozen_or_duplicate_frames(facts: SourceFacts, freezes: list[FreezeInterval]) -> bool:
    return bool(freezes) or facts.probable_duplicates > 0


def diagnosis_warnings(
    facts: SourceFacts,
    interlace: InterlaceReport | None,
    blocking: MetricSummary | None,
    blur: MetricSummary | None,
    freezes: list[FreezeInterval],
    stats: FrameStats | None,
) -> list[str]:
    checks = (
        (is_interlaced(interlace), "cctv.warning.interlaced"),
        (_median_at_least(blocking, HEAVY_BLOCKING_MEDIAN), "cctv.warning.heavyBlocking"),
        (_median_at_least(blur, HEAVY_BLUR_MEDIAN), "cctv.warning.blurry"),
        (_has_frozen_or_duplicate_frames(facts, freezes), "cctv.warning.frozenFrames"),
        (_clips_highlights(stats), "cctv.warning.clippedHighlights"),
        (stats is not None and stats.monochrome, "cctv.warning.monochrome"),
        (_is_low_fps(facts), "cctv.warning.lowFrameRate"),
        (facts.is_vfr, "cctv.warning.variableFrameRate"),
        (facts.is_lite, "cctv.warning.liteAspect"),
    )
    return [key for triggered, key in checks if triggered]


def build_diagnosis(
    facts: SourceFacts,
    interlace: InterlaceReport | None,
    blocking: MetricSummary | None,
    blur: MetricSummary | None,
    freezes: list[FreezeInterval],
    stats: FrameStats | None,
) -> CctvDiagnosis:
    return CctvDiagnosis(
        interlace=interlace,
        blocking=blocking,
        blur=blur,
        freezes=freezes,
        frame_stats=stats,
        suggested_preset=suggest_preset(facts, interlace, stats),
        warnings=diagnosis_warnings(facts, interlace, blocking, blur, freezes, stats),
    )


def _stderr_tail(stderr: bytes, limit: int = 2000) -> str:
    return stderr.decode("utf-8", errors="replace")[-limit:].strip()


async def run_pass(
    name: str, command: list[str], run: ProcessRunner, timeout: float
) -> tuple[bytes, str]:
    stdout, stderr, returncode = await run(command, timeout)
    if returncode != 0:
        raise VideoAnalysisError(name, _stderr_tail(stderr) or f"exit code {returncode}")
    return stdout, stderr.decode("utf-8", errors="replace")


async def measure_interlace(
    ffmpeg: Path, video: Path, run: ProcessRunner, timeout: float
) -> InterlaceReport | None:
    _, stderr = await run_pass("idet", build_idet_command(ffmpeg, video), run, timeout)
    return parse_interlace(stderr)


async def measure_blocking(
    ffmpeg: Path, video: Path, run: ProcessRunner, timeout: float
) -> MetricSummary | None:
    _, stderr = await run_pass("blockdetect", build_blockdetect_command(ffmpeg, video), run, timeout)
    return summarize(parse_metadata_values(stderr, BLOCK_KEY))


async def measure_blur(ffmpeg: Path, video: Path, run: ProcessRunner, timeout: float) -> MetricSummary | None:
    _, stderr = await run_pass("blurdetect", build_blurdetect_command(ffmpeg, video), run, timeout)
    return summarize(parse_metadata_values(stderr, BLUR_KEY))


async def measure_freezes(ffmpeg: Path, video: Path, run: ProcessRunner, timeout: float) -> list[FreezeInterval]:
    _, stderr = await run_pass("freezedetect", build_freezedetect_command(ffmpeg, video), run, timeout)
    return parse_freezes(stderr)


async def measure_frame_stats(
    ffmpeg: Path, video: Path, facts: SourceFacts, run: ProcessRunner, timeout: float
) -> FrameStats | None:
    stdout, _ = await run_pass("frame_stats", build_frame_stats_command(ffmpeg, video, facts), run, timeout)
    return parse_frame_stats(stdout, facts)


async def analyze_video(
    ffmpeg: Path,
    video: Path,
    facts: SourceFacts,
    *,
    run: ProcessRunner = run_guarded_process,
    timeout: float = PASS_TIMEOUT_SECONDS,
) -> CctvDiagnosis:
    return build_diagnosis(
        facts,
        await measure_interlace(ffmpeg, video, run, timeout),
        await measure_blocking(ffmpeg, video, run, timeout),
        await measure_blur(ffmpeg, video, run, timeout),
        await measure_freezes(ffmpeg, video, run, timeout),
        await measure_frame_stats(ffmpeg, video, facts, run, timeout),
    )
