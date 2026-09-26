"""Comparativo lado a lado de CCTV (spec §4.12).

`[0:v]` es el original con el mismo recorte (trim y crop) y sin filtros,
ampliado con `neighbor` al tamano del procesado para no "mejorarlo" sin querer;
`[1:v]` es el procesado. Se unen con `hstack` y arriba va una banda de titulo
bilingue. Los contadores del original (`drawtext`) usan la fuente OFL
bundleada: Consolas cambia entre builds de Windows y romperia la
reproducibilidad. Si `drawtext` falla, sale sin contadores y con aviso.

El modo "difference" (P4-DIFF) usa las mismas dos entradas pero las resta con
`blend=all_mode=difference` en RGB: negro donde el procesado no cambio nada.
Los contadores van despues de la resta para no contar como diferencia.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from PIL import Image

from app.services.cctv_chain import Lane, ResolvedStep
from app.services.cctv_clarify_runner import (
    OUTPUT_BITEXACT,
    X264_CRF,
    X264_PRESET,
    ClarifyThreads,
    ClarifyTools,
    ffmpeg_head,
    trim_step,
)
from app.services.ffmpeg_filters import (
    FILTER_SEPARATOR,
    FrameGeometry,
    build_filter,
    build_scale_to,
    escape_filter_path,
    escape_filter_value,
)
from app.services.ffmpeg_progress_runner import FfmpegProcessError, run_ffmpeg_with_progress
from app.services.label_band import (
    BAND_PADDING_RATIO,
    BLACK,
    FONT_PATH,
    TextBlock,
    band_font_cap,
    fitted_block,
    render_block,
    verified_font,
)

COMPARISON_NAME = "comparison.mp4"
TITLE_NAME = "comparison_title.png"
DIFFERENCE_NAME = "difference.mp4"
DIFFERENCE_TITLE_NAME = "difference_title.png"
Caption = tuple[str, str]
ORIGINAL_CAPTION: Caption = ("ORIGINAL", "ORIGINAL")
CLASSIC_CAPTION: Caption = ("PROCESSED (classic filters)", "PROCESADO (filtros clásicos)")
AI_CAPTION: Caption = ("AI VISUALIZATION", "VISUALIZACIÓN CON IA")
DIFFERENCE_CAPTIONS: dict[str, Caption] = {
    "classic": ("DIFFERENCE: ORIGINAL vs PROCESSED (black = unchanged)",
                "DIFERENCIA: ORIGINAL vs PROCESADO (negro = sin cambios)"),
    "ai": ("DIFFERENCE: ORIGINAL vs AI VISUALIZATION (black = unchanged)",
           "DIFERENCIA: ORIGINAL vs VISUALIZACIÓN CON IA (negro = sin cambios)"),
}  # fmt: skip
COUNTERS_MISSING ="cctv.warning.comparisonWithoutCounters"

COUNTER_TEXT = "#%{frame_num}  %{pts:hms}"
COUNTER_FONT_DIVISOR = 24
COUNTER_FONT_FLOOR = 12
COUNTER_OFFSET = 8
ORIGINAL_LABEL = "sbs_original"
PROCESSED_LABEL = "sbs_processed"
TITLE_LABEL = "sbs_title"
OUTPUT_LABEL = "sbs_out"
STACK_PIX_FMT = "yuv420p"
# En YUV la resta de croma deja U=V=0 (verde); en RGB lo que no cambio queda negro.
DIFFERENCE_PIX_FMT = "gbrp"
DIFFERENCE_BLEND = "blend=all_mode=difference"
RESTART_PTS = "setpts=PTS-STARTPTS"

ProgressCallback = Callable[[float], None]
ComparisonMode = Literal["side_by_side", "difference"]


@dataclass(frozen=True, slots=True)
class ComparisonPlan:
    original: Path
    processed: Path
    steps: tuple[ResolvedStep, ...]
    processed_geometry: FrameGeometry
    lane: Lane
    processed_frames: int
    with_counters: bool = True
    mode: ComparisonMode = "side_by_side"


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    path: Path
    command: tuple[str, ...]
    title: Path
    warnings: tuple[str, ...] = ()


# --- Geometria ---


def display_size(geometry: FrameGeometry) -> tuple[int, int]:
    sar = Fraction(geometry.sar)
    # Solo SAR entero (el modo lite): duplicar columnas con neighbor no calcula valores nuevos.
    if sar.denominator != 1 or sar <= 1:
        return geometry.width, geometry.height
    return geometry.width * sar.numerator, geometry.height


def same_cut_filters(steps: Sequence[ResolvedStep]) -> list[str]:
    return [build_filter(step) for step in steps if step.id in ("trim", "crop")]


def trim_start(steps: Sequence[ResolvedStep]) -> int:
    trim = trim_step(steps)
    return 0 if trim is None else int(trim.params["start_frame"])


# --- Contadores ---


def counter_font_size(height: int) -> int:
    return max(COUNTER_FONT_FLOOR, height // COUNTER_FONT_DIVISOR)


def counters_filter(font: Path, start_frame: int, height: int) -> str:
    options = [
        f"fontfile={escape_filter_path(font)}",
        f"text={escape_filter_value(COUNTER_TEXT)}",
        f"start_number={start_frame}",
        f"fontsize={counter_font_size(height)}",
        "fontcolor=white",
        "box=1",
        "boxcolor=black@0.6",
        "boxborderw=4",
        f"x={COUNTER_OFFSET}",
        f"y={COUNTER_OFFSET}",
    ]
    return "drawtext=" + ":".join(options)


# --- Grafo ---


def original_chain(
    steps: Sequence[ResolvedStep], size: tuple[int, int], counters: str | None, pix_fmt: str = STACK_PIX_FMT
) -> str:
    filters = [
        *same_cut_filters(steps),
        build_scale_to(*size),
        "setsar=1",
        *([counters] if counters else []),
        f"format={pix_fmt}",
        RESTART_PTS,
    ]
    return f"[0:v]{FILTER_SEPARATOR.join(filters)}[{ORIGINAL_LABEL}]"


def processed_chain(geometry: FrameGeometry, size: tuple[int, int], pix_fmt: str = STACK_PIX_FMT) -> str:
    scale = [] if (geometry.width, geometry.height) == size else [build_scale_to(*size)]
    filters = [*scale, "setsar=1", f"format={pix_fmt}", RESTART_PTS]
    return f"[1:v]{FILTER_SEPARATOR.join(filters)}[{PROCESSED_LABEL}]"


def title_chain(title_height: int) -> str:
    return (
        f"pad=w=iw:h=ih+{title_height}:x=0:y={title_height}:color=black[sbs_stacked];"
        f"[sbs_stacked][{TITLE_LABEL}]overlay=x=0:y=0:eval=init[{OUTPUT_LABEL}]"
    )


def stack_chain(title_height: int) -> str:
    return f"[{ORIGINAL_LABEL}][{PROCESSED_LABEL}]hstack=inputs=2,{title_chain(title_height)}"


def difference_chain(title_height: int, counters: str | None) -> str:
    filters = [DIFFERENCE_BLEND, f"format={STACK_PIX_FMT}", *([counters] if counters else [])]
    return f"[{ORIGINAL_LABEL}][{PROCESSED_LABEL}]{FILTER_SEPARATOR.join(filters)},{title_chain(title_height)}"


def title_source(title: Path) -> str:
    return f"movie={escape_filter_path(title)}[{TITLE_LABEL}]"


def side_by_side_graph(plan: ComparisonPlan, title: Path, title_height: int, counters: str | None) -> str:
    size = display_size(plan.processed_geometry)
    return ";".join(
        [
            original_chain(plan.steps, size, counters),
            processed_chain(plan.processed_geometry, size),
            title_source(title),
            stack_chain(title_height),
        ]
    )


def difference_graph(plan: ComparisonPlan, title: Path, title_height: int, counters: str | None) -> str:
    size = display_size(plan.processed_geometry)
    return ";".join(
        [
            original_chain(plan.steps, size, None, DIFFERENCE_PIX_FMT),
            processed_chain(plan.processed_geometry, size, DIFFERENCE_PIX_FMT),
            title_source(title),
            difference_chain(title_height, counters),
        ]
    )


def comparison_graph(plan: ComparisonPlan, title: Path, title_height: int, counters: str | None) -> str:
    graph = difference_graph if plan.mode == "difference" else side_by_side_graph
    return graph(plan, title, title_height, counters)


def x264_args(threads: ClarifyThreads) -> list[str]:
    x = str(threads.x264_threads)
    return [
        *("-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(X264_CRF), "-pix_fmt", STACK_PIX_FMT),
        *("-threads", x, "-x264-params", f"threads={x}:lookahead_threads=1"),
    ]


def build_comparison_command(
    ffmpeg: Path,
    plan: ComparisonPlan,
    title: Path,
    title_height: int,
    output: Path,
    threads: ClarifyThreads,
    counters: str | None,
) -> list[str]:
    return [
        *ffmpeg_head(ffmpeg),
        *("-i", str(plan.original), "-i", str(plan.processed)),
        *("-filter_complex", comparison_graph(plan, title, title_height, counters)),
        *("-map", f"[{OUTPUT_LABEL}]", "-fps_mode", "passthrough", "-an"),
        *x264_args(threads),
        *("-flags", "+bitexact"),
        *OUTPUT_BITEXACT,
        *("-movflags", "+faststart"),
        str(output),
    ]


# --- Banda de titulo ---


def processed_caption(lane: Lane) -> Caption:
    return AI_CAPTION if lane == "ai" else CLASSIC_CAPTION


def caption_block(font: Path, caption: Caption, half_width: int, height: int) -> TextBlock:
    return fitted_block(font, caption, half_width, band_font_cap(height))


def smallest_size(blocks: Sequence[TextBlock]) -> int:
    return min(block.size for block in blocks)


def render_title(font: Path, size: tuple[int, int], lane: Lane) -> Image.Image:
    width, height = size
    captions = (ORIGINAL_CAPTION, processed_caption(lane))
    fitted = [caption_block(font, caption, width, height) for caption in captions]
    common = smallest_size(fitted)
    halves = [render_block(TextBlock(block.lines, common), width, font, BLACK, BAND_PADDING_RATIO) for block in fitted]
    title = Image.new("RGB", (2 * width, max(half.height for half in halves)), BLACK[:3])
    for column, half in enumerate(halves):
        title.paste(half.convert("RGB"), (column * width, 0))
    return title


def difference_caption(lane: Lane) -> Caption:
    return DIFFERENCE_CAPTIONS["ai" if lane == "ai" else "classic"]


def render_difference_title(font: Path, size: tuple[int, int], lane: Lane) -> Image.Image:
    width, height = size
    block = caption_block(font, difference_caption(lane), width, height)
    return render_block(block, width, font, BLACK, BAND_PADDING_RATIO).convert("RGB")


def render_mode_title(font: Path, size: tuple[int, int], lane: Lane, mode: ComparisonMode) -> Image.Image:
    render = render_difference_title if mode == "difference" else render_title
    return render(font, size, lane)


def write_title(
    path: Path, font: Path, size: tuple[int, int], lane: Lane, mode: ComparisonMode = "side_by_side"
) -> int:
    image = render_mode_title(font, size, lane, mode)
    image.save(path, format="PNG")
    return image.height


# --- Orquestacion ---


def output_name(plan: ComparisonPlan) -> str:
    return DIFFERENCE_NAME if plan.mode == "difference" else COMPARISON_NAME


def title_name(plan: ComparisonPlan) -> str:
    return DIFFERENCE_TITLE_NAME if plan.mode == "difference" else TITLE_NAME


def planned_counters(plan: ComparisonPlan, font: Path) -> str | None:
    if not plan.with_counters:
        return None
    return counters_filter(font, trim_start(plan.steps), display_size(plan.processed_geometry)[1])


def counters_warning(counters: str | None) -> tuple[str, ...]:
    return () if counters else (COUNTERS_MISSING,)


async def encode_comparison(
    tools: ClarifyTools, command: list[str], plan: ComparisonPlan, on_progress: ProgressCallback
) -> None:
    await run_ffmpeg_with_progress(
        command, total_frames=plan.processed_frames, on_progress=on_progress, spawn=tools.spawn
    )


def monotonic_progress(on_progress: ProgressCallback) -> ProgressCallback:
    # El reintento sin contadores vuelve a empezar; la barra no debe retroceder.
    best = [0.0]

    def report(fraction: float) -> None:
        if fraction > best[0]:
            best[0] = fraction
            on_progress(fraction)

    return report


CommandBuilder = Callable[[str | None], list[str]]


async def encode_with_fallback(
    tools: ClarifyTools,
    plan: ComparisonPlan,
    build: CommandBuilder,
    counters: str | None,
    on_progress: ProgressCallback,
) -> tuple[list[str], str | None]:
    command = build(counters)
    try:
        await encode_comparison(tools, command, plan, on_progress)
        return command, counters
    except FfmpegProcessError:
        if counters is None:
            raise
    fallback = build(None)
    await encode_comparison(tools, fallback, plan, on_progress)
    return fallback, None


async def run_side_by_side(
    tools: ClarifyTools,
    plan: ComparisonPlan,
    output_dir: Path,
    assets_dir: Path,
    threads: ClarifyThreads,
    on_progress: ProgressCallback = lambda fraction: None,
    font_path: Path = FONT_PATH,
) -> ComparisonResult:
    font = verified_font(font_path)
    title = assets_dir / title_name(plan)
    title_height = write_title(title, font, display_size(plan.processed_geometry), plan.lane, plan.mode)
    output = output_dir / output_name(plan)

    def build(counters: str | None) -> list[str]:
        return build_comparison_command(tools.ffmpeg, plan, title, title_height, output, threads, counters)

    progress = monotonic_progress(on_progress)
    command, drawn = await encode_with_fallback(tools, plan, build, planned_counters(plan, font), progress)
    return ComparisonResult(output, tuple(command), title, counters_warning(drawn))
