"""Carril clasico de CCTV: runner determinista (spec §4.6).

Un ffmpeg arma `analysis.mkv` (FFV1 sin perdida) desde `work.mkv` y otro arma
`viewing.mp4` (x264) desde `analysis.mkv`. `-fflags +bitexact` va del lado de
la salida, justo antes de cada archivo: antes de `-i` es opcion del demuxer y
el MKV sale distinto en cada corrida. Filtros y FFV1 usan todos los nucleos con
`-slices` fijo; solo x264 lleva hilos fijos, porque es lo unico que cambia de
bits con la cantidad de hilos. Despues se verifica el conteo de cuadros, se
mide el clipping antes y despues y se arma `reproduce.cmd`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from app.models import utc_now
from app.services.cctv_chain import Limitation, ResolvedStep, in_catalog_order
from app.services.cctv_frame_index import FrameEntry
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.ffmpeg_filters import (
    Box,
    FrameGeometry,
    build_filter,
    build_osd_graph,
    build_scale_to,
    output_dims_after,
)
from app.services.ffmpeg_progress_runner import (
    PROGRESS_ARGS,
    Spawner,
    run_ffmpeg_with_progress,
    spawn_process,
)
from app.services.process_runner import run_guarded_process
from app.services.video_analysis import (
    STATS_FRAME_COUNT,
    FrameStats,
    ProcessRunner,
    SourceFacts,
    every_nth_frame,
    parse_frame_stats,
    stats_dimensions,
    stats_step,
)

ANALYSIS_NAME = "analysis.mkv"
VIEWING_NAME = "viewing.mp4"
VIDEO_LABEL = "v"
AUDIO_LABEL = "a"
GRAY_PIX_FMT = "gray"

OUTPUT_BITEXACT: tuple[str, ...] = ("-fflags", "+bitexact")
FFV1_ARGS: tuple[str, ...] = ("-c:v", "ffv1", "-level", "3", "-g", "1", "-slicecrc", "1")
X264_PRESET = "medium"
X264_CRF = 16
VIEWING_AUDIO_BITRATE = "128k"
STEP_TIMEOUT_SECONDS = 3600.0

FRAME_COUNT_MISMATCH = "cctv.error.frameCountMismatch"
CLIPPING_INCREASE_LIMIT_PCT = 0.5
CLIPPING_INCREASED_KEY = "cctv.limitation.clippingIncreased"

ANALYSIS_SHARE = 0.6
STAGE_CLARIFYING = "clarifying"
STAGE_VERIFYING = "verifying"

REPRODUCE_DIR = "reproduced"
SHA256SUMS_NAME = "SHA256SUMS.txt"
RESTORE_CODEPAGE = "if defined UPFLOW_CODEPAGE chcp %UPFLOW_CODEPAGE% >nul"
_CMD_FORBIDDEN = ('"', "\n", "\r", "\x00")

StageProgress = Callable[[str, float], None]


class FrameCountMismatch(RuntimeError):
    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"The processed video has {actual} frames but the source range has {expected}.")
        self.key = FRAME_COUNT_MISMATCH
        self.expected = expected
        self.actual = actual


class ClarifyStepError(RuntimeError):
    def __init__(self, step: str, detail: str) -> None:
        super().__init__(f"Clarify step '{step}' failed: {detail}")
        self.step = step


@dataclass(frozen=True, slots=True)
class ClarifyThreads:
    filter_threads: int
    ffv1_slices: int
    x264_threads: int


def clarify_threads(x264_threads: int, ffv1_slices: int, cpu_count: int | None = None) -> ClarifyThreads:
    return ClarifyThreads(max(1, cpu_count or os.cpu_count() or 1), ffv1_slices, x264_threads)


@dataclass(frozen=True, slots=True)
class ClarifyPlan:
    work: Path
    steps: tuple[ResolvedStep, ...]
    geometry: FrameGeometry
    source_pix_fmt: str
    frames: tuple[FrameEntry, ...]
    has_audio: bool
    app_version: str
    osd_boxes: tuple[Box, ...] = ()


@dataclass(frozen=True, slots=True)
class ClarifyTools:
    ffmpeg: Path
    ffprobe: Path
    run: ProcessRunner = run_guarded_process
    spawn: Spawner = spawn_process
    timeout: float = STEP_TIMEOUT_SECONDS


# --- Recorte y pix_fmt ---


def trim_step(steps: Sequence[ResolvedStep]) -> ResolvedStep | None:
    return next((step for step in steps if step.id == "trim"), None)


def trimmed_count(total: int, trim: ResolvedStep | None) -> int:
    if trim is None:
        return total
    start, end = int(trim.params["start_frame"]), int(trim.params["end_frame"])
    return max(0, min(total, end + 1) - start)


def output_pix_fmt(source_pix_fmt: str, steps: Sequence[ResolvedStep]) -> str:
    return GRAY_PIX_FMT if any(step.id == "gray" for step in steps) else source_pix_fmt


def frame_time(frames: Sequence[FrameEntry], n: int) -> float | None:
    return frames[n].pts_time if 0 <= n < len(frames) else None


def seconds_text(value: float) -> str:
    # Sin notacion exponencial: ffmpeg lee duraciones como "[-]S+[.m...]".
    return f"{value:.6f}".rstrip("0").rstrip(".")


def atrim_filter(trim: ResolvedStep, frames: Sequence[FrameEntry]) -> str:
    start = frame_time(frames, int(trim.params["start_frame"]))
    if start is None:
        raise ValueError("The first trimmed frame has no timestamp in the frame index")
    end = frame_time(frames, int(trim.params["end_frame"]) + 1)
    return f"atrim=start={seconds_text(start)}" + ("" if end is None else f":end={seconds_text(end)}")


# --- Pasada de analisis (FFV1) ---


def audio_segment(plan: ClarifyPlan) -> str | None:
    trim = trim_step(plan.steps)
    if trim is None or not plan.has_audio:
        return None
    return f"[0:a]{atrim_filter(trim, plan.frames)}[{AUDIO_LABEL}]"


def analysis_graph(plan: ClarifyPlan) -> str:
    video = build_osd_graph(
        plan.steps,
        plan.osd_boxes,
        plan.geometry,
        output_label=VIDEO_LABEL,
        pix_fmt=output_pix_fmt(plan.source_pix_fmt, plan.steps),
    )
    audio = audio_segment(plan)
    return video if audio is None else f"{video};{audio}"


def audio_map(plan: ClarifyPlan) -> list[str]:
    source = f"[{AUDIO_LABEL}]" if audio_segment(plan) is not None else "0:a?"
    return ["-map", source]


def processing_comment(app_version: str) -> str:
    return f"Processed with Upflow classic filters {app_version}; see report.json"


def ffmpeg_head(ffmpeg: Path) -> list[str]:
    return [str(ffmpeg), "-hide_banner", "-nostdin", "-y", *PROGRESS_ARGS]


def build_analysis_command(ffmpeg: Path, plan: ClarifyPlan, output: Path, threads: ClarifyThreads) -> list[str]:
    n = str(threads.filter_threads)
    return [
        *ffmpeg_head(ffmpeg),
        *("-threads", n, "-filter_threads", n, "-filter_complex_threads", n),
        *("-i", str(plan.work)),
        *("-filter_complex", analysis_graph(plan)),
        *("-map", f"[{VIDEO_LABEL}]", "-fps_mode", "passthrough"),
        *("-threads", n),
        *FFV1_ARGS,
        *("-slices", str(threads.ffv1_slices), "-flags:v", "+bitexact"),
        *audio_map(plan),
        *("-c:a", "flac", "-flags:a", "+bitexact"),
        *("-metadata", f"comment={processing_comment(plan.app_version)}"),
        *OUTPUT_BITEXACT,
        str(output),
    ]


# --- Copia de visualizacion (x264) ---


def square_pixel_filters(geometry: FrameGeometry) -> list[str]:
    sar = Fraction(geometry.sar)
    # Solo SAR entero (el modo lite): duplicar columnas con neighbor no calcula valores nuevos.
    if sar.denominator != 1 or sar <= 1:
        return []
    return [build_scale_to(geometry.width * sar.numerator, geometry.height), "setsar=1"]


def viewing_vf(geometry: FrameGeometry) -> list[str]:
    filters = square_pixel_filters(geometry)
    return ["-vf", ",".join(filters)] if filters else []


def build_viewing_command(
    ffmpeg: Path, analysis: Path, output: Path, geometry: FrameGeometry, threads: ClarifyThreads
) -> list[str]:
    x = str(threads.x264_threads)
    return [
        *ffmpeg_head(ffmpeg),
        *("-i", str(analysis)),
        *viewing_vf(geometry),
        *("-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(X264_CRF), "-pix_fmt", "yuv420p"),
        *("-threads", x, "-x264-params", f"threads={x}:lookahead_threads=1"),
        *("-c:a", "aac", "-b:a", VIEWING_AUDIO_BITRATE, "-flags", "+bitexact"),
        *OUTPUT_BITEXACT,
        *("-movflags", "+faststart"),
        str(output),
    ]


# --- Conteo de cuadros ---


def build_frame_count_command(ffprobe: Path, video: Path) -> list[str]:
    return [
        str(ffprobe),
        *("-v", "error", "-count_frames", "-select_streams", "v:0"),
        *("-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1"),
        str(video),
    ]


def parse_frame_count(stdout: bytes) -> int:
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text.isdigit():
        raise ClarifyStepError("count_frames", f"ffprobe did not report a frame count: {text[:200]!r}")
    return int(text)


def check_frame_counts(expected: int, actual: int) -> None:
    if expected != actual:
        raise FrameCountMismatch(expected, actual)


# --- Clipping antes y despues ---


def build_clipping_command(
    ffmpeg: Path, video: Path, facts: SourceFacts, prefix: Sequence[str] = ()
) -> list[str]:
    width, height = stats_dimensions(facts.width, facts.height)
    filters = [
        *prefix,
        every_nth_frame(stats_step(facts.frame_count)),
        f"scale={width}:{height}:flags=neighbor:out_range=full",
        "format=yuv420p",
    ]
    return [
        *(str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-v", "error"),
        *("-i", str(video), "-map", "0:v:0", "-vf", ",".join(filters)),
        *("-fps_mode", "passthrough", "-frames:v", str(STATS_FRAME_COUNT)),
        *("-f", "rawvideo", "-"),
    ]


def clipped_pct(stats: FrameStats | None) -> float | None:
    return None if stats is None else stats.clipped_high_pct + stats.clipped_low_pct


@dataclass(frozen=True, slots=True)
class ClippingReport:
    before_pct: float | None
    after_pct: float | None

    @property
    def increase_pct(self) -> float | None:
        if self.before_pct is None or self.after_pct is None:
            return None
        return self.after_pct - self.before_pct

    @property
    def increased(self) -> bool:
        increase = self.increase_pct
        return increase is not None and increase > CLIPPING_INCREASE_LIMIT_PCT

    def to_json(self) -> dict[str, float | bool | None]:
        return {
            "beforePct": self.before_pct,
            "afterPct": self.after_pct,
            "increasePct": self.increase_pct,
            "increased": self.increased,
        }


def clipping_limitation(report: ClippingReport) -> Limitation | None:
    if not report.increased:
        return None
    return Limitation(CLIPPING_INCREASED_KEY, f"Processing increased clipped pixels by {report.increase_pct:.2f}%")


# --- reproduce.cmd ---


@dataclass(frozen=True, slots=True)
class ReproduceStep:
    label: str
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReproduceCheck:
    produced: str
    listed_as: str


def windows_path(relative: str) -> str:
    return relative.replace("/", "\\")


def cmd_escape(text: str) -> str:
    return text.replace("%", "%%")


def cmd_safe(text: str) -> str:
    if any(char in text for char in _CMD_FORBIDDEN):
        raise ValueError(f"Argument {text!r} can't be written safely to a .cmd file")
    return cmd_escape(text)


def cmd_quote(arg: str) -> str:
    return f'"{cmd_safe(arg)}"'


def cmd_set(name: str, value: str) -> str:
    return f'set "{name}={cmd_safe(value)}"'


def relative_argv(argv: Sequence[str], path_map: Mapping[str, str]) -> list[str]:
    return [windows_path(path_map[arg]) if arg in path_map else arg for arg in argv[1:]]


def command_line(step: ReproduceStep, path_map: Mapping[str, str]) -> list[str]:
    args = " ".join(cmd_quote(arg) for arg in relative_argv(step.argv, path_map))
    return [f"echo == {cmd_escape(step.label)}", f'"%FFMPEG%" {args}', "if errorlevel 1 goto :failed"]


def build_header(caps: FfmpegCapabilities) -> list[str]:
    return [
        "@echo off",
        # UTF-8 para los nombres no ASCII del original; la pagina de codigos se restaura al salir.
        "for /f \"tokens=2 delims=:.\" %%C in ('chcp') do set \"UPFLOW_CODEPAGE=%%C\"",
        "chcp 65001 >nul",
        "setlocal EnableExtensions DisableDelayedExpansion",
        "rem Reproduces the Upflow classic-filter outputs of this package and compares them with SHA256SUMS.txt.",
        f"rem Expected ffmpeg build: {cmd_escape(caps.version)}",
        f"rem Expected ffmpeg.exe sha256: {caps.binary_sha256}",
        f"rem CPU extensions used: {cmd_escape(' '.join(caps.cpu_extensions) or 'none detected')}",
        "rem Identical results need the same ffmpeg build and the same CPU features.",
        'pushd "%~dp0"',
        'if not defined FFMPEG for %%F in (ffmpeg.exe) do set "FFMPEG=%%~$PATH:F"',
        'if not defined FFMPEG (echo Set FFMPEG to the full path of ffmpeg.exe & goto :failed)',
        f'set "EXPECTED_FFMPEG_SHA256={caps.binary_sha256}"',
        'call :sha256 "%FFMPEG%"',
        'if /i not "%SHA%"=="%EXPECTED_FFMPEG_SHA256%" echo WARNING: this ffmpeg.exe is not the build Upflow used.',
        f'if not exist "{REPRODUCE_DIR}" mkdir "{REPRODUCE_DIR}"',
        'set "FAILED="',
    ]


def check_lines(checks: Sequence[ReproduceCheck]) -> list[str]:
    # Por variables y no como argumentos: CALL vuelve a expandir los % y se comeria los del nombre.
    return [
        line
        for c in checks
        for line in (cmd_set("PRODUCED", windows_path(c.produced)), cmd_set("LISTED", c.listed_as), "call :check")
    ]


def build_footer() -> list[str]:
    return [
        "popd",
        RESTORE_CODEPAGE,
        "if defined FAILED (echo Some files are different. & exit /b 1)",
        "echo All reproduced files match SHA256SUMS.txt.",
        "exit /b 0",
        "",
        ":failed",
        "popd",
        RESTORE_CODEPAGE,
        "exit /b 1",
        "",
        ":sha256",
        'set "SHA="',
        'for /f "delims=" %%H in (\'certutil -hashfile "%~1" SHA256 ^| findstr /v ":"\') do '
        'if not defined SHA set "SHA=%%H"',
        'if defined SHA set "SHA=%SHA: =%"',
        "exit /b 0",
        "",
        ":check",
        'call :sha256 "%PRODUCED%"',
        f'findstr /l /i /c:"%SHA% *%LISTED%" "{SHA256SUMS_NAME}" >nul',
        # Comillas en el echo: el nombre sale del archivo del usuario y puede traer & o ).
        'if errorlevel 1 (echo DIFFERENT "%LISTED%" & set "FAILED=1") else (echo MATCH "%LISTED%")',
        "exit /b 0",
    ]


def build_reproduce_script(
    steps: Sequence[ReproduceStep],
    checks: Sequence[ReproduceCheck],
    path_map: Mapping[str, str],
    caps: FfmpegCapabilities,
) -> str:
    body = [line for step in steps for line in command_line(step, path_map)]
    lines = [*build_header(caps), *body, *check_lines(checks), *build_footer()]
    return "\r\n".join(lines) + "\r\n"


# --- Orquestacion ---


@dataclass(frozen=True, slots=True)
class PassTiming:
    started_at: datetime
    ended_at: datetime


@dataclass(frozen=True, slots=True)
class ClarifyResult:
    analysis: Path
    viewing: Path
    analysis_command: tuple[str, ...]
    viewing_command: tuple[str, ...]
    expected_frames: int
    output_frames: int
    output_geometry: FrameGeometry
    clipping: ClippingReport
    limitations: tuple[Limitation, ...] = ()
    analysis_timing: PassTiming | None = None
    viewing_timing: PassTiming | None = None

    def reproduce_steps(self) -> tuple[ReproduceStep, ...]:
        return (
            ReproduceStep("analysis copy (FFV1)", self.analysis_command),
            ReproduceStep("viewing copy (H.264)", self.viewing_command),
        )


def scaled_progress(on_progress: StageProgress, offset: float, share: float) -> Callable[[float], None]:
    return lambda fraction: on_progress(STAGE_CLARIFYING, offset + share * fraction)


async def run_step(name: str, command: list[str], tools: ClarifyTools) -> bytes:
    stdout, stderr, returncode = await tools.run(command, tools.timeout)
    if returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-2000:].strip() or f"exit code {returncode}"
        raise ClarifyStepError(name, detail)
    return stdout


async def count_frames(tools: ClarifyTools, video: Path) -> int:
    return parse_frame_count(await run_step("count_frames", build_frame_count_command(tools.ffprobe, video), tools))


async def measure_clipping(
    tools: ClarifyTools, video: Path, facts: SourceFacts, prefix: Sequence[str] = ()
) -> float | None:
    stdout = await run_step("clipping", build_clipping_command(tools.ffmpeg, video, facts, prefix), tools)
    return clipped_pct(parse_frame_stats(stdout, facts))


def trim_prefix(steps: Sequence[ResolvedStep]) -> list[str]:
    trim = trim_step(steps)
    return [] if trim is None else [build_filter(trim)]


async def clipping_report(
    tools: ClarifyTools, plan: ClarifyPlan, analysis: Path, geometry: FrameGeometry, counts: tuple[int, int]
) -> ClippingReport:
    before_facts = SourceFacts(width=plan.geometry.width, height=plan.geometry.height, frame_count=counts[0])
    after_facts = SourceFacts(width=geometry.width, height=geometry.height, frame_count=counts[1])
    return ClippingReport(
        before_pct=await measure_clipping(tools, plan.work, before_facts, trim_prefix(plan.steps)),
        after_pct=await measure_clipping(tools, analysis, after_facts),
    )


async def verified_counts(tools: ClarifyTools, plan: ClarifyPlan, analysis: Path) -> tuple[int, int]:
    expected = trimmed_count(await count_frames(tools, plan.work), trim_step(plan.steps))
    actual = await count_frames(tools, analysis)
    check_frame_counts(expected, actual)
    return expected, actual


async def run_analysis_pass(
    tools: ClarifyTools, plan: ClarifyPlan, analysis: Path, threads: ClarifyThreads, on_progress: StageProgress
) -> list[str]:
    command = build_analysis_command(tools.ffmpeg, plan, analysis, threads)
    total = trimmed_count(len(plan.frames), trim_step(plan.steps))
    report = scaled_progress(on_progress, 0.0, ANALYSIS_SHARE)
    await run_ffmpeg_with_progress(command, total_frames=total, on_progress=report, spawn=tools.spawn)
    return command


async def run_viewing_pass(
    tools: ClarifyTools,
    analysis: Path,
    viewing: Path,
    geometry: FrameGeometry,
    frames: int,
    threads: ClarifyThreads,
    on_progress: StageProgress,
) -> list[str]:
    command = build_viewing_command(tools.ffmpeg, analysis, viewing, geometry, threads)
    report = scaled_progress(on_progress, ANALYSIS_SHARE, 1.0 - ANALYSIS_SHARE)
    await run_ffmpeg_with_progress(command, total_frames=frames, on_progress=report, spawn=tools.spawn)
    return command


def plan_output_geometry(plan: ClarifyPlan) -> FrameGeometry:
    source = plan.geometry
    return output_dims_after(in_catalog_order(plan.steps), source.width, source.height, source.sar)


def limitations_of(clipping: ClippingReport) -> tuple[Limitation, ...]:
    limitation = clipping_limitation(clipping)
    return () if limitation is None else (limitation,)


async def run_clarify(
    tools: ClarifyTools,
    plan: ClarifyPlan,
    output_dir: Path,
    threads: ClarifyThreads,
    on_progress: StageProgress = lambda stage, fraction: None,
    now: Callable[[], datetime] = utc_now,
) -> ClarifyResult:
    geometry = plan_output_geometry(plan)
    analysis, viewing = output_dir / ANALYSIS_NAME, output_dir / VIEWING_NAME
    analysis_started = now()
    analysis_command = await run_analysis_pass(tools, plan, analysis, threads, on_progress)
    analysis_timing = PassTiming(analysis_started, now())
    counts = await verified_counts(tools, plan, analysis)
    viewing_started = now()
    viewing_command = await run_viewing_pass(tools, analysis, viewing, geometry, counts[1], threads, on_progress)
    viewing_timing = PassTiming(viewing_started, now())
    on_progress(STAGE_VERIFYING, 0.0)
    clipping = await clipping_report(tools, plan, analysis, geometry, counts)
    on_progress(STAGE_VERIFYING, 1.0)
    return ClarifyResult(
        analysis=analysis,
        viewing=viewing,
        analysis_command=tuple(analysis_command),
        viewing_command=tuple(viewing_command),
        expected_frames=counts[0],
        output_frames=counts[1],
        output_geometry=geometry,
        clipping=clipping,
        limitations=limitations_of(clipping),
        analysis_timing=analysis_timing,
        viewing_timing=viewing_timing,
    )
