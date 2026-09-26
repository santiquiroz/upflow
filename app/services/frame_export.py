"""Cuadros exactos de CCTV (spec §4.10).

El cuadro N es el orden de salida del decodificador sobre `work.mkv`, el mismo
que usan `trim`, `select` y el indice de cuadros. Cada cuadro se saca con
`select=eq(n\\,N)` decodificando desde el inicio (exacto) y se hashea con
`-f framehash -hash sha256` sobre el cuadro decodificado, no sobre el PNG.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.services.cctv_clarify_runner import ClarifyStepError, ClarifyTools, run_step
from app.services.label_band import LabelAssets, label_band_graph, tag_png_with_xmp

MAX_STILL_FRAMES = 20
TOO_MANY_STILL_FRAMES = "cctv.error.tooManyStillFrames"
STILL_FRAME_OUT_OF_RANGE = "cctv.error.stillFrameOutOfRange"
APPROXIMATE_NOTE = "approximate (re-timed frames)"
HASH_ALGORITHM = "sha256"
FRAMEHASH_FIELDS = 6

Role = Literal["original", "processed"]


class StillFrameError(ValueError):
    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key


# --- Validacion ---


def check_still_count(frames: Sequence[int]) -> None:
    if len(frames) > MAX_STILL_FRAMES:
        raise StillFrameError(TOO_MANY_STILL_FRAMES, f"At most {MAX_STILL_FRAMES} still frames per job")


def check_still_range(frame: int, first: int, last: int) -> None:
    if not first <= frame <= last:
        raise StillFrameError(STILL_FRAME_OUT_OF_RANGE, f"Frame {frame} is outside the range {first}-{last}")


def checked_still_frames(requested: Sequence[int], first: int, last: int) -> tuple[int, ...]:
    frames = tuple(sorted(set(requested)))
    check_still_count(frames)
    for frame in frames:
        check_still_range(frame, first, last)
    return frames


# --- Comandos ---


def select_frame_filter(frame: int) -> str:
    if frame < 0:
        raise ValueError(f"Frame numbers start at 0, got {frame}")
    return f"select=eq(n\\,{frame})"


def still_vf(frame: int, label: LabelAssets | None = None) -> str:
    select = select_frame_filter(frame)
    return select if label is None else label_band_graph(label, prefix=(select,))


def single_frame_args(vf: str) -> list[str]:
    return ["-map", "0:v:0", "-vf", vf, "-frames:v", "1", "-fps_mode", "passthrough"]


def ffmpeg_quiet(ffmpeg: Path) -> list[str]:
    return [str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-v", "error", "-y"]


def build_still_command(
    ffmpeg: Path, source: Path, frame: int, output: Path, label: LabelAssets | None = None
) -> list[str]:
    return [
        *ffmpeg_quiet(ffmpeg),
        *("-i", str(source)),
        *single_frame_args(still_vf(frame, label)),
        *("-flags", "+bitexact", "-fflags", "+bitexact"),
        str(output),
    ]


def build_framehash_command(ffmpeg: Path, source: Path, frame: int | None = None) -> list[str]:
    selection = ["-map", "0:v:0", "-fps_mode", "passthrough"]
    if frame is not None:
        selection = single_frame_args(select_frame_filter(frame))
    return [*ffmpeg_quiet(ffmpeg), "-i", str(source), *selection, "-f", "framehash", "-hash", HASH_ALGORITHM, "-"]


# --- framehash y tiempos ---


@dataclass(frozen=True, slots=True)
class FrameHash:
    pts: int
    sha256: str


def parse_framehash_line(line: str) -> FrameHash:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != FRAMEHASH_FIELDS:
        raise ValueError(f"Unexpected framehash line: {line[:200]!r}")
    return FrameHash(pts=int(fields[2]), sha256=fields[5].lower())


def parse_framehashes(stdout: bytes) -> tuple[FrameHash, ...]:
    lines = stdout.decode("utf-8", errors="replace").splitlines()
    return tuple(parse_framehash_line(line) for line in lines if line.strip() and not line.startswith("#"))


def single_hash(hashes: Sequence[FrameHash], frame: int) -> str:
    if len(hashes) != 1:
        raise ClarifyStepError("framehash", f"expected one hash for frame {frame}, got {len(hashes)}")
    return hashes[0].sha256


def timecode(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    millis = round(abs(seconds) * 1000)
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f"{sign}{hours:02d}:{minutes:02d}:{rest // 1000:02d}.{rest % 1000:03d}"


def nearest_frame(times: Sequence[float], target: float) -> int:
    if not times:
        raise ValueError("No processed frames to pair with")
    right = bisect.bisect_left(times, target)
    candidates = [index for index in (right - 1, right) if 0 <= index < len(times)]
    return min(candidates, key=lambda index: (abs(times[index] - target), index))


def relative_times(times: Sequence[float]) -> tuple[float, ...]:
    return tuple(time - times[0] for time in times) if times else ()


# --- Pares de cuadros ---


@dataclass(frozen=True, slots=True)
class StillFrame:
    role: Role
    frame: int
    path: Path
    pts_time: float
    framehash: str

    def to_json(self, base: Path) -> dict[str, Any]:
        return {
            "role": self.role,
            "frame": self.frame,
            "file": self.path.relative_to(base).as_posix(),
            "ptsTime": self.pts_time,
            "timecode": timecode(self.pts_time),
            "framehashSha256": self.framehash,
        }


@dataclass(frozen=True, slots=True)
class StillPair:
    frame: int
    original: StillFrame
    processed: StillFrame
    approximate: bool

    def to_json(self, base: Path) -> dict[str, Any]:
        note = {"note": APPROXIMATE_NOTE} if self.approximate else {}
        return {
            "frame": self.frame,
            "original": self.original.to_json(base),
            "processed": self.processed.to_json(base),
            "approximate": self.approximate,
            **note,
        }


@dataclass(frozen=True, slots=True)
class StillSource:
    path: Path
    times: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class StillRequest:
    original: StillSource
    processed: StillSource
    frames: tuple[int, ...]
    output_dir: Path
    trim_start: int = 0
    retimed: bool = False
    label: LabelAssets | None = None
    xmp_packet: str | None = None
    # El video del carril IA ya trae la banda quemada: se exporta tal cual, sin una segunda banda.
    label_burned_in: bool = False

    def __post_init__(self) -> None:
        has_band = self.label is not None or self.label_burned_in
        if has_band != (self.xmp_packet is not None):
            raise ValueError("AI-lane stills need both the visible band and the XMP packet")


def still_name(role: Role, frame: int) -> str:
    return f"{role}_f{frame}.png"


def processed_frame(request: StillRequest, frame: int) -> int:
    if not request.retimed:
        return frame - request.trim_start
    offset = request.original.times[frame] - request.original.times[request.trim_start]
    return nearest_frame(relative_times(request.processed.times), offset)


async def frame_sha256(tools: ClarifyTools, source: Path, frame: int) -> str:
    stdout = await run_step("framehash", build_framehash_command(tools.ffmpeg, source, frame), tools)
    return single_hash(parse_framehashes(stdout), frame)


async def export_still(
    tools: ClarifyTools, source: StillSource, frame: int, output: Path, label: LabelAssets | None = None
) -> tuple[Path, str]:
    await run_step("still", build_still_command(tools.ffmpeg, source.path, frame, output, label), tools)
    return output, await frame_sha256(tools, source.path, frame)


async def export_original(tools: ClarifyTools, request: StillRequest, frame: int) -> StillFrame:
    output = request.output_dir / still_name("original", frame)
    path, digest = await export_still(tools, request.original, frame, output)
    return StillFrame("original", frame, path, request.original.times[frame], digest)


def tag_processed(request: StillRequest, path: Path) -> None:
    if request.xmp_packet is not None:
        tag_png_with_xmp(path, request.xmp_packet)


async def export_processed(tools: ClarifyTools, request: StillRequest, frame: int) -> StillFrame:
    index = processed_frame(request, frame)
    output = request.output_dir / still_name("processed", frame)
    path, digest = await export_still(tools, request.processed, index, output, request.label)
    tag_processed(request, path)
    time = request.processed.times[index] if request.retimed else request.original.times[frame]
    return StillFrame("processed", index, path, time, digest)


async def export_still_pairs(tools: ClarifyTools, request: StillRequest) -> tuple[StillPair, ...]:
    pairs = []
    for frame in request.frames:
        original = await export_original(tools, request, frame)
        processed = await export_processed(tools, request, frame)
        pairs.append(StillPair(frame, original, processed, request.retimed))
    return tuple(pairs)
