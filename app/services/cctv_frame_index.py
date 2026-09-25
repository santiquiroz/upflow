"""Indice de cuadros de la copia de trabajo CCTV (spec §4.2 paso 5), sin E/S.

Parsea la salida de `ffprobe -show_entries frame=...` y de ahi saca el conteo
real, los fps medidos (mediana de los deltas de PTS), la clasificacion CFR/VFR,
los huecos, los duplicados probables y la estructura del GOP. La tabla cuadro ->
PTS/`pict_type` la usan el recorte, los cuadros exportados y la fusion de ROI.
Los umbrales son provisionales hasta ver exports reales (P2-VAL).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

FRAME_INDEX_COLUMNS = ("n", "pts_time", "key_frame", "pict_type", "pkt_size")
VFR_DEVIATION = 0.10
GAP_FACTOR = 1.5


@dataclass(frozen=True)
class FrameEntry:
    n: int
    pts_time: float | None
    key_frame: bool
    pict_type: str
    pkt_size: int | None


@dataclass(frozen=True)
class FrameGap:
    after_frame: int
    start: float
    end: float

    def to_json(self) -> dict[str, Any]:
        return {"afterFrame": self.after_frame, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class GopStructure:
    keyframes: int
    min_length: int | None
    max_length: int | None
    median_length: float | None
    p_ratio: float
    b_ratio: float

    def to_json(self) -> dict[str, Any]:
        return {
            "keyframes": self.keyframes,
            "minLength": self.min_length,
            "maxLength": self.max_length,
            "medianLength": self.median_length,
            "pRatio": self.p_ratio,
            "bRatio": self.b_ratio,
        }


@dataclass(frozen=True)
class FrameIndexSummary:
    frame_count: int
    measured_fps: float | None
    median_delta: float | None
    is_vfr: bool
    gaps: tuple[FrameGap, ...]
    probable_duplicates: int
    gop: GopStructure

    def to_json(self) -> dict[str, Any]:
        return {
            "frameCount": self.frame_count,
            "measuredFps": self.measured_fps,
            "medianDelta": self.median_delta,
            "isVfr": self.is_vfr,
            "gaps": [gap.to_json() for gap in self.gaps],
            "probableDuplicates": self.probable_duplicates,
            "gop": self.gop.to_json(),
        }


def optional_float(raw: str | None) -> float | None:
    try:
        value = float(raw) if raw is not None else math.nan
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def optional_int(raw: str | None) -> int | None:
    return int(raw) if raw is not None and raw.isdigit() else None


def parse_frame_fields(line: str) -> dict[str, str]:
    pairs = (token.split("=", 1) for token in line.split(",") if "=" in token)
    return {key.strip(): value.strip() for key, value in pairs}


def frame_entry(n: int, fields: dict[str, str]) -> FrameEntry:
    return FrameEntry(
        n=n,
        pts_time=optional_float(fields.get("best_effort_timestamp_time")),
        key_frame=fields.get("key_frame") == "1",
        pict_type=fields.get("pict_type", "?"),
        pkt_size=optional_int(fields.get("pkt_size")),
    )


def parse_frame_index(text: str) -> tuple[FrameEntry, ...]:
    lines = [line for line in text.splitlines() if line.strip()]
    return tuple(frame_entry(n, parse_frame_fields(line)) for n, line in enumerate(lines))


def frame_csv_row(frame: FrameEntry) -> str:
    stamp = "" if frame.pts_time is None else f"{frame.pts_time:.6f}"
    size = "" if frame.pkt_size is None else str(frame.pkt_size)
    return f"{frame.n},{stamp},{int(frame.key_frame)},{frame.pict_type},{size}"


def frame_index_csv(frames: Sequence[FrameEntry]) -> str:
    return "".join(f"{row}\n" for row in (",".join(FRAME_INDEX_COLUMNS), *map(frame_csv_row, frames)))


def timed_frames(frames: Sequence[FrameEntry]) -> list[FrameEntry]:
    return [frame for frame in frames if frame.pts_time is not None]


def pts_deltas(frames: Sequence[FrameEntry]) -> list[float]:
    timed = timed_frames(frames)
    return [after.pts_time - before.pts_time for before, after in zip(timed, timed[1:])]


def median_delta(deltas: Sequence[float]) -> float | None:
    positive = [delta for delta in deltas if delta > 0]
    return statistics.median(positive) if positive else None


def measured_fps(median: float | None) -> float | None:
    return 1.0 / median if median else None


def is_gap(delta: float, median: float) -> bool:
    return delta > GAP_FACTOR * median


def is_vfr(deltas: Sequence[float], median: float | None) -> bool:
    # Los huecos se informan aparte: un CFR con cuadros perdidos sigue siendo CFR.
    regular = [delta for delta in deltas if median and not is_gap(delta, median)]
    if len(regular) < 2 or not median:
        return False
    return statistics.pstdev(regular) / median > VFR_DEVIATION


def frame_gaps(frames: Sequence[FrameEntry], median: float | None) -> tuple[FrameGap, ...]:
    if not median:
        return ()
    timed = timed_frames(frames)
    pairs = zip(timed, timed[1:])
    return tuple(
        FrameGap(before.n, before.pts_time, after.pts_time)
        for before, after in pairs
        if is_gap(after.pts_time - before.pts_time, median)
    )


def probable_duplicates(frames: Sequence[FrameEntry]) -> int:
    sizes = [frame.pkt_size for frame in frames if frame.pict_type == "P" and frame.pkt_size is not None]
    repeats = sizes.count(min(sizes)) if sizes else 0
    return repeats if repeats > 1 else 0


def type_ratio(frames: Sequence[FrameEntry], pict_type: str) -> float:
    return sum(1 for frame in frames if frame.pict_type == pict_type) / len(frames) if frames else 0.0


def gop_structure(frames: Sequence[FrameEntry]) -> GopStructure:
    keys = [frame.n for frame in frames if frame.key_frame]
    lengths = [after - before for before, after in zip(keys, keys[1:])]
    return GopStructure(
        keyframes=len(keys),
        min_length=min(lengths, default=None),
        max_length=max(lengths, default=None),
        median_length=statistics.median(lengths) if lengths else None,
        p_ratio=type_ratio(frames, "P"),
        b_ratio=type_ratio(frames, "B"),
    )


def summarize_frame_index(frames: Sequence[FrameEntry]) -> FrameIndexSummary:
    deltas = pts_deltas(frames)
    median = median_delta(deltas)
    return FrameIndexSummary(
        frame_count=len(frames),
        measured_fps=measured_fps(median),
        median_delta=median,
        is_vfr=is_vfr(deltas, median),
        gaps=frame_gaps(frames, median),
        probable_duplicates=probable_duplicates(frames),
        gop=gop_structure(frames),
    )
