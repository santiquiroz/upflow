"""Vistas previas de una sesion CCTV (spec §2.6 pasos 6-7, §5.5 `preview` y `osd-check`).

El cuadro N se decodifica con `-ss` aproximado (con margen) y `-copyts`, y se elige
con `select` por su PTS del indice: es para dibujar, no un cuadro exportado. Con
pasos, solo clasicos y validados igual que un job, se filtra una ventana de +-K
cuadros alrededor de N y se toma el cuadro N por su posicion en la ventana (la
cadena conserva el conteo de cuadros).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.cctv_chain import INVALID_STEP, CctvChainError, ResolvedStep, steps_from_request
from app.services.cctv_frame_index import FrameEntry, median_delta, parse_frame_index_csv, pts_deltas
from app.services.cctv_ingest import FRAME_INDEX_NAME, MediaTools, probe_media
from app.services.cctv_job_runner import frame_geometry
from app.services.cctv_job_validation import check_filters_available, check_geometry
from app.services.cctv_session import SESSION_NOT_FOUND, SESSION_NOT_ANALYZED, session_dir, working_copy
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.ffmpeg_filters import FrameGeometry, compose_vf
from app.services.osd_check import Box, OsdBoxCheck, run_osd_check

MAX_PREVIEW_WINDOW = 12
SEEK_MARGIN_SECONDS = 1.0
DEFAULT_HALF_FRAME = 0.001
PREVIEW_TIMEOUT_SECONDS = 120.0
STDERR_TAIL_CHARS = 500
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PREVIEW_LANE = "classic"
# Un cuadro suelto no tiene recorte que mostrar, y la proteccion del OSD necesita el grafo del job.
PREVIEW_SKIPPED_STEPS = frozenset({"trim", "osd_protect"})

FRAME_OUT_OF_RANGE = "cctv.error.frameOutOfRange"
FRAME_WITHOUT_TIMESTAMP = "cctv.error.frameWithoutTimestamp"
INVALID_WINDOW = "cctv.error.invalidPreviewWindow"
PREVIEW_FAILED = "cctv.error.previewFailed"


@dataclass(frozen=True, slots=True)
class PreviewSource:
    work: Path
    frames: tuple[FrameEntry, ...]
    geometry: FrameGeometry

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True, slots=True)
class FrameWindow:
    first: int
    target: int
    last: int


# --- Sesion ---


def session_frames(directory: Path) -> tuple[FrameEntry, ...]:
    index = directory / FRAME_INDEX_NAME
    if not index.is_file():
        raise CctvChainError(SESSION_NOT_ANALYZED, "The CCTV session has no frame index; analyze the file first.")
    frames = parse_frame_index_csv(index.read_text(encoding="utf-8"))
    if not frames:
        raise CctvChainError(SESSION_NOT_ANALYZED, "The CCTV session has no decoded frames.")
    return frames


def existing_session(work_root: Path, token: str) -> Path:
    directory = session_dir(work_root, token)
    if not directory.is_dir():
        raise CctvChainError(SESSION_NOT_FOUND, "The CCTV session was not found or has expired; upload the file again.")
    return directory


async def load_preview_source(work_root: Path, token: str, media: MediaTools) -> PreviewSource:
    directory = existing_session(work_root, token)
    work = working_copy(directory)
    frames = await asyncio.to_thread(session_frames, directory)
    probe = await probe_media(media, work)
    return PreviewSource(work, frames, frame_geometry(probe))


# --- Tiempos y ventana ---


def check_frame(frame: int, frame_count: int) -> int:
    if not 0 <= frame < frame_count:
        raise CctvChainError(FRAME_OUT_OF_RANGE, f"Frame {frame} is outside the video (0-{frame_count - 1}).")
    return frame


def frame_time(frames: Sequence[FrameEntry], frame: int) -> float:
    stamp = frames[frame].pts_time
    if stamp is None:
        raise CctvChainError(FRAME_WITHOUT_TIMESTAMP, f"Frame {frame} has no timestamp in the index.")
    return stamp


def first_time(frames: Sequence[FrameEntry]) -> float:
    return next((frame.pts_time for frame in frames if frame.pts_time is not None), 0.0)


def half_frame(frames: Sequence[FrameEntry]) -> float:
    median = median_delta(pts_deltas(frames))
    return DEFAULT_HALF_FRAME if median is None else median / 2


def seek_seconds(frames: Sequence[FrameEntry], frame: int) -> float:
    # -ss se mide desde el inicio del archivo; el margen cubre el keyframe previo y un start_time distinto de 0.
    return max(0.0, frame_time(frames, frame) - first_time(frames) - SEEK_MARGIN_SECONDS)


def frame_window(frame: int, radius: int, frame_count: int) -> FrameWindow:
    if not 0 <= radius <= MAX_PREVIEW_WINDOW:
        raise CctvChainError(INVALID_WINDOW, f"The preview window must be between 0 and {MAX_PREVIEW_WINDOW} frames.")
    target = check_frame(frame, frame_count)
    return FrameWindow(max(0, target - radius), target, min(frame_count - 1, target + radius))


def seconds(value: float) -> str:
    return f"{value:.6f}"


# --- Filtros y comandos ---


def frame_filter(frames: Sequence[FrameEntry], frame: int) -> str:
    return f"select=gte(t\\,{seconds(frame_time(frames, frame) - half_frame(frames))})"


def window_filter(frames: Sequence[FrameEntry], window: FrameWindow, chain: str) -> str:
    margin = half_frame(frames)
    start = frame_time(frames, window.first) - margin
    end = frame_time(frames, window.last) + margin
    span = f"select=between(t\\,{seconds(start)}\\,{seconds(end)})"
    parts = [span, chain, f"select=eq(n\\,{window.target - window.first})"]
    return ",".join(part for part in parts if part)


def build_preview_command(ffmpeg: Path, work: Path, seek: float, vf: str) -> list[str]:
    return [
        str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-v", "error",
        "-ss", seconds(seek), "-copyts", "-i", str(work),
        "-map", "0:v:0", "-vf", vf, "-frames:v", "1", "-fps_mode", "passthrough",
        "-f", "image2pipe", "-c:v", "png", "pipe:1",
    ]  # fmt: skip


# --- Pasos de la vista previa ---


def parse_steps_param(raw: str) -> list[Any]:
    try:
        steps = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CctvChainError(INVALID_STEP, "'steps' must be a JSON array of steps.") from exc
    if not isinstance(steps, list):
        raise CctvChainError(INVALID_STEP, "'steps' must be a JSON array of steps.")
    return steps


def preview_steps(
    raw_steps: Sequence[Any], caps: FfmpegCapabilities, geometry: FrameGeometry
) -> tuple[ResolvedStep, ...]:
    steps = steps_from_request(raw_steps, PREVIEW_LANE)
    check_filters_available(steps, caps)
    check_geometry(steps, geometry)
    return tuple(step for step in steps if step.id not in PREVIEW_SKIPPED_STEPS)


# --- Ejecucion ---


def stderr_tail(stderr: bytes) -> str:
    return stderr.decode("utf-8", errors="replace")[-STDERR_TAIL_CHARS:].strip()


async def render_png(media: MediaTools, command: list[str]) -> bytes:
    stdout, stderr, returncode = await media.run(command, PREVIEW_TIMEOUT_SECONDS)
    if returncode != 0 or not stdout.startswith(PNG_SIGNATURE):
        raise CctvChainError(PREVIEW_FAILED, f"The preview could not be decoded: {stderr_tail(stderr) or 'no frame'}")
    return stdout


async def render_frame(media: MediaTools, source: PreviewSource, frame: int) -> bytes:
    target = check_frame(frame, source.frame_count)
    vf = frame_filter(source.frames, target)
    seek = seek_seconds(source.frames, target)
    return await render_png(media, build_preview_command(media.ffmpeg, source.work, seek, vf))


async def render_processed_frame(
    media: MediaTools, source: PreviewSource, window: FrameWindow, steps: Sequence[ResolvedStep]
) -> bytes:
    vf = window_filter(source.frames, window, compose_vf(steps))
    seek = seek_seconds(source.frames, window.first)
    return await render_png(media, build_preview_command(media.ffmpeg, source.work, seek, vf))


async def check_osd_on_session(
    media: MediaTools, source: PreviewSource, boxes: Sequence[Box], frame: int | None
) -> tuple[OsdBoxCheck, ...]:
    start = 0.0 if frame is None else frame_time(source.frames, check_frame(frame, source.frame_count))
    offset = max(0.0, start - first_time(source.frames))
    geometry = source.geometry
    return await run_osd_check(media.ffmpeg, source.work, geometry.width, geometry.height, boxes, offset, media.run)
