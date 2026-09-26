"""Proceso ffmpeg largo con progreso en streaming (spec §4.6).

`run_guarded_process` espera con `communicate()` y no sirve para procesos de
horas. Este runner lee `-progress pipe:1` bloque a bloque, reporta contra el
conteo de cuadros del indice, mata el proceso si la tarea se cancela y guarda
solo la cola del stderr.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PROGRESS_ARGS: tuple[str, ...] = ("-progress", "pipe:1", "-nostats")
STDERR_TAIL_BYTES = 64 * 1024
STDERR_CHUNK_BYTES = 4096
ERROR_MESSAGE_TAIL_CHARS = 500
PROGRESS_KEY = "progress"
PROGRESS_END = "end"

ProgressCallback = Callable[[float], None]
Spawner = Callable[[list[str]], Awaitable[asyncio.subprocess.Process]]


@dataclass(frozen=True)
class ProgressSnapshot:
    frame: int | None
    out_time_us: int | None
    done: bool


@dataclass(frozen=True)
class StderrTail:
    data: bytes
    truncated: bool

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class FfmpegRunResult:
    last: ProgressSnapshot | None
    stderr_tail: str
    stderr_truncated: bool


class FfmpegProcessError(RuntimeError):
    def __init__(self, message: str, returncode: int | None, stderr_tail: str) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


def parse_progress_line(line: str) -> tuple[str, str] | None:
    key, separator, value = line.strip().partition("=")
    if not separator or not key:
        return None
    return key.strip(), value.strip()


def _optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def snapshot_from_block(fields: Mapping[str, str]) -> ProgressSnapshot:
    return ProgressSnapshot(
        frame=_optional_int(fields.get("frame")),
        out_time_us=_optional_int(fields.get("out_time_us")),
        done=fields.get(PROGRESS_KEY) == PROGRESS_END,
    )


def fraction_done(snapshot: ProgressSnapshot, total_frames: int) -> float:
    if snapshot.done:
        return 1.0
    if snapshot.frame is None or total_frames <= 0:
        return 0.0
    return min(1.0, max(0.0, snapshot.frame / total_frames))


def append_bounded(tail: bytes, chunk: bytes, limit: int) -> bytes:
    return (tail + chunk)[-limit:]


def has_progress_pipe(command: Sequence[str]) -> bool:
    pairs = zip(command, command[1:])
    return any(flag == "-progress" and target == "pipe:1" for flag, target in pairs)


async def spawn_process(command: list[str], env: Mapping[str, str] | None = None) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=None if env is None else dict(env),
    )


def spawner_with_env(extra: Mapping[str, str]) -> Spawner:
    async def spawn(command: list[str]) -> asyncio.subprocess.Process:
        return await spawn_process(command, {**os.environ, **extra})

    return spawn


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    await process.wait()


async def _read_progress(
    stream: asyncio.StreamReader, total_frames: int, on_progress: ProgressCallback
) -> ProgressSnapshot | None:
    fields: dict[str, str] = {}
    last: ProgressSnapshot | None = None
    reported = 0.0
    async for raw in stream:
        parsed = parse_progress_line(raw.decode("utf-8", errors="replace"))
        if parsed is None:
            continue
        fields = {**fields, parsed[0]: parsed[1]}
        if parsed[0] != PROGRESS_KEY:
            continue
        last = snapshot_from_block(fields)
        reported = max(reported, fraction_done(last, total_frames))
        on_progress(reported)
        fields = {}
    return last


async def _read_stderr_tail(stream: asyncio.StreamReader, limit: int) -> StderrTail:
    tail = b""
    total = 0
    while chunk := await stream.read(STDERR_CHUNK_BYTES):
        total += len(chunk)
        tail = append_bounded(tail, chunk, limit)
    return StderrTail(data=tail, truncated=total > limit)


async def _start(command: list[str], spawn: Spawner) -> asyncio.subprocess.Process:
    try:
        return await spawn(command)
    except OSError as exc:
        raise FfmpegProcessError(f"Could not start '{Path(command[0]).name}': {exc}", None, "") from exc


def _pipes(process: asyncio.subprocess.Process) -> tuple[asyncio.StreamReader, asyncio.StreamReader]:
    if process.stdout is None or process.stderr is None:
        raise ValueError("The ffmpeg process must be spawned with stdout and stderr pipes")
    return process.stdout, process.stderr


async def _collect(
    process: asyncio.subprocess.Process, total_frames: int, on_progress: ProgressCallback, stderr_limit: int
) -> tuple[ProgressSnapshot | None, StderrTail]:
    stdout, stderr = _pipes(process)
    last, tail = await asyncio.gather(
        _read_progress(stdout, total_frames, on_progress),
        _read_stderr_tail(stderr, stderr_limit),
    )
    await process.wait()
    return last, tail


def _failure(command: Sequence[str], returncode: int | None, tail: StderrTail) -> FfmpegProcessError:
    name = Path(command[0]).name
    message = f"'{name}' exited with code {returncode}: {tail.text.strip()[-ERROR_MESSAGE_TAIL_CHARS:]}"
    return FfmpegProcessError(message, returncode, tail.text)


async def run_ffmpeg_with_progress(
    command: Sequence[str],
    *,
    total_frames: int,
    on_progress: ProgressCallback,
    stderr_limit: int = STDERR_TAIL_BYTES,
    spawn: Spawner = spawn_process,
) -> FfmpegRunResult:
    if not has_progress_pipe(command):
        raise ValueError("The ffmpeg command must include '-progress pipe:1'")
    argv = list(command)
    process = await _start(argv, spawn)
    try:
        last, tail = await _collect(process, total_frames, on_progress, stderr_limit)
    except BaseException:
        await asyncio.shield(_kill_process(process))
        raise
    if process.returncode != 0:
        raise _failure(argv, process.returncode, tail)
    return FfmpegRunResult(last=last, stderr_tail=tail.text, stderr_truncated=tail.truncated)
