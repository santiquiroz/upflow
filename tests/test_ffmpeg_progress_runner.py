from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest

from app.config import Settings
from app.services.ffmpeg_progress_runner import (
    PROGRESS_ARGS,
    STDERR_TAIL_BYTES,
    FfmpegProcessError,
    ProgressSnapshot,
    append_bounded,
    fraction_done,
    has_progress_pipe,
    parse_progress_line,
    run_ffmpeg_with_progress,
    snapshot_from_block,
    spawn_process,
)
from ffmpeg_support import needs_ffmpeg

FAKE_TIMEOUT_SECONDS = 20


def fake_ffmpeg(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "fake_ffmpeg.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(script), *PROGRESS_ARGS]


def progress_block(frame: int, out_time_us: int, state: str = "continue") -> str:
    return f"frame={frame}\nfps=25.0\nout_time_us={out_time_us}\nprogress={state}\n"


def emitting_script(blocks: list[str], exit_code: int = 0, stderr: str = "") -> str:
    lines = "".join(f"sys.stdout.write({block!r})\nsys.stdout.flush()\n" for block in blocks)
    return f"import sys\nsys.stderr.write({stderr!r})\n{lines}sys.exit({exit_code})\n"


class Recorder:
    def __init__(self) -> None:
        self.values: list[float] = []

    def __call__(self, fraction: float) -> None:
        self.values.append(fraction)


class SpawnSpy:
    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None

    async def __call__(self, command: list[str]) -> asyncio.subprocess.Process:
        self.process = await spawn_process(command)
        return self.process


async def run_fake(command: list[str], total_frames: int, **kwargs: object):
    recorder = Recorder()
    result = await asyncio.wait_for(
        run_ffmpeg_with_progress(command, total_frames=total_frames, on_progress=recorder, **kwargs),
        timeout=FAKE_TIMEOUT_SECONDS,
    )
    return result, recorder.values


def test_parse_progress_line_splits_key_and_value() -> None:
    assert parse_progress_line("frame=42\r\n") == ("frame", "42")
    assert parse_progress_line("out_time_us=1000000") == ("out_time_us", "1000000")


def test_parse_progress_line_ignores_lines_without_equals() -> None:
    assert parse_progress_line("\n") is None
    assert parse_progress_line("garbage") is None


def test_snapshot_from_block_reads_frame_time_and_end() -> None:
    snapshot = snapshot_from_block({"frame": "12", "out_time_us": "480000", "progress": "end"})

    assert snapshot == ProgressSnapshot(frame=12, out_time_us=480000, done=True)


def test_snapshot_from_block_tolerates_na_values() -> None:
    snapshot = snapshot_from_block({"frame": "N/A", "out_time_us": "N/A", "progress": "continue"})

    assert snapshot == ProgressSnapshot(frame=None, out_time_us=None, done=False)


def test_fraction_done_counts_against_the_index_and_clamps() -> None:
    assert fraction_done(ProgressSnapshot(frame=25, out_time_us=None, done=False), 100) == 0.25
    assert fraction_done(ProgressSnapshot(frame=130, out_time_us=None, done=False), 100) == 1.0
    assert fraction_done(ProgressSnapshot(frame=None, out_time_us=None, done=False), 100) == 0.0
    assert fraction_done(ProgressSnapshot(frame=10, out_time_us=None, done=False), 0) == 0.0


def test_fraction_done_is_complete_at_progress_end() -> None:
    assert fraction_done(ProgressSnapshot(frame=90, out_time_us=None, done=True), 100) == 1.0


def test_append_bounded_keeps_only_the_tail() -> None:
    assert append_bounded(b"abc", b"def", 4) == b"cdef"
    assert append_bounded(b"", b"xy", 4) == b"xy"


def test_has_progress_pipe_requires_progress_to_stdout() -> None:
    assert has_progress_pipe(["ffmpeg", *PROGRESS_ARGS, "-i", "in.mkv", "out.mkv"])
    assert not has_progress_pipe(["ffmpeg", "-i", "in.mkv", "out.mkv"])
    assert not has_progress_pipe(["ffmpeg", "-progress", "progress.txt", "-i", "in.mkv"])


async def test_a_command_without_the_progress_pipe_is_rejected() -> None:
    with pytest.raises(ValueError):
        await run_ffmpeg_with_progress(["ffmpeg", "-i", "in.mkv"], total_frames=10, on_progress=Recorder())


async def test_streamed_progress_is_monotonic_and_ends_complete(tmp_path: Path) -> None:
    blocks = [progress_block(frame, frame * 40_000) for frame in (10, 40, 30, 70)]
    blocks.append(progress_block(100, 4_000_000, "end"))
    command = fake_ffmpeg(tmp_path, emitting_script(blocks))

    result, values = await run_fake(command, total_frames=100)

    assert values == sorted(values)
    assert values[0] == pytest.approx(0.1)
    assert values[-1] == 1.0
    assert result.last == ProgressSnapshot(frame=100, out_time_us=4_000_000, done=True)


async def test_progress_is_reported_block_by_block_while_the_process_runs(tmp_path: Path) -> None:
    marker = tmp_path / "release"
    command = fake_ffmpeg(
        tmp_path,
        f"""
        import pathlib, sys, time
        sys.stdout.write("frame=5\\nprogress=continue\\n")
        sys.stdout.flush()
        while not pathlib.Path({str(marker)!r}).exists():
            time.sleep(0.05)
        sys.stdout.write("frame=10\\nprogress=end\\n")
        sys.stdout.flush()
        """,
    )
    recorder = Recorder()
    task = asyncio.create_task(run_ffmpeg_with_progress(command, total_frames=10, on_progress=recorder))

    await wait_until(lambda: recorder.values == [0.5])
    marker.write_text("go", encoding="utf-8")
    await asyncio.wait_for(task, timeout=FAKE_TIMEOUT_SECONDS)

    assert recorder.values == [0.5, 1.0]


async def test_cancel_kills_the_process_and_raises_cancelled(tmp_path: Path) -> None:
    command = fake_ffmpeg(
        tmp_path,
        """
        import sys, time
        sys.stdout.write("frame=3\\nprogress=continue\\n")
        sys.stdout.flush()
        time.sleep(120)
        """,
    )
    recorder = Recorder()
    spy = SpawnSpy()
    task = asyncio.create_task(
        run_ffmpeg_with_progress(command, total_frames=30, on_progress=recorder, spawn=spy)
    )
    await wait_until(lambda: recorder.values == [0.1])

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=FAKE_TIMEOUT_SECONDS)

    assert spy.process is not None
    assert spy.process.returncode is not None


async def test_success_returns_the_bounded_stderr_tail(tmp_path: Path) -> None:
    noise = "x" * (STDERR_TAIL_BYTES * 3) + "LAST-WARNING"
    command = fake_ffmpeg(tmp_path, emitting_script([progress_block(1, 0, "end")], stderr=noise))

    result, _ = await run_fake(command, total_frames=1)

    assert len(result.stderr_tail.encode("utf-8")) <= STDERR_TAIL_BYTES
    assert result.stderr_tail.endswith("LAST-WARNING")
    assert result.stderr_truncated


async def test_failure_raises_with_returncode_and_bounded_stderr(tmp_path: Path) -> None:
    noise = "y" * 5000 + "Invalid argument"
    command = fake_ffmpeg(tmp_path, emitting_script([progress_block(1, 0)], exit_code=3, stderr=noise))

    with pytest.raises(FfmpegProcessError) as caught:
        await run_fake(command, total_frames=10, stderr_limit=1024)

    assert caught.value.returncode == 3
    assert len(caught.value.stderr_tail) <= 1024
    assert caught.value.stderr_tail.endswith("Invalid argument")
    assert "Invalid argument" in str(caught.value)


async def test_heavy_stderr_does_not_block_the_progress_stream(tmp_path: Path) -> None:
    command = fake_ffmpeg(
        tmp_path,
        """
        import sys
        for index in range(200):
            sys.stderr.write("warning " * 1000 + "\\n")
            sys.stdout.write(f"frame={index + 1}\\nprogress=continue\\n")
        sys.stdout.write("frame=200\\nprogress=end\\n")
        """,
    )

    result, values = await run_fake(command, total_frames=200)

    assert values[-1] == 1.0
    assert values == sorted(values)
    assert result.stderr_truncated


async def test_a_missing_binary_raises_process_error(tmp_path: Path) -> None:
    command = [str(tmp_path / "missing-ffmpeg.exe"), *PROGRESS_ARGS]

    with pytest.raises(FfmpegProcessError):
        await run_ffmpeg_with_progress(command, total_frames=1, on_progress=Recorder())


@needs_ffmpeg
async def test_the_real_ffmpeg_streams_progress_to_the_end() -> None:
    ffmpeg = str(Settings().ffmpeg_binary_path)
    command = [
        ffmpeg, "-hide_banner", "-nostdin", *PROGRESS_ARGS,
        "-f", "lavfi", "-i", "testsrc=size=64x48:rate=25:duration=2",
        "-f", "null", "-",
    ]

    result, values = await run_fake(command, total_frames=50)

    assert values[-1] == 1.0
    assert values == sorted(values)
    assert result.last is not None and result.last.frame == 50


async def wait_until(condition, timeout: float = FAKE_TIMEOUT_SECONDS) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(poll(), timeout=timeout)
