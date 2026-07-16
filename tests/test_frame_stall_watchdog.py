from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.stall_watchdog import StallWatchdog
from app.services.video_upscaler import VideoStallError, VideoUpscaler

# ---------------------------------------------------------------------------
# SP5 fix - stall watchdog replaces the fixed subprocess timeout for the video
# pipeline: a job making progress (new frames / growing output file) must
# never be killed, no matter how long it runs. Only a genuine stall (no new
# output for FRAME_STALL_TIMEOUT_SECONDS) cancels the stage.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), **overrides)  # type: ignore[arg-type]


def make_upscaler(
    tmp_path: Path, *, frame_poll_interval_seconds: float, frame_stall_timeout_seconds: float
) -> VideoUpscaler:
    settings = make_settings(tmp_path)
    return VideoUpscaler(
        settings,
        engine=object(),  # type: ignore[arg-type]
        media_tools=object(),  # type: ignore[arg-type]
        frame_poll_interval_seconds=frame_poll_interval_seconds,
        frame_stall_timeout_seconds=frame_stall_timeout_seconds,
    )


def make_video_job(source_path: Path) -> VideoUpscaleJob:
    return VideoUpscaleJob(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )


# ---------------------------------------------------------------------------
# StallWatchdog (pure logic)
# ---------------------------------------------------------------------------


def test_stall_watchdog_not_stalled_while_value_keeps_increasing() -> None:
    watchdog = StallWatchdog(stall_timeout_seconds=0.05)

    assert watchdog.observe(1) is False
    time.sleep(0.03)
    assert watchdog.observe(2) is False
    time.sleep(0.03)
    assert watchdog.observe(3) is False
    assert watchdog.triggered is False


def test_stall_watchdog_triggers_once_value_stops_changing() -> None:
    watchdog = StallWatchdog(stall_timeout_seconds=0.05)

    watchdog.observe(1)
    time.sleep(0.02)
    assert watchdog.observe(1) is False, "should not trigger before the stall timeout elapses"
    time.sleep(0.06)
    assert watchdog.observe(1) is True
    assert watchdog.triggered is True


def test_stall_watchdog_resets_clock_when_value_grows_again() -> None:
    watchdog = StallWatchdog(stall_timeout_seconds=0.05)

    watchdog.observe(1)
    time.sleep(0.03)
    watchdog.observe(2)  # progress resets the stall clock
    time.sleep(0.03)

    assert watchdog.observe(2) is False, "only 0.03s elapsed since the reset, under the 0.05s threshold"


# ---------------------------------------------------------------------------
# Frame stages (extract / upscale / interpolate) via _track_frame_progress
# ---------------------------------------------------------------------------


async def test_frame_stage_progressing_slowly_is_not_killed(tmp_path: Path) -> None:
    output_dir = tmp_path / "frames-out"
    output_dir.mkdir()
    upscaler = make_upscaler(tmp_path, frame_poll_interval_seconds=0.01, frame_stall_timeout_seconds=0.15)
    job = make_video_job(tmp_path / "input.mp4")

    async def slow_but_steady_stage() -> None:
        for index in range(5):
            await asyncio.sleep(0.02)
            (output_dir / f"{index:08d}.png").write_bytes(b"frame")

    async with upscaler._track_frame_progress(job, output_dir, "upscaling_frames"):
        await slow_but_steady_stage()

    assert job.metadata["framesDone"] == 5


async def test_frame_stage_that_stops_producing_frames_is_killed_with_clear_stall_message(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "frames-out"
    output_dir.mkdir()
    upscaler = make_upscaler(tmp_path, frame_poll_interval_seconds=0.01, frame_stall_timeout_seconds=0.05)
    job = make_video_job(tmp_path / "input.mp4")

    async def hangs_after_one_frame() -> None:
        (output_dir / "00000001.png").write_bytes(b"frame")
        await asyncio.Event().wait()

    tasks_before = asyncio.all_tasks()

    with pytest.raises(VideoStallError) as exc_info:
        async with upscaler._track_frame_progress(job, output_dir, "upscaling_frames"):
            await hangs_after_one_frame()

    assert "estancado" in str(exc_info.value)
    assert "timed out" not in str(exc_info.value)

    tasks_after = asyncio.all_tasks()
    assert tasks_after - tasks_before == set(), "poller task must not be left dangling after a stall"


async def test_stall_watchdog_does_not_mask_a_real_stage_exception(tmp_path: Path) -> None:
    output_dir = tmp_path / "frames-out"
    output_dir.mkdir()
    upscaler = make_upscaler(tmp_path, frame_poll_interval_seconds=0.01, frame_stall_timeout_seconds=5.0)
    job = make_video_job(tmp_path / "input.mp4")

    async def failing_stage() -> None:
        await asyncio.sleep(0.01)
        raise RuntimeError("boom: real ffmpeg failure")

    with pytest.raises(RuntimeError, match="boom: real ffmpeg failure"):
        async with upscaler._track_frame_progress(job, output_dir, "upscaling_frames"):
            await failing_stage()


async def test_external_cancellation_of_a_frame_stage_is_not_reported_as_stall(tmp_path: Path) -> None:
    output_dir = tmp_path / "frames-out"
    output_dir.mkdir()
    upscaler = make_upscaler(tmp_path, frame_poll_interval_seconds=0.01, frame_stall_timeout_seconds=5.0)
    job = make_video_job(tmp_path / "input.mp4")

    async def run_stage() -> None:
        async with upscaler._track_frame_progress(job, output_dir, "upscaling_frames"):
            await asyncio.Event().wait()

    task = asyncio.create_task(run_stage())
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# Encode stage via _track_encode_progress (output file size growth)
# ---------------------------------------------------------------------------


async def test_encode_stage_with_growing_output_is_not_killed(tmp_path: Path) -> None:
    output_path = tmp_path / "out.mp4"
    upscaler = make_upscaler(tmp_path, frame_poll_interval_seconds=0.01, frame_stall_timeout_seconds=0.15)

    async def growing_output() -> None:
        for _ in range(5):
            await asyncio.sleep(0.02)
            with output_path.open("ab") as handle:
                handle.write(b"x" * 100)

    async with upscaler._track_encode_progress(output_path):
        await growing_output()

    assert output_path.stat().st_size == 500


async def test_encode_stage_with_stalled_output_is_killed_with_clear_stall_message(tmp_path: Path) -> None:
    output_path = tmp_path / "out.mp4"
    output_path.write_bytes(b"partial-mux-header")
    upscaler = make_upscaler(tmp_path, frame_poll_interval_seconds=0.01, frame_stall_timeout_seconds=0.05)

    async def hangs() -> None:
        await asyncio.Event().wait()

    tasks_before = asyncio.all_tasks()

    with pytest.raises(VideoStallError) as exc_info:
        async with upscaler._track_encode_progress(output_path):
            await hangs()

    assert "estancado" in str(exc_info.value)
    assert "timed out" not in str(exc_info.value)

    tasks_after = asyncio.all_tasks()
    assert tasks_after - tasks_before == set(), "poller task must not be left dangling after a stall"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_frame_stall_timeout_seconds_has_a_15_minute_default(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.frame_stall_timeout_seconds == 900


def test_frame_stall_timeout_seconds_is_configurable(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, FRAME_STALL_TIMEOUT_SECONDS=120)
    assert settings.frame_stall_timeout_seconds == 120


def test_subprocess_timeout_default_is_a_24h_backstop(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    assert settings.subprocess_timeout == 86400


def test_video_upscaler_defaults_frame_stall_timeout_from_settings(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, FRAME_STALL_TIMEOUT_SECONDS=42)
    upscaler = VideoUpscaler(settings, engine=object(), media_tools=object())  # type: ignore[arg-type]

    assert upscaler.frame_stall_timeout_seconds == 42
