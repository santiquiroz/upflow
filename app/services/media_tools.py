from __future__ import annotations

import asyncio
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from app.config import Settings

FFPROBE_TIMEOUT_SECONDS = 30
DEFAULT_FPS = Fraction(30, 1)


def parse_fps_fraction(value: str | None) -> Fraction | None:
    """Parses an ffprobe frame-rate string (e.g. "24000/1001") into a Fraction.

    Returns None for anything that cannot represent a real frame rate: empty/missing
    values, malformed strings, "0/0" (ZeroDivisionError), and non-positive rates
    such as "0/1".
    """
    if not value:
        return None
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if fraction <= 0:
        return None
    return fraction


def resolve_video_fps(avg_frame_rate: str | None, r_frame_rate: str | None) -> Fraction:
    """Resolves a valid fps from ffprobe fields, falling through avg -> r_frame_rate -> 30/1."""
    return (
        parse_fps_fraction(avg_frame_rate)
        or parse_fps_fraction(r_frame_rate)
        or DEFAULT_FPS
    )


def compute_interpolated_fps(source_fps: str | None, multiplier: int) -> Fraction:
    """Computes the encode framerate after RIFE interpolation: source_fps * multiplier.

    Keeps playback duration (and audio sync) identical once frame count is multiplied.
    Raises ValueError for anything parse_fps_fraction rejects (missing, malformed, <= 0).
    """
    fraction = parse_fps_fraction(source_fps)
    if fraction is None:
        raise ValueError(f"Cannot compute interpolated fps from invalid source fps: {source_fps!r}")
    return fraction * multiplier


def compute_target_frame_count(source_frame_count: int, source_fps: str | None, target_fps: str | None) -> int:
    """Computes the absolute RIFE `-n` target for TARGET_FPS mode.

    target_frames = round(source_count * target_fps / source_fps). Rounding means the
    resulting duration can drift from the source by less than one frame at target_fps.
    """
    source_fraction = parse_fps_fraction(source_fps)
    if source_fraction is None:
        raise ValueError(f"Cannot compute target frame count from invalid source fps: {source_fps!r}")
    target_fraction = parse_fps_fraction(target_fps)
    if target_fraction is None:
        raise ValueError(f"Cannot compute target frame count from invalid target fps: {target_fps!r}")
    return round(source_frame_count * target_fraction / source_fraction)


def format_fps_fraction(value: str | None) -> str:
    """Normalizes an fps value ("60", "60000/1001") into ffmpeg's "num/den" form."""
    fraction = parse_fps_fraction(value)
    if fraction is None:
        raise ValueError(f"Cannot format invalid fps value: {value!r}")
    return f"{fraction.numerator}/{fraction.denominator}"


class MediaTools:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ffmpeg_path = settings.ffmpeg_binary_path
        self.ffprobe_path = settings.ffprobe_binary_path

    def available(self) -> bool:
        return self.ffmpeg_path.exists() and self.ffprobe_path.exists()

    async def ffprobe_json(self, source_path: Path) -> dict[str, Any]:
        if not self.available():
            raise RuntimeError("FFmpeg/FFprobe not available. Run scripts/download-ffmpeg.ps1 first.")

        stdout = await self._run_ffprobe(source_path)
        return json.loads(stdout)

    def _build_ffprobe_command(self, source_path: Path) -> list[str]:
        return [
            str(self.ffprobe_path),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(source_path),
        ]

    async def _run_ffprobe(self, source_path: Path) -> str:
        command = self._build_ffprobe_command(source_path)
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=FFPROBE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            await self._kill_process(process)
            raise RuntimeError(f"ffprobe timed out after {FFPROBE_TIMEOUT_SECONDS}s") from exc
        except asyncio.CancelledError:
            await self._kill_process(process)
            raise

        stdout_text = stdout.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode or -1,
                command,
                output=stdout_text,
                stderr=stderr.decode("utf-8", errors="replace"),
            )
        return stdout_text

    @staticmethod
    async def _kill_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            process.kill()
        await process.wait()
