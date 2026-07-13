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


class MediaTools:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ffmpeg_path = Path(settings.ffmpeg_binary)
        self.ffprobe_path = Path(settings.ffprobe_binary)

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
