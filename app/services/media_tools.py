from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

from app.config import Settings

FFPROBE_TIMEOUT_SECONDS = 30


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

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_ffprobe, source_path),
                timeout=FFPROBE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(f"ffprobe timed out after {FFPROBE_TIMEOUT_SECONDS}s") from exc
        return json.loads(result.stdout)

    def _run_ffprobe(self, source_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                str(self.ffprobe_path),
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(source_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
