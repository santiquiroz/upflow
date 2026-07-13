from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from app.config import Settings


class MediaTools:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ffmpeg_path = Path(settings.ffmpeg_binary)
        self.ffprobe_path = Path(settings.ffprobe_binary)

    def available(self) -> bool:
        return self.ffmpeg_path.exists() and self.ffprobe_path.exists()

    def ffprobe_json(self, source_path: Path) -> dict[str, Any]:
        if not self.available():
            raise RuntimeError("FFmpeg/FFprobe not available. Run scripts/download-ffmpeg.ps1 first.")

        result = subprocess.run(
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
        return json.loads(result.stdout)
