from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.models import UpscaleJob
from app.services.engines.base import UpscaleEngine


class RealEsrganNcnnEngine(UpscaleEngine):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.binary_path = Path(settings.engine_binary)
        self.models_dir = Path(settings.engine_models_dir)

    def available(self) -> bool:
        return self.binary_path.exists() and self.models_dir.exists()

    async def run(self, job: UpscaleJob) -> Path:
        if not self.available():
            raise RuntimeError(
                "Real-ESRGAN NCNN engine is not available. Run scripts/download-realesrgan.ps1 first."
            )

        output_suffix = f".{job.output_format.lower()}"
        output_path = self.settings.outputs_path / f"{job.id}{output_suffix}"

        command = [
            str(self.binary_path),
            "-i",
            str(job.source_path),
            "-o",
            str(output_path),
            "-n",
            job.model_name,
            "-s",
            str(job.scale),
            "-m",
            str(self.models_dir),
            "-f",
            job.output_format.lower(),
            "-g",
            "0",
        ]

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()

        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="ignore") or "Upscaling process failed")

        if not output_path.exists():
            raise RuntimeError("Upscaling process completed but no output file was produced")

        return output_path
