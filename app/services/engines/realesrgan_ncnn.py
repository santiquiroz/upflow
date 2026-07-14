from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import UpscaleJob
from app.services.engines.base import UpscaleEngine
from app.services.process_runner import run_guarded_process


class RealEsrganNcnnEngine(UpscaleEngine):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.binary_path = settings.engine_binary_path
        self.models_dir = settings.engine_models_path

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

        _, stderr, returncode = await run_guarded_process(command, self.settings.subprocess_timeout)

        if returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="ignore") or "Upscaling process failed")

        if not self._is_non_empty_file(output_path):
            raise RuntimeError("Upscaling process completed but no output file was produced")

        return output_path

    @staticmethod
    def _is_non_empty_file(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0
