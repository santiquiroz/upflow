from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.services.engines.realesrgan_ncnn import gpu_index_for_device
from app.services.process_runner import run_guarded_process

OUTPUT_FRAME_PATTERN = "%08d.png"


class RifeNcnnEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.binary_path = settings.rife_binary_path
        self.models_dir = settings.rife_models_path
        self.model_path = self.models_dir / settings.rife_model

    def available(self) -> bool:
        return self.settings.interpolation_available()

    async def run(
        self,
        frames_in: Path,
        frames_out: Path,
        source_frame_count: int,
        multiplier: int = 1,
        *,
        target_frame_count: int | None = None,
        device: str | None = None,
    ) -> Path:
        if not self.available():
            raise RuntimeError(
                "RIFE NCNN interpolation engine is not available. Run scripts/download-rife.ps1 first."
            )

        resolved_target_frame_count = self._resolve_target_frame_count(
            source_frame_count, multiplier, target_frame_count
        )
        frames_out.mkdir(parents=True, exist_ok=True)

        command = self._build_command(
            frames_in, frames_out, resolved_target_frame_count, self._gpu_index(device)
        )
        _, stderr, returncode = await run_guarded_process(command, self.settings.subprocess_timeout)

        if returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="ignore") or "Frame interpolation process failed")

        self._validate_output_frame_count(frames_out, resolved_target_frame_count)

        return frames_out

    @staticmethod
    def _resolve_target_frame_count(
        source_frame_count: int, multiplier: int, target_frame_count: int | None
    ) -> int:
        if target_frame_count is not None:
            return target_frame_count
        return source_frame_count * multiplier

    @staticmethod
    def _gpu_index(device: str | None) -> str:
        # RIFE runs on a Vulkan GPU (no CPU path). "cpu"/None -> "0" preserves the
        # historical default instead of letting gpu_index_for_device raise on cpu;
        # a dml:N id runs RIFE on the SAME GPU as the upscale (multi-GPU affinity).
        if device is None or device == "cpu":
            return "0"
        return gpu_index_for_device(device)

    def _build_command(
        self, frames_in: Path, frames_out: Path, target_frame_count: int, gpu_index: str
    ) -> list[str]:
        return [
            str(self.binary_path),
            "-i",
            str(frames_in),
            "-o",
            str(frames_out),
            "-m",
            str(self.model_path),
            "-n",
            str(target_frame_count),
            "-g",
            gpu_index,
            "-f",
            OUTPUT_FRAME_PATTERN,
        ]

    def _validate_output_frame_count(self, frames_out: Path, target_frame_count: int) -> None:
        actual_frame_count = self._count_output_frames(frames_out)

        if actual_frame_count == 0:
            raise RuntimeError("Frame interpolation completed but no output frames were produced")

        if actual_frame_count != target_frame_count:
            raise RuntimeError(
                "Frame interpolation completed with "
                f"{actual_frame_count} frames, expected {target_frame_count}"
            )

    @staticmethod
    def _count_output_frames(frames_out: Path) -> int:
        return sum(1 for _ in frames_out.glob("*.png"))
