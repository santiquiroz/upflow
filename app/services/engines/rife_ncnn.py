from __future__ import annotations

import os
import struct
from pathlib import Path

from app.config import Settings
from app.services.engines.realesrgan_ncnn import gpu_index_for_device
from app.services.process_runner import run_guarded_process

OUTPUT_FRAME_PATTERN = "%08d.png"
# RIFE's -u (UHD mode) halves the flow-estimation resolution; it exists for
# >2K inputs, where full-res flow is the dominant cost. Threshold ~1440p.
UHD_PIXEL_THRESHOLD = 2560 * 1440
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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
        command = [
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
            "-j",
            self._thread_spec(),
            "-f",
            OUTPUT_FRAME_PATTERN,
        ]
        if self._use_uhd_mode(frames_in):
            command.append("-u")
        return command

    def _thread_spec(self) -> str:
        # El default del binario es 1:2:2 -- UN solo thread decodificando PNG.
        # A 4K cada frame pesa 10-20MB y la GPU se queda esperando I/O (misma
        # leccion que los save-threads del upscaler ONNX en SP11).
        configured = self.settings.rife_threads
        if configured != "auto":
            return configured
        return _auto_thread_spec(os.cpu_count())

    def _use_uhd_mode(self, frames_in: Path) -> bool:
        mode = self.settings.rife_uhd_mode
        if mode == "on":
            return True
        if mode == "off":
            return False
        dimensions = _first_frame_dimensions(frames_in)
        if dimensions is None:
            return False
        width, height = dimensions
        return width * height >= UHD_PIXEL_THRESHOLD

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


def _auto_thread_spec(cpu_count: int | None) -> str:
    cpus = cpu_count or 4
    load = min(8, max(2, cpus // 2))
    proc = min(4, max(2, cpus // 4))
    save = min(12, max(2, cpus))
    return f"{load}:{proc}:{save}"


def _first_frame_dimensions(frames_in: Path) -> tuple[int, int] | None:
    first = next(iter(sorted(frames_in.glob("*.png"))), None)
    if first is None:
        return None
    try:
        with open(first, "rb") as handle:
            header = handle.read(24)
    except OSError:
        return None
    if len(header) < 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)
