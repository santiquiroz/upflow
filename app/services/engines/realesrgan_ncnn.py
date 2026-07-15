from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import UpscaleJob
from app.services.engines.base import UpscaleEngine
from app.services.process_runner import run_guarded_process


def gpu_index_for_device(device: str | None) -> str:
    """Maps a device id to the ncnn `-g` GPU index argument.

    `None` (no device resolved, e.g. a job created before device wiring)
    preserves the historical hardcoded "-g 0" behavior. "cpu" has no ncnn
    Vulkan equivalent -- validation must reject it before an engine ever
    runs, so reaching this function with "cpu" is a bug, not user error.

    IMPORTANT ordering caveat (SP1 fast-follow I2): `dml:N` ids come from
    DXGI adapter enumeration (devices_service.py), but `-g N` here is a
    Vulkan physical-device index consumed by the ncnn binary. This function
    assumes DXGI order == Vulkan order for the same N, which is NOT
    guaranteed by either API on a multi-adapter machine. The only mapping
    empirically verified end-to-end is the single-dGPU default (`dml:0` ->
    `-g 0`, see `.superpowers/sdd/sp1-task-8-smoke-report.md`, PART A). The
    onnx/DirectML path (`_create_session` in onnx_upscaler.py) does NOT have
    this problem: `device_id` is passed straight through to
    `DmlExecutionProvider`, which resolves it against the same DXGI-ordered
    list -- no second enumeration to drift out of sync with. There is no way
    to query ncnn's own Vulkan device list from Python to verify this
    mapping without the binary itself, so treat `-g N` for N > 0 as
    best-effort on multi-GPU systems until verified on real hardware.
    """
    if device is None:
        return "0"
    if device.startswith("dml:"):
        return device.partition(":")[2]
    if device == "cpu":
        raise RuntimeError("Real-ESRGAN NCNN engine requires a Vulkan GPU device; 'cpu' is not supported")
    return "0"


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
            gpu_index_for_device(job.device),
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
