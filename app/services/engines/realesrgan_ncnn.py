from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from app.config import Settings
from app.models import UpscaleJob
from app.services.dml_device import DML_DEVICE_PREFIX
from app.services.engines.base import UpscaleEngine
from app.services.missing_pack import missing_pack_message
from app.services import process_runner
from app.services.process_runner import is_non_empty_file
from app.services.resource_probes import DxgiVramProbe, ResourceProbe
from app.services.scale_fit import engine_output_path, native_scale_of
from app.services.tile_params import NCNN_AUTO_TILE, NCNN_PREPADDING_PX, choose_ncnn_tile

# El binario acepta jpg/png/webp en `-f`; "jpeg" (valido en la API) lo rechaza.
_NCNN_FORMAT_ALIASES = {"jpeg": "jpg"}
# Un fallo de Vulkan (tipicamente VRAM agotada por un tile grande) sale por stderr
# como "vkAllocateMemory failed -2" y el binario igual termina con exit 0 y una
# imagen plana. Medido 2026-09-02 con -t 1024 en una RX 7800 XT. Solo llamadas
# vk*: "decode image X failed" es otra cosa (entrada invalida, sin archivo de salida).
_VULKAN_FAILURE = re.compile(rb"\bvk\w+ failed\b", re.IGNORECASE)


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
    if device.startswith(DML_DEVICE_PREFIX):
        return device.partition(":")[2]
    if device == "cpu":
        raise RuntimeError("Real-ESRGAN NCNN engine requires a Vulkan GPU device; 'cpu' is not supported")
    return "0"


def ncnn_output_format(output_path: Path) -> str:
    suffix = output_path.suffix.lstrip(".").lower()
    return _NCNN_FORMAT_ALIASES.get(suffix, suffix)


def vulkan_failure_line(stderr: bytes) -> str | None:
    for line in stderr.splitlines():
        if _VULKAN_FAILURE.search(line):
            return line.decode("utf-8", errors="ignore").strip()
    return None


def raise_on_ncnn_failure(returncode: int, stderr: bytes) -> None:
    if returncode != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="ignore") or "Upscaling process failed")
    failure = vulkan_failure_line(stderr)
    if failure is not None:
        raise RuntimeError(
            f"Real-ESRGAN NCNN reported a Vulkan failure ({failure}); usually the tile does not fit "
            "in VRAM. Retry with a smaller tile_size (or leave it unset for auto)."
        )


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


class RealEsrganNcnnEngine(UpscaleEngine):
    def __init__(self, settings: Settings, vram_probe: ResourceProbe | None = None) -> None:
        self.settings = settings
        self.binary_path = settings.engine_binary_path
        self.models_dir = settings.engine_models_path
        self.vram_probe = vram_probe or DxgiVramProbe()

    def available(self) -> bool:
        return self.binary_path.exists() and self.models_dir.exists()

    async def run(self, job: UpscaleJob) -> Path:
        if not self.available():
            raise RuntimeError(missing_pack_message("realesrgan"))

        output_path = engine_output_path(self.settings, job)
        tile = self.tile_for(job)
        command = self.build_command(job, output_path, tile)
        job.metadata["effective"] = self.describe(job, command, tile)

        # Via el modulo (no import directo): los tests parchean process_runner.run_guarded_process.
        _, stderr, returncode = await process_runner.run_guarded_process(command, self.settings.subprocess_timeout)
        try:
            raise_on_ncnn_failure(returncode, stderr)
            if not is_non_empty_file(output_path):
                raise RuntimeError("Upscaling process completed but no output file was produced")
        except RuntimeError:
            # El binario deja una imagen plana al fallar Vulkan: no puede quedar en outputs/.
            output_path.unlink(missing_ok=True)
            raise

        return output_path

    def build_command(self, job: UpscaleJob, output_path: Path, tile: int) -> list[str]:
        return [
            str(self.binary_path),
            "-i",
            str(job.source_path),
            "-o",
            str(output_path),
            "-n",
            job.model_name,
            "-s",
            str(native_scale_of(job)),
            "-t",
            str(tile),
            "-m",
            str(self.models_dir),
            "-f",
            ncnn_output_format(output_path),
            "-g",
            gpu_index_for_device(job.device),
        ]

    def tile_for(self, job: UpscaleJob) -> int:
        requested = getattr(job, "tile_size", None)
        if requested is None:
            return NCNN_AUTO_TILE
        if requested > 0:
            return requested
        width, height = image_size(job.source_path)
        free_vram_mb = self.vram_probe.free_capacity_mb(job.device or "")
        return choose_ncnn_tile(width, height, free_vram_mb)

    @staticmethod
    def describe(job: UpscaleJob, command: list[str], tile: int) -> dict[str, Any]:
        return {
            "engine": "realesrgan-ncnn-vulkan",
            "model": job.model_name,
            "device": job.device,
            "gpuIndex": gpu_index_for_device(job.device),
            "nativeScale": native_scale_of(job),
            "requestedScale": job.scale,
            "tileSize": tile,
            "tileSizeMeaning": "binary auto (heap-based)" if tile == NCNN_AUTO_TILE else "explicit",
            "tileOverlap": NCNN_PREPADDING_PX,
            "command": command,
        }
