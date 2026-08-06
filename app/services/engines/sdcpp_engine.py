from __future__ import annotations

import logging
from pathlib import Path

from app.config import Settings
from app.services.missing_pack import missing_pack_message
from app.services.process_runner import run_guarded_process

# ---------------------------------------------------------------------------
# Fase 3 (2026-07-31) - lane EXPERIMENTAL de difusión por stable-diffusion.cpp
# con backend Vulkan (multi-vendor: corre sin ningún execution provider de
# onnxruntime). Mismo rol que GMFSS: opt-in por flag, jamás en el camino
# default. El checkpoint lo provee el usuario (SDCPP_MODEL): los .safetensors
# originales no sobreviven a la instalación ONNX. Texto-a-imagen solamente;
# img2img/inpaint quedan fuera de este lane a propósito.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

SDCPP_MODEL_ID = "experimental-sdcpp"
SDCPP_MODEL_LABEL = "sd.cpp Vulkan (experimental)"


class SdcppEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return self.settings.sdcpp_available()

    def build_command(self, request, output_path: Path, checkpoint: Path | None = None) -> list[str]:
        model_path = checkpoint or self.settings.sdcpp_model_path
        command = [
            str(self.settings.sdcpp_binary_path),
            "-m", str(model_path),
            "-p", request.prompt,
            "--steps", str(request.steps),
            "--cfg-scale", str(request.guidance),
            "-W", str(request.width),
            "-H", str(request.height),
            "-o", str(output_path),
        ]
        if request.negative_prompt:
            command += ["-n", request.negative_prompt]
        if request.seed is not None:
            command += ["-s", str(request.seed)]
        return command

    async def run(self, request, output_path: Path, checkpoint: Path | None = None) -> Path:
        if not self.available():
            raise RuntimeError(
                missing_pack_message(
                    "sdcpp",
                    detail="Ademas hay que prender ENABLE_SDCPP=true y apuntar SDCPP_MODEL a un checkpoint.",
                )
            )
        command = self.build_command(request, output_path, checkpoint)
        stdout, stderr, returncode = await run_guarded_process(
            command, self.settings.subprocess_timeout
        )
        if returncode != 0:
            tail = stderr.decode(errors="replace")[-600:]
            raise RuntimeError(f"sd.cpp failed with exit code {returncode}: {tail}")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("sd.cpp exited 0 but produced no output image")
        return output_path
