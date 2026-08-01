from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.services.process_runner import run_guarded_process

# ---------------------------------------------------------------------------
# Generación de video por stable-diffusion.cpp con backend Vulkan.
#
# El mismo binario que ya genera imágenes (`sd-cli.exe`) trae `-M vid_gen`, así
# que este lane no agrega runtime nuevo: corre en AMD, NVIDIA e Intel sin CUDA,
# sin ROCm y sin ComfyUI. Un pack = un difusor + el VAE + el text encoder, los
# tres en `{SDCPP_MODELS_DIR}/video/`. Esa subcarpeta es deliberada: la lista de
# modelos de imagen recorre el directorio padre sin recursión, así que el VAE y
# el encoder no aparecen como si fueran modelos de imagen.
#
# Las constantes de acá abajo salieron de corridas reales en una RX 7800 XT
# (medidas en docs/superpowers/specs/2026-08-01-video-generation-findings.md),
# no del manual: cada una evita un fallo concreto.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

VIDEO_MODEL_PREFIX = "sdcppvid:"
VIDEO_SUBDIR = "video"
CHECKPOINT_SUFFIXES = (".gguf", ".safetensors")

VAE_MARKER = "vae"
ENCODER_MARKER = "umt5"
# Decodificador chico (23 MB) que reemplaza al VAE de Wan en el decode. Medido en
# una RX 7800 XT: 85,8 s con el VAE nativo contra 5,4 s con este, y sin él no
# entran 832x480 en 16 GB. El nombre no contiene "vae", así que no se confunden.
TAE_MARKER = "taew"

# El encoder umt5-xxl pide un único buffer de ~4,2 GB y el driver Vulkan de AMD
# topea el buffer individual en 4 GiB: revienta con 16 GB de VRAM libres. Va a
# CPU siempre. Cuesta ~17 s por prompt contra minutos de difusión.
BACKEND_ASSIGNMENT = "te=cpu,diffusion=vulkan0,vae=vulkan0"

# Sin flash attention, Vulkan devuelve artefactos verdes en vez de video.
# Vulkan1 suele ser la iGPU: fijar el dispositivo evita generar en la integrada.
FLASH_ATTENTION_FLAG = "--diffusion-fa"
# Solo hace falta cuando decodifica el VAE grande: con el TAE el decode ya no es
# el cuello y el tiling solo agregaría costuras.
VAE_TILING_FLAG = "--vae-tiling"

# El VAE de Wan comprime 4x en el eje temporal: los frames válidos son 4n+1.
FRAME_PERIOD = 4
# Fallos no lineales por alineación (287 anda, 288 no): las dimensiones se pegan a 32.
DIMENSION_ALIGNMENT = 32

TURBO_MARKER = "turbo"
# El destilado se entrenó con self-forcing a 4 pasos y CFG 1; con los defaults
# del modelo base sale quemado.
TURBO_STEPS, TURBO_GUIDANCE = 4, 1.0
BASE_STEPS, BASE_GUIDANCE = 20, 5.0

SAMPLING_METHOD = "euler"
FLOW_SHIFT = "3.0"


@dataclass(frozen=True, slots=True)
class VideoModel:
    id: str
    name: str
    diffusion: Path
    vae: Path
    encoder: Path
    turbo: bool
    tae: Path | None = None

    @property
    def default_steps(self) -> int:
        return TURBO_STEPS if self.turbo else BASE_STEPS

    @property
    def default_guidance(self) -> float:
        return TURBO_GUIDANCE if self.turbo else BASE_GUIDANCE

    @property
    def ignores_negative_prompt(self) -> bool:
        """Con CFG 1 no hay pasada incondicional: el prompt negativo no se usa."""
        return self.turbo


@dataclass(frozen=True, slots=True)
class VideoRequest:
    prompt: str
    negative_prompt: str | None = None
    steps: int | None = None
    guidance: float | None = None
    width: int = 704
    height: int = 704
    seed: int | None = None
    frames: int = 33
    fps: int = 16
    init_image: Path | None = None


def normalize_frames(frames: int) -> int:
    if frames <= 1:
        return 1
    return ((frames - 1) // FRAME_PERIOD) * FRAME_PERIOD + 1


def align_dimension(value: int) -> int:
    aligned = round(value / DIMENSION_ALIGNMENT) * DIMENSION_ALIGNMENT
    return max(DIMENSION_ALIGNMENT, aligned)


def video_model_id(path: Path) -> str:
    return f"{VIDEO_MODEL_PREFIX}{path.stem}"


def _video_dir(settings: Settings) -> Path:
    return settings.sdcpp_models_dir_path / VIDEO_SUBDIR


def _find_support_file(files: list[Path], marker: str) -> Path | None:
    for path in files:
        if marker in path.name.lower():
            return path
    return None


def _is_support_file(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in (VAE_MARKER, ENCODER_MARKER, TAE_MARKER))


def list_video_models(settings: Settings) -> list[VideoModel]:
    directory = _video_dir(settings)
    if not directory.is_dir():
        return []

    files = [p for p in sorted(directory.iterdir())
             if p.is_file() and p.suffix.lower() in CHECKPOINT_SUFFIXES]
    vae = _find_support_file(files, VAE_MARKER)
    encoder = _find_support_file(files, ENCODER_MARKER)
    if vae is None or encoder is None:
        return []

    return [
        VideoModel(
            id=video_model_id(path),
            name=path.stem,
            diffusion=path,
            vae=vae,
            encoder=encoder,
            turbo=TURBO_MARKER in path.name.lower(),
            tae=_find_support_file(files, TAE_MARKER),
        )
        for path in files
        if not _is_support_file(path)
    ]


def resolve_video_model(model_id: str, settings: Settings) -> VideoModel | None:
    """Se resuelve contra la lista real: concatenar el id dejaría leer cualquier archivo."""
    if not model_id.startswith(VIDEO_MODEL_PREFIX):
        return None
    for model in list_video_models(settings):
        if model.id == model_id:
            return model
    return None


class SdcppVideoEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        if not self.settings.enable_sdcpp or not self.settings.sdcpp_binary_path.exists():
            return False
        return bool(list_video_models(self.settings))

    def build_command(self, request: VideoRequest, output_path: Path, model: VideoModel) -> list[str]:
        steps = request.steps if request.steps is not None else model.default_steps
        guidance = request.guidance if request.guidance is not None else model.default_guidance
        decoder = (["--tae", str(model.tae)] if model.tae is not None
                   else ["--vae", str(model.vae), VAE_TILING_FLAG])
        command = [
            str(self.settings.sdcpp_binary_path),
            "-M", "vid_gen",
            "--diffusion-model", str(model.diffusion),
            *decoder,
            "--t5xxl", str(model.encoder),
            "--backend", BACKEND_ASSIGNMENT,
            "-p", request.prompt,
            "--steps", str(steps),
            "--cfg-scale", str(guidance),
            "--sampling-method", SAMPLING_METHOD,
            "--flow-shift", FLOW_SHIFT,
            "-W", str(align_dimension(request.width)),
            "-H", str(align_dimension(request.height)),
            "--video-frames", str(normalize_frames(request.frames)),
            "--fps", str(request.fps),
            FLASH_ATTENTION_FLAG,
            "-o", str(output_path),
        ]
        if request.negative_prompt and not model.ignores_negative_prompt:
            command += ["-n", request.negative_prompt]
        if request.seed is not None:
            command += ["-s", str(request.seed)]
        if request.init_image is not None:
            command += ["-i", str(request.init_image)]
        return command

    async def run(self, request: VideoRequest, output_path: Path, model: VideoModel) -> Path:
        if not self.available():
            raise RuntimeError(
                "El lane de video Vulkan no está disponible: instalá el componente "
                "de generación de video o corré scripts/download-wan-video.ps1"
            )
        command = self.build_command(request, output_path, model)
        _stdout, stderr, returncode = await run_guarded_process(
            command, self.settings.subprocess_timeout
        )
        if returncode != 0:
            tail = stderr.decode(errors="replace")[-600:]
            raise RuntimeError(f"sd.cpp vid_gen falló con código {returncode}: {tail}")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("sd.cpp terminó con código 0 pero no dejó ningún video")
        return output_path
