from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

NEUTRAL_BIND_HOSTS = frozenset({"127.0.0.1", "0.0.0.0", "localhost"})

DEEPFILTER_MODE = "deepfilter"
RNNOISE_MODE = "rnnoise"
AUDIO_ENHANCE_MODES = frozenset({DEEPFILTER_MODE, RNNOISE_MODE})

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_against_project_root(path_str: str) -> Path:
    """Resolves a relative path against the project root, not the process CWD.

    Keeps absolute overrides (e.g. an absolute RUNTIME_DIR) untouched so the
    app still works regardless of the directory it was launched from.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


class ModelOption(TypedDict):
    key: str
    engine_name: str
    label: str
    category: str
    description: str
    scales: list[int]


class VideoProfile(TypedDict):
    key: str
    label: str
    category: str
    description: str
    model_key: str
    scale: int
    video_codec: str
    video_preset: str
    crf: int
    keep_audio: bool
    fps_multiplier: int


MODEL_CATALOG: list[ModelOption] = [
    {
        "key": "realesrgan-x4plus",
        "engine_name": "realesrgan-x4plus",
        "label": "RealESRGAN x4 Plus",
        "category": "general",
        "description": "Best general-purpose photo upscaling.",
        "scales": [4],
    },
    {
        "key": "realesrgan-x4plus-anime",
        "engine_name": "realesrgan-x4plus-anime",
        "label": "RealESRGAN x4 Plus Anime",
        "category": "anime",
        "description": "Still anime images, illustrations and line art.",
        "scales": [4],
    },
    {
        "key": "realesr-animevideov3-x2",
        "engine_name": "realesr-animevideov3-x2",
        "label": "RealESR AnimeVideo v3 x2",
        "category": "anime",
        "description": "Anime/video style model optimized for 2x.",
        "scales": [2],
    },
    {
        "key": "realesr-animevideov3-x3",
        "engine_name": "realesr-animevideov3-x3",
        "label": "RealESR AnimeVideo v3 x3",
        "category": "anime",
        "description": "Anime/video style model optimized for 3x.",
        "scales": [3],
    },
    {
        "key": "realesr-animevideov3-x4",
        "engine_name": "realesr-animevideov3-x4",
        "label": "RealESR AnimeVideo v3 x4",
        "category": "anime",
        "description": "Anime/video style model optimized for 4x.",
        "scales": [4],
    },
    {
        "key": "realesr-animevideov3",
        "engine_name": "realesr-animevideov3",
        "label": "RealESR AnimeVideo v3 (auto by scale)",
        "category": "anime",
        "description": "Convenience preset that maps automatically to x2/x3/x4.",
        "scales": [2, 3, 4],
    },
]

VIDEO_PROFILE_CATALOG: list[VideoProfile] = [
    {
        "key": "general-balanced-4x",
        "label": "General Balanced 4x",
        "category": "general",
        "description": "Good default for long videos when you want reasonable size and speed.",
        "model_key": "realesrgan-x4plus",
        "scale": 4,
        "video_codec": "libx264",
        "video_preset": "medium",
        "crf": 18,
        "keep_audio": True,
        "fps_multiplier": 1,
    },
    {
        "key": "general-hq-4x",
        "label": "General High Quality 4x",
        "category": "general",
        "description": "Higher quality archival-style output for non-anime footage.",
        "model_key": "realesrgan-x4plus",
        "scale": 4,
        "video_codec": "libx265",
        "video_preset": "slow",
        "crf": 17,
        "keep_audio": True,
        "fps_multiplier": 1,
    },
    {
        "key": "anime-balanced-2x",
        "label": "Anime Balanced 2x",
        "category": "anime",
        "description": "Best starting point for anime episodes and longer clips.",
        "model_key": "realesr-animevideov3-x2",
        "scale": 2,
        "video_codec": "libx264",
        "video_preset": "medium",
        "crf": 17,
        "keep_audio": True,
        "fps_multiplier": 1,
    },
    {
        "key": "anime-quality-3x",
        "label": "Anime Quality 3x",
        "category": "anime",
        "description": "Sharper upscale for anime scenes where detail matters.",
        "model_key": "realesr-animevideov3-x3",
        "scale": 3,
        "video_codec": "libx265",
        "video_preset": "slow",
        "crf": 16,
        "keep_audio": True,
        "fps_multiplier": 1,
    },
    {
        "key": "anime-max-detail-4x",
        "label": "Anime Max Detail 4x",
        "category": "anime",
        "description": "Heavy upscale for short anime clips when you want to push the GPU harder.",
        "model_key": "realesr-animevideov3-x4",
        "scale": 4,
        "video_codec": "libx265",
        "video_preset": "slow",
        "crf": 15,
        "keep_audio": True,
        "fps_multiplier": 1,
    },
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="Upflow", alias="APP_NAME")
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8090, alias="APP_PORT")
    web_title: str = Field(default="Upflow", alias="WEB_TITLE")

    max_upload_mb: int = Field(default=50, alias="MAX_UPLOAD_MB")
    max_video_upload_mb: int = Field(default=2048, alias="MAX_VIDEO_UPLOAD_MB")
    max_image_pixels: int = Field(default=120_000_000, alias="MAX_IMAGE_PIXELS")
    gpu_concurrency: int = Field(default=1, alias="GPU_CONCURRENCY")
    cpu_fallback_workers: int = Field(default=2, alias="CPU_FALLBACK_WORKERS")
    subprocess_timeout: float = Field(default=3600, alias="SUBPROCESS_TIMEOUT")
    ffmpeg_binary: str = Field(default="vendor/ffmpeg/bin/ffmpeg.exe", alias="FFMPEG_BINARY")
    ffprobe_binary: str = Field(default="vendor/ffmpeg/bin/ffprobe.exe", alias="FFPROBE_BINARY")
    ffmpeg_decode_threads: int = Field(default=12, alias="FFMPEG_DECODE_THREADS")
    ffmpeg_encode_threads: int = Field(default=24, alias="FFMPEG_ENCODE_THREADS")
    ffmpeg_x265_threads: int = Field(default=8, alias="FFMPEG_X265_THREADS")

    runtime_dir: str = Field(default="runtime", alias="RUNTIME_DIR")
    engine: str = Field(default="realesrgan-ncnn", alias="ENGINE")
    engine_binary: str = Field(default="vendor/realesrgan/realesrgan-ncnn-vulkan.exe", alias="ENGINE_BINARY")
    engine_models_dir: str = Field(default="vendor/realesrgan/models", alias="ENGINE_MODELS_DIR")
    default_model: str = Field(default="realesrgan-x4plus", alias="DEFAULT_MODEL")
    default_scale: int = Field(default=4, alias="DEFAULT_SCALE")
    allowed_scales: str = Field(default="2,3,4", alias="ALLOWED_SCALES")
    default_video_profile: str = Field(default="anime-balanced-2x", alias="DEFAULT_VIDEO_PROFILE")
    output_ttl_hours: int = Field(default=24, alias="OUTPUT_TTL_HOURS")
    allowed_origins: str | None = Field(default=None, alias="ALLOWED_ORIGINS")
    max_queue_size: int = Field(default=20, alias="MAX_QUEUE_SIZE")

    rife_binary: str = Field(default="vendor/rife/rife-ncnn-vulkan.exe", alias="RIFE_BINARY")
    rife_models_dir: str = Field(default="vendor/rife/models", alias="RIFE_MODELS_DIR")
    rife_model: str = Field(default="rife-v4.6", alias="RIFE_MODEL")
    enable_interpolation: bool = Field(default=False, alias="ENABLE_INTERPOLATION")
    allowed_fps_multipliers: str = Field(default="2,3,4", alias="ALLOWED_FPS_MULTIPLIERS")

    deepfilter_binary: str = Field(
        default="vendor/deepfilternet/deep-filter.exe", alias="DEEPFILTER_BINARY"
    )
    rnnoise_model: str = Field(
        default="vendor/deepfilternet/models/sh.rnnn", alias="RNNOISE_MODEL"
    )
    enable_audio_enhance: bool = Field(default=False, alias="ENABLE_AUDIO_ENHANCE")

    default_device: str = Field(default="dml:0", alias="DEFAULT_DEVICE")

    models_dir: str = Field(default="models", alias="MODELS_DIR")

    @model_validator(mode="after")
    def _apply_default_allowed_origins(self) -> "Settings":
        """Fills ALLOWED_ORIGINS from app_host/app_port when the caller left it unset.

        Only kicks in when no value came from init kwargs, env vars, or the
        dotenv file, so an explicit override (even an empty string) always
        wins verbatim.
        """
        if self.allowed_origins is None:
            self.allowed_origins = self._default_allowed_origins()
        return self

    def _default_allowed_origins(self) -> str:
        origins = [f"http://127.0.0.1:{self.app_port}", f"http://localhost:{self.app_port}"]
        if self.app_host not in NEUTRAL_BIND_HOSTS:
            origins.append(f"http://{self.app_host}:{self.app_port}")
        return ",".join(origins)

    @property
    def runtime_path(self) -> Path:
        return resolve_against_project_root(self.runtime_dir)

    @property
    def uploads_path(self) -> Path:
        return self.runtime_path / "uploads"

    @property
    def outputs_path(self) -> Path:
        return self.runtime_path / "outputs"

    @property
    def temp_path(self) -> Path:
        return self.runtime_path / "temp"

    @property
    def video_work_path(self) -> Path:
        return self.runtime_path / "video-work"

    @property
    def models_path(self) -> Path:
        # Derived from runtime_path (already resolved), not
        # resolve_against_project_root: MODELS_DIR must follow an overridden
        # RUNTIME_DIR the same way uploads/outputs/temp do. An absolute
        # MODELS_DIR override still wins outright (Path.__truediv__ discards
        # the left side when the right side is absolute).
        return self.runtime_path / self.models_dir

    @property
    def allowed_scale_values(self) -> list[int]:
        return [int(item.strip()) for item in self.allowed_scales.split(",") if item.strip()]

    @property
    def allowed_origin_values(self) -> frozenset[str]:
        return frozenset(item.strip() for item in self.allowed_origins.split(",") if item.strip())

    @property
    def ffmpeg_binary_path(self) -> Path:
        return resolve_against_project_root(self.ffmpeg_binary)

    @property
    def ffprobe_binary_path(self) -> Path:
        return resolve_against_project_root(self.ffprobe_binary)

    @property
    def engine_binary_path(self) -> Path:
        return resolve_against_project_root(self.engine_binary)

    @property
    def engine_models_path(self) -> Path:
        return resolve_against_project_root(self.engine_models_dir)

    @property
    def rife_binary_path(self) -> Path:
        return resolve_against_project_root(self.rife_binary)

    @property
    def rife_models_path(self) -> Path:
        return resolve_against_project_root(self.rife_models_dir)

    @property
    def allowed_fps_multiplier_values(self) -> list[int]:
        return [int(item.strip()) for item in self.allowed_fps_multipliers.split(",") if item.strip()]

    def interpolation_available(self) -> bool:
        # Capability-only (is RIFE installed?); callers check ENABLE_INTERPOLATION
        # separately so diagnostics can distinguish "not installed" from
        # "installed but disabled". Checks the configured model folder too,
        # guarding against partial installs.
        return (
            self.rife_binary_path.exists()
            and self.rife_models_path.exists()
            and (self.rife_models_path / self.rife_model).exists()
        )

    @property
    def deepfilter_binary_path(self) -> Path:
        return resolve_against_project_root(self.deepfilter_binary)

    @property
    def rnnoise_model_path(self) -> Path:
        return resolve_against_project_root(self.rnnoise_model)

    def _deepfilter_available(self) -> bool:
        return self.deepfilter_binary_path.exists()

    def _rnnoise_available(self) -> bool:
        return self.rnnoise_model_path.exists()

    def audio_enhance_available(self, mode: str) -> bool:
        # Capability-only per mode (is the binary/model installed?); callers
        # check ENABLE_AUDIO_ENHANCE separately, same split as
        # interpolation_available(). ffmpeg itself is not re-checked here for
        # "rnnoise" -- it is a hard startup dependency already.
        if mode not in AUDIO_ENHANCE_MODES:
            raise ValueError(f"Unknown audio enhance mode: {mode!r}")
        if mode == DEEPFILTER_MODE:
            return self._deepfilter_available()
        return self._rnnoise_available()

    @property
    def model_catalog(self) -> list[ModelOption]:
        return MODEL_CATALOG

    @property
    def model_keys(self) -> set[str]:
        return {item["key"] for item in self.model_catalog}

    @property
    def video_profile_catalog(self) -> list[VideoProfile]:
        return VIDEO_PROFILE_CATALOG

    @property
    def video_profile_keys(self) -> set[str]:
        return {item["key"] for item in self.video_profile_catalog}

    def get_model_option(self, model_name: str) -> ModelOption | None:
        return next((item for item in self.model_catalog if item["key"] == model_name), None)

    def get_video_profile(self, profile_key: str) -> VideoProfile | None:
        return next((item for item in self.video_profile_catalog if item["key"] == profile_key), None)

    def resolve_engine_model_name(self, model_name: str, scale: int) -> str:
        if model_name == "realesr-animevideov3":
            return f"realesr-animevideov3-x{scale}"
        option = self.get_model_option(model_name)
        if option:
            return option["engine_name"]
        return model_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
