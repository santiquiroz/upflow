from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _logical_cpus() -> int:
    return os.cpu_count() or 8


def _default_onnx_save_threads() -> int:
    # The 4x-output PNG encode (~510ms/frame @ 5120x2880) is the onnx video
    # bottleneck, not GPU infer (~68ms). It is CPU/zlib-bound and embarrassingly
    # parallel, so saver count scales with cores to keep infer the ceiling
    # (measured 7900X3D: 4 threads -> 5.5fps, 10 -> 11.5fps, 14 -> 12.6fps).
    return max(4, min(12, _logical_cpus()))


def _default_onnx_load_threads() -> int:
    # Input PNG decode (720p, ~30ms) is cheap; a few threads saturate it.
    return max(2, min(4, _logical_cpus() // 4))


def _default_ffmpeg_decode_threads() -> int:
    return max(2, min(12, _logical_cpus()))


def _default_ffmpeg_encode_threads() -> int:
    # Capped, NOT `-threads 0`: 0 lets ffmpeg grab every core, which oversubscribes
    # the box when several jobs encode concurrently.
    return max(2, min(24, _logical_cpus()))

NEUTRAL_BIND_HOSTS = frozenset({"127.0.0.1", "0.0.0.0", "localhost"})

DEEPFILTER_MODE = "deepfilter"
RNNOISE_MODE = "rnnoise"
AUDIO_ENHANCE_MODES = frozenset({DEEPFILTER_MODE, RNNOISE_MODE})

APOLLO_MODE = "apollo"
AUDIOSR_MODE = "audiosr"
AUDIO_RESTORE_MODES = frozenset({APOLLO_MODE, AUDIOSR_MODE})

# Standalone-module output format selector (Fase C Task 9). "flac" is the
# default (lossless, ~50% smaller than wav); see AudioPipeline._write_output.
AUDIO_OUTPUT_FORMATS = frozenset({"wav", "flac", "mp3"})

# Frame-interpolation engine selector (Task 4.2). `rife` is ALWAYS the
# default -- GMFSS (much higher quality, 10x or more slower -- see
# gmfss_engine.py; short clips measure even higher due to cold-start model
# load overhead) is strictly opt-in per job, same split as
# AUDIO_RESTORE_MODES's Apollo/AudioSR.
RIFE_ENGINE = "rife"
GMFSS_ENGINE = "gmfss"
INTERP_ENGINES = frozenset({RIFE_ENGINE, GMFSS_ENGINE})

# Upscale runtime selector (SP11). `auto` picks onnx vs ncnn per the rule in
# app/services/backend_registry.py; `ncnn`/`onnx` force a specific runtime.
# The selector changes the RUNTIME, never the model the user picked.
UPSCALE_BACKEND_AUTO = "auto"
UPSCALE_BACKEND_NCNN = "ncnn"
UPSCALE_BACKEND_ONNX = "onnx"
UPSCALE_BACKENDS = frozenset({UPSCALE_BACKEND_AUTO, UPSCALE_BACKEND_NCNN, UPSCALE_BACKEND_ONNX})

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

    max_upload_mb: int = Field(default=50, alias="MAX_UPLOAD_MB")
    max_video_upload_mb: int = Field(default=2048, alias="MAX_VIDEO_UPLOAD_MB")
    max_image_pixels: int = Field(default=120_000_000, alias="MAX_IMAGE_PIXELS")
    # Per-device concurrency (DeviceSemaphores): each physical device_id
    # (dml:0, dml:1, cpu...) gets its own semaphore, so jobs on distinct
    # devices run in parallel instead of serializing behind one shared gate.
    per_device_gpu_concurrency: int = Field(default=1, alias="PER_DEVICE_GPU_CONCURRENCY")
    cpu_concurrency: int = Field(default=2, alias="CPU_CONCURRENCY")
    # Workers per manager (JobManager/VideoJobManager): must exceed the
    # number of devices expected to run in parallel, or idle device
    # semaphores never get a worker to pull a job into them.
    max_concurrent_jobs: int = Field(default=4, alias="MAX_CONCURRENT_JOBS")
    # Threads load:proc:save de Real-ESRGAN NCNN en el upscale de video. El proc
    # (GPU) por defecto era 2 -> subutilizaba la GPU. Medido en RX 7800 XT (720p 4x):
    # 2:2:2 -> 2.2 fps ; 2:24:12 -> 5.3 fps (~2.4x, cerca del techo de NCNN Vulkan).
    # NCNN mantiene el tile chico (auto) asi que muchos proc-threads no revientan VRAM;
    # en una GPU debil bajar este valor si va lento/inestable.
    ncnn_upscale_threads: str = Field(default="2:24:12", alias="NCNN_UPSCALE_THREADS")
    subprocess_timeout: float = Field(default=86400, alias="SUBPROCESS_TIMEOUT")
    frame_stall_timeout_seconds: float = Field(default=900, alias="FRAME_STALL_TIMEOUT_SECONDS")
    ffmpeg_binary: str = Field(default="vendor/ffmpeg/bin/ffmpeg.exe", alias="FFMPEG_BINARY")
    ffprobe_binary: str = Field(default="vendor/ffmpeg/bin/ffprobe.exe", alias="FFPROBE_BINARY")
    ffmpeg_decode_threads: int = Field(
        default_factory=_default_ffmpeg_decode_threads, alias="FFMPEG_DECODE_THREADS"
    )
    ffmpeg_encode_threads: int = Field(
        default_factory=_default_ffmpeg_encode_threads, alias="FFMPEG_ENCODE_THREADS"
    )
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
    rife_model: str = Field(default="rife-v4.25", alias="RIFE_MODEL")
    # Threads load:proc:save del binario RIFE. "auto" escala con los nucleos
    # (el default upstream 1:2:2 deja la GPU esperando el decode PNG a 4K).
    rife_threads: str = Field(default="auto", alias="RIFE_THREADS")
    # Modo UHD de RIFE (-u, flujo a media resolucion): auto = solo si los
    # frames de entrada son >=1440p; on/off lo fuerzan.
    rife_uhd_mode: str = Field(default="auto", alias="RIFE_UHD_MODE")
    enable_interpolation: bool = Field(default=False, alias="ENABLE_INTERPOLATION")
    allowed_fps_multipliers: str = Field(default="2,3,4", alias="ALLOWED_FPS_MULTIPLIERS")

    deepfilter_binary: str = Field(
        default="vendor/deepfilternet/deep-filter.exe", alias="DEEPFILTER_BINARY"
    )
    rnnoise_model: str = Field(
        default="vendor/deepfilternet/models/sh.rnnn", alias="RNNOISE_MODEL"
    )
    enable_audio_enhance: bool = Field(default=False, alias="ENABLE_AUDIO_ENHANCE")

    # Apollo audio restoration (experimental, ONNX/DirectML). Off by default:
    # it reconstructs codec-lost high band and is gated behind its own flag,
    # unlike denoise (production-ready, install-gated only).
    enable_audio_restore: bool = Field(default=False, alias="ENABLE_AUDIO_RESTORE")
    apollo_restore_model: str = Field(default="vendor/apollo/apollo.onnx", alias="APOLLO_RESTORE_MODEL")
    # DirectML breaks on long tensors: Apollo runs in chunks <=3s with a
    # 0.5s Hann overlap-add, so this bounds the per-inference tensor length.
    # 1.0s mantiene cada inferencia DirectML por debajo del limite TDR de Windows
    # (~2s antes de que el driver resetee la GPU). En la RX 7800 XT: 3.0s -> ~2.3s/chunk
    # (TDR, reset repetido); 1.0s -> ~0.8s/chunk (seguro). En GPUs mas debiles, si igual
    # aparece "GPU timeout", usar device=cpu para el restore (sin TDR, correcto a cualquier largo).
    audio_restore_chunk_seconds: float = Field(default=1.0, alias="AUDIO_RESTORE_CHUNK_SECONDS")
    # CPU no tiene el limite TDR de DirectML -> chunk grande = mas contexto para el
    # modelo y menos bordes = mejor calidad (menos "borroso" en gritos/notas altas).
    audio_restore_cpu_chunk_seconds: float = Field(default=30.0, alias="AUDIO_RESTORE_CPU_CHUNK_SECONDS")
    # Pausa (s) entre inferencias GPU del restore: le devuelve la GPU al escritorio
    # de Windows entre chunks para que el PC no se laguee tanto mientras procesa.
    # Solo aplica a device=dml:N (en CPU no hay contencion de GPU). 0 = sin respiro.
    audio_restore_gpu_throttle_seconds: float = Field(default=0.15, alias="AUDIO_RESTORE_GPU_THROTTLE_SECONDS")
    # AudioSR (segundo motor de restore, difusion latente 48kHz, port propio
    # santiquiroz/port-audiosr-onnx). Mucho mas pesado que Apollo (RTF ~1.9 en
    # una 7800 XT a 50 pasos) pero es super-resolucion general de audio, no
    # solo band-restore. Flag propio: se puede habilitar sin Apollo y viceversa.
    enable_audiosr: bool = Field(default=False, alias="ENABLE_AUDIOSR")
    audiosr_model_dir: str = Field(default="vendor/audiosr", alias="AUDIOSR_MODEL_DIR")
    # 50 pasos = calidad del paper; 25 corta el tiempo a la mitad con perdida
    # leve. Cada paso son 2 pasadas del UNet 258M (CFG).
    audiosr_ddim_steps: int = Field(default=50, alias="AUDIOSR_DDIM_STEPS")
    # Mismo rol que audio_restore_gpu_throttle_seconds pero por paso DDIM
    # (~90ms de UNet x2 por paso): respiro corto para el compositor de Windows.
    audiosr_gpu_throttle_seconds: float = Field(default=0.01, alias="AUDIOSR_GPU_THROTTLE_SECONDS")
    max_audio_upload_mb: int = Field(default=200, alias="MAX_AUDIO_UPLOAD_MB")

    # GMFSS (second interpolation engine, max-quality anime frame interpolation,
    # own port santiquiroz/port-gmfss-onnx). 10x or more slower than RIFE by
    # design -- a short-clip smoke test measured closer to 20x due to
    # cold-start model load overhead, so treat 10x as a floor, not a stable
    # figure -- opt-in via its own flag, same split as enable_audiosr next to
    # Apollo.
    enable_gmfss: bool = Field(default=False, alias="ENABLE_GMFSS")
    gmfss_model_dir: str = Field(default="vendor/gmfss", alias="GMFSS_MODEL_DIR")

    # Fase 2 (Tasks 7-8): fusiona GMFSS interpolar + el escalador ONNX in-process
    # en una sola pasada, sin escribir el PNG intermedio a resolucion fuente.
    # Task 9 lo midio en hardware real (RX 7800 XT, 1080p->4x/8K,
    # general-balanced-4x): ~1.7x MAS LENTO que las dos pasadas, no mas rapido
    # (ver README "Benchmark real: fusion interpolar+escalar"). Causa probable:
    # el loop fusionado es un generador secuencial de un solo hilo, sin el
    # overlap load/compute/save en threads que SI tienen las dos pasadas. Opt-in
    # y apagado por defecto hasta entender/arreglar la regresion -- el codigo
    # fusionado sigue completo, probado y disponible, solo no se activa solo.
    enable_interp_upscale_fusion: bool = Field(default=False, alias="ENABLE_INTERP_UPSCALE_FUSION")

    default_device: str = Field(default="dml:0", alias="DEFAULT_DEVICE")
    # When on and a job request doesn't pin a device, routes.py hands the job
    # the "auto" sentinel instead of DEFAULT_DEVICE -- see
    # app/services/device_router.py for the actual routing decision. Off by
    # default: existing per-job device selection behavior is unchanged.
    enable_auto_route: bool = Field(default=False, alias="ENABLE_AUTO_ROUTE")

    models_dir: str = Field(default="models", alias="MODELS_DIR")

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    max_model_download_mb: int = Field(default=2048, alias="MAX_MODEL_DOWNLOAD_MB")
    max_generation_model_download_mb: int = Field(default=8192, alias="MAX_GENERATION_MODEL_DOWNLOAD_MB")

    onnx_tile_size: int = Field(default=256, alias="ONNX_TILE_SIZE")

    # --- Optimized ONNX video backend + runtime selector (SP11) ---
    # Runtime that upscales video frames for builtin Real-ESRGAN models:
    # `auto` (default) picks onnx when the model has a vendored ONNX export and
    # a capable GPU EP is present, else ncnn; `ncnn`/`onnx` force one runtime.
    upscale_backend: str = Field(default=UPSCALE_BACKEND_AUTO, alias="UPSCALE_BACKEND")
    # Where the uint8-in/out ONNX exports of the builtin models live
    # (scripts/download-realesrgan-onnx.ps1 populates it). Vendored + gitignored.
    builtin_onnx_dir: str = Field(default="vendor/realesrgan-onnx", alias="BUILTIN_ONNX_DIR")
    # Whole-frame inference (no tiling) is the fast path; tiling is only a
    # fallback for frames whose INPUT pixel count exceeds this (huge frames /
    # low-VRAM GPUs). 0 disables tiling entirely (always whole-frame). Default
    # ~= 3840x2160 input.
    onnx_whole_frame_max_pixels: int = Field(default=8_294_400, alias="ONNX_WHOLE_FRAME_MAX_PIXELS")
    # PNG compression level (0-9) for intermediate upscaled frames written by
    # the onnx video pipeline. Low = fast; frames are re-encoded by ffmpeg
    # anyway, so speed beats size here (OpenCV default is 1).
    onnx_video_png_compression: int = Field(default=1, alias="ONNX_VIDEO_PNG_COMPRESSION")
    # Threads that overlap frame load (N+1) and save (N-1) with GPU infer (N)
    # in the onnx video pipeline. GPU inference itself stays single-flight.
    # Defaults scale with cores because PNG save (not infer) is the bottleneck;
    # override to dial down on a weak/oversubscribed CPU.
    onnx_video_load_threads: int = Field(default_factory=_default_onnx_load_threads, alias="ONNX_VIDEO_LOAD_THREADS")
    onnx_video_save_threads: int = Field(default_factory=_default_onnx_save_threads, alias="ONNX_VIDEO_SAVE_THREADS")
    # Techo de RAM del pipeline ONNX: acota cuantos frames 4x de salida (44MB c/u
    # @5120x2880) viven a la vez en la cola de guardado. Sin esto la cola escala
    # con save-threads (12*2=24 frames -> ~1GB solo en cola) sin relacion con la
    # RAM. maxsize se deriva de este presupuesto y el tamano real del frame, con
    # piso = save-threads para no matar el throughput.
    onnx_video_max_pipeline_mb: int = Field(default=1024, alias="ONNX_VIDEO_MAX_PIPELINE_MB")
    # Usar el export fp16 del modelo builtin cuando corre en GPU y el archivo existe.
    # El cuerpo de la red trabaja a resolucion de ENTRADA, asi que domina el tiempo
    # en cualquier escala; medido en 7800 XT (1080p->4x): 154.9 -> 116.5 ms/frame
    # (1.33x) con diferencia maxima de 3/255 por pixel. En CPU se ignora (fp16
    # emulado = mas lento). Poner en False para forzar fp32 en todos lados.
    onnx_prefer_fp16: bool = Field(default=True, alias="ONNX_PREFER_FP16")
    # Raw-pipe: cuando la interpolacion esta OFF y el modelo builtin corre por ONNX,
    # pipear los frames escalados crudos (rgb24) directo a ffmpeg stdin en vez de
    # escribir PNG a disco y re-encodear. Borra el round-trip PNG y solapa upscale+
    # encode. Cae al camino con PNG ante cualquier fallo. False = siempre PNG.
    enable_raw_pipe: bool = Field(default=True, alias="ENABLE_RAW_PIPE")
    # El raw-pipe solo gana cuando el encode PNG del frame de salida es caro. Por
    # debajo de este tamano de salida el PNG es barato y el overhead del subproceso
    # no compensa (medido: 256x256 mas lento, 7680x4320 1.24x mas rapido). Default
    # ~= 2560x1440 (720p->2x y para arriba usan raw-pipe; salidas chicas usan PNG).
    raw_pipe_min_output_pixels: int = Field(default=3_686_400, alias="RAW_PIPE_MIN_OUTPUT_PIXELS")

    update_repo: str = Field(default="santiquiroz/upflow", alias="UPDATE_REPO")
    # Package whose installed metadata gives the running version to compare
    # against the latest release. Reuse in another project = change UPDATE_REPO
    # + UPDATE_PACKAGE_NAME (no code edit).
    update_package_name: str = Field(default="upflow", alias="UPDATE_PACKAGE_NAME")
    update_check_enabled: bool = Field(default=True, alias="UPDATE_CHECK_ENABLED")
    update_check_ttl_seconds: int = Field(default=3600, alias="UPDATE_CHECK_TTL_SECONDS")
    # A failed check with no prior good result is cached only this long, so a
    # startup-time network blip retries in minutes instead of after the full
    # TTL. A successful result (even "up to date") uses the full TTL above.
    update_error_retry_seconds: int = Field(default=300, alias="UPDATE_ERROR_RETRY_SECONDS")
    update_api_timeout_seconds: float = Field(default=5.0, alias="UPDATE_API_TIMEOUT_SECONDS")
    # Elevated (UAC) PowerShell fix scripts launched by CapabilityProbe.apply_fix
    # (HAGS registry write, disk write-cache registry write, Defender exclusion
    # add) -- bounds how long the backend waits on the elevated child process.
    capability_fix_timeout_seconds: float = Field(default=120.0, alias="CAPABILITY_FIX_TIMEOUT_SECONDS")

    @field_validator("per_device_gpu_concurrency", "cpu_concurrency", "max_concurrent_jobs")
    @classmethod
    def _validate_concurrency_at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Concurrency settings must be at least 1")
        return value

    @field_validator("update_check_ttl_seconds", "update_error_retry_seconds")
    @classmethod
    def _validate_update_ttl_at_least_one(cls, value: int) -> int:
        # A non-positive TTL would disable caching and hammer GitHub's 60 req/h
        # anonymous limit -- exactly what the cache exists to prevent.
        if value < 1:
            raise ValueError("Update cache TTL settings must be at least 1 second")
        return value

    @field_validator("update_api_timeout_seconds")
    @classmethod
    def _validate_update_timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("UPDATE_API_TIMEOUT_SECONDS must be greater than 0")
        return value

    @field_validator("capability_fix_timeout_seconds")
    @classmethod
    def _validate_capability_fix_timeout_minimum(cls, value: float) -> float:
        # capability_probe._run_elevated derives an inner PowerShell-side
        # WaitForExit deadline from this value (5s margin, 1s floor) that must
        # always fire before this outer run_guarded_process timeout, or the
        # elevated process can be orphaned on the user's desktop instead of
        # being killed. 10s is comfortably above margin + floor (6s).
        if value < 10.0:
            raise ValueError("CAPABILITY_FIX_TIMEOUT_SECONDS must be at least 10 seconds")
        return value

    @field_validator("upscale_backend")
    @classmethod
    def _validate_upscale_backend(cls, value: str) -> str:
        if value not in UPSCALE_BACKENDS:
            raise ValueError(f"UPSCALE_BACKEND must be one of {sorted(UPSCALE_BACKENDS)}")
        return value

    @field_validator("audio_restore_chunk_seconds")
    @classmethod
    def _validate_chunk_seconds_positive(cls, value: float) -> float:
        # A non-positive chunk length makes the Apollo overlap-add hop <= 0,
        # which would infinite-loop the chunker.
        if value <= 0:
            raise ValueError("AUDIO_RESTORE_CHUNK_SECONDS must be greater than 0")
        return value

    @field_validator("rife_threads")
    @classmethod
    def _validate_rife_threads(cls, value: str) -> str:
        if value == "auto":
            return value
        parts = value.split(":")
        if len(parts) != 3 or not all(part.isdigit() and int(part) >= 1 for part in parts):
            raise ValueError('RIFE_THREADS must be "auto" or "load:proc:save" positive integers')
        return value

    @field_validator("rife_uhd_mode")
    @classmethod
    def _validate_rife_uhd_mode(cls, value: str) -> str:
        if value not in ("auto", "on", "off"):
            raise ValueError("RIFE_UHD_MODE must be one of: auto, on, off")
        return value

    @field_validator("audiosr_ddim_steps")
    @classmethod
    def _validate_audiosr_ddim_steps(cls, value: int) -> int:
        # <1 no genera nada; >500 no aporta calidad y multiplica el costo x2/paso.
        if not 1 <= value <= 500:
            raise ValueError("AUDIOSR_DDIM_STEPS must be between 1 and 500")
        return value

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
    def builtin_onnx_path(self) -> Path:
        return resolve_against_project_root(self.builtin_onnx_dir)

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
    def apollo_restore_model_path(self) -> Path:
        return resolve_against_project_root(self.apollo_restore_model)

    def audio_restore_available(self) -> bool:
        # Unlike audio_enhance_available (capability-only), restore folds the
        # enable flag in: it is experimental, so "available" means both
        # explicitly enabled AND the model file present. Never raises -- a
        # missing model just yields False, so the app never breaks.
        return self.enable_audio_restore and self.apollo_restore_model_path.exists()

    @property
    def audiosr_model_dir_path(self) -> Path:
        return resolve_against_project_root(self.audiosr_model_dir)

    def audiosr_available(self) -> bool:
        if not self.enable_audiosr:
            return False
        from app.services.engines.audiosr.assets import AudioSrAssets

        return AudioSrAssets.is_complete(self.audiosr_model_dir_path)

    def audio_restore_mode_available(self, mode: str) -> bool:
        if mode == APOLLO_MODE:
            return self.audio_restore_available()
        if mode == AUDIOSR_MODE:
            return self.audiosr_available()
        return False

    @property
    def gmfss_model_dir_path(self) -> Path:
        return resolve_against_project_root(self.gmfss_model_dir)

    def gmfss_available(self) -> bool:
        if not self.enable_gmfss:
            return False
        from app.services.engines.gmfss.assets import GmfssAssets

        return GmfssAssets.is_complete(self.gmfss_model_dir_path)

    def interp_engine_available(self, engine: str) -> bool:
        if engine == RIFE_ENGINE:
            return self.interpolation_available()
        if engine == GMFSS_ENGINE:
            return self.gmfss_available()
        return False

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
