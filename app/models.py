from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from app.services.generation_variants import Precision


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL_JOB_STATUSES = (JobStatus.completed, JobStatus.failed, JobStatus.cancelled)


@dataclass(slots=True)
class UpscaleJob:
    source_path: Path
    original_filename: str
    model_name: str
    scale: int
    output_format: str
    model_id: str | None = None
    device: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None


@dataclass(slots=True, frozen=True)
class UpdateStatus:
    current_version: str
    latest_version: str | None
    update_available: bool
    release_url: str | None
    published_at: str | None
    checked_at: datetime
    error: str | None


@dataclass(slots=True)
class VideoUpscaleJob:
    source_path: Path
    original_filename: str
    model_name: str
    scale: int
    output_container: str
    video_codec: str
    video_preset: str
    crf: int
    keep_audio: bool
    # Alto de salida pedido. Presente = el usuario pidio una RESOLUCION y no un
    # multiplicador: `scale` sigue siendo el escalado entero del modelo y el
    # redimensionado final lleva a la medida exacta. Ver target_resolution.py.
    target_height: int | None = None
    fps_multiplier: int = 1
    target_fps: str | None = None
    audio_enhance: str | None = None
    audio_restore: str | None = None
    # Which audio streams to keep in the output (Fase A Task 2). None means
    # "keep_audio decides alone" (existing behavior); a list selects specific
    # ffprobe stream indices, consumed by the pipeline in Task 3.
    audio_track_indices: list[int] | None = None
    # Copy subtitle streams into the output (Fase A Task 2). Forces an mkv
    # container upgrade when the requested container can't carry subtitles
    # losslessly -- see VideoJobManager._resolve_output_container.
    keep_subtitles: bool = False
    # Elegible output codec for the (enhanced/restored) primary audio track
    # (Fase C Task 8): "auto" (default) re-encodes to lossless FLAC only when
    # a restore actually ran (mirrors _resolve_output_container's mkv
    # upgrade), "flac" always wants lossless, "aac" always forces the
    # pre-existing lossy path regardless of restore. See
    # VideoUpscaler._prepare_processed_audio.
    audio_output_format: str = "auto"
    # Frame-interpolation engine (Task 4.2): "rife" (default, always) or
    # "gmfss" (opt-in, much higher quality, 10x or more slower -- even higher
    # on short clips due to model load overhead). Only consulted when
    # interpolation is actually requested (fps_multiplier>1 or target_fps set).
    interp_engine: str = "rife"
    model_id: str | None = None
    device: str | None = None
    # Upscale runtime override (SP11): None|auto -> Auto rule; ncnn|onnx force one.
    backend: str | None = None
    # Video encoder (SP12): "auto" (default) picks a hardware encoder AMF/NVENC/QSV
    # by the job's GPU and falls back to software; "software" forces libx264/libx265.
    # Default is "auto" because software x265 slow at 4x costs ~112 min/episode vs
    # ~16 min on the GPU -- the software default was the dominant wall-time cost.
    video_encoder: str = "auto"
    # ffprobe output captured during job validation, reused by the pipeline so the
    # same file isn't probed twice. In-memory only: the API response is built field
    # by field, so this never serializes (it holds the absolute source path).
    probe: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None


@dataclass(slots=True)
class AudioJob:
    source_path: Path
    original_filename: str
    denoise: str | None = None
    restore: str | None = None
    # Cadena de mejora de voz. Los pasos activos vienen como lista de ids del
    # step_catalog de voice_chain; el orden lo fija el catalogo, no el request,
    # porque el orden de la cadena tiene causalidad documentada.
    voice_steps: list[str] = field(default_factory=list)
    voice_delivery: str | None = None
    voice_presence_db: float | None = None
    # Acabado profesional: normaliza la sonoridad al estandar elegido (EBU R128).
    # None = no se toca el volumen, que es el comportamiento de siempre.
    master: str | None = None
    # Standalone-module output format (Fase C Task 9): "wav" (lossless, no
    # re-encode -- current is already PCM from decode/denoise/restore),
    # "flac" (lossless, ~50% smaller, default), "mp3" (lossy, smallest). See
    # AudioPipeline._write_output.
    output_format: str = "flac"
    device: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None


@dataclass(slots=True)
class TranscribeJob:
    """Audio a texto.

    A diferencia del resto de los jobs, el resultado es TEXTO y no un archivo de
    medios: `text` es la respuesta, y `output_path` es solo la copia en .txt para
    poder descargarla con el mismo patron que los demas.
    """

    source_path: Path
    original_filename: str
    model_id: str
    # None deja que el modelo detecte el idioma.
    language: str | None = None
    # Segmentos con tiempo. `text` es su concatenacion; los segmentos son lo que
    # permite escribir un .srt, que sin tiempos no existiria.
    segments: list[Any] = field(default_factory=list)
    # "text" entrega solo la transcripcion; "video" ademas devuelve el archivo
    # original con la pista de subtitulos adentro. El modo decide si el fuente
    # se conserva hasta el final o se borra apenas termina de transcribirse.
    output_mode: str = "text"
    # Solo en modo doblaje: a que idioma hablar. Sin esto no hay nada que doblar.
    target_language: str | None = None
    # Cuantas lineas del doblaje no entraron en su hueco ni al maximo de
    # velocidad. Se cuenta para avisarlo en vez de entregar un doblaje corrido.
    dub_overflow_segments: int | None = None
    subtitled_video_path: Path | None = None
    device: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress_pct: float | None = None
    text: str | None = None
    output_path: Path | None = None
    error: str | None = None
    owner_id: str | None = None


@dataclass(slots=True)
class Shape3dJob:
    """Texto a malla 3D.

    El resultado NO es solo el archivo: viaja con el veredicto del banco. Una
    malla generada que no cierra no es una pieza, y entregarla sin decirlo seria
    mandar a imprimir con confianza prestada.
    """

    prompt: str
    printer: str = "ender-3"
    # "mesh" = Shap-E, que da FORMA sin cotas y por eso se escala con `target_mm`.
    # "cad"  = OpenSCAD escrito por un modelo, que da COTAS y por eso se verifica
    #          contra `expected_size` en vez de escalarse.
    source: str = "mesh"
    # Solo en "mesh": la malla se escala para que su lado mas largo mida esto.
    target_mm: float | None = None
    # Solo en "cad": lo que la pieza TIENE que medir. Si no coincide, el error
    # vuelve al modelo en vez de entregar algo que no entra.
    expected_size: tuple[float, float, float] | None = None
    # Solo en "cad": el codigo, que es la pieza EDITABLE. Entregar solo el STL
    # seria entregar algo que no se puede ajustar, justo lo contrario de tener cotas.
    code: str | None = None
    retries: int = 0
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    output_path: Path | None = None
    can_print: bool | None = None
    size_mm: tuple[float, float, float] | None = None
    triangle_count: int | None = None
    blockers: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    error: str | None = None
    owner_id: str | None = None


@dataclass(slots=True)
class DownloadJob:
    """Traer un video o audio de una URL.

    A diferencia del resto, la entrada no es un archivo sino una URL, y la SALIDA es lo
    que despues puede entrar al pipeline de mejora: por eso `output_paths` es una lista
    (una playlist produce varios) y aterriza en el directorio de uploads.
    """

    url: str
    max_height: int = 1080
    audio_only: bool = False
    audio_format: str = "mp3"
    audio_bitrate_kbps: int | None = None
    video_container: str = "mp4"
    include_playlist: bool = False
    playlist_limit: int = 10
    subtitle_languages: list[str] = field(default_factory=list)
    # Lo que el probe encontro antes de descargar: titulo, duracion, sitio.
    media_title: str | None = None
    media_uploader: str | None = None
    extractor: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress_pct: float | None = None
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    output_paths: list[Path] = field(default_factory=list)
    error: str | None = None
    owner_id: str | None = None


@dataclass(slots=True)
class GenerationJob:
    prompt: str
    model_id: str
    negative_prompt: str | None = None
    steps: int = 25
    guidance: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None
    # Scheduler alternativo ("lcm" | "euler_a" | "euler_trailing"); None = el
    # del repo. Lo puede fijar la API o el anclaje por clase de velocidad.
    scheduler: str | None = None
    device: str | None = None
    # Imagen de partida ya staged en disco. Presente = imagen a imagen.
    init_image_path: Path | None = None
    # Cuanto se aparta del original: 0 lo devuelve casi igual, 1 lo ignora.
    strength: float = 0.6
    # Máscara de inpainting (blanco=editar, negro=conservar); requiere init.
    mask_image_path: Path | None = None
    auto_upscale: bool = False
    upscale_model_name: str | None = None
    upscale_scale: int | None = None
    upscale_model_id: str | None = None
    # Solo para el lane de video: cuántos cuadros y a qué ritmo. En un job de
    # imagen quedan en None y nada los mira.
    frames: int | None = None
    fps: int | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None


@dataclass(slots=True)
class ConversionJob:
    repo_id: str
    precision: Precision = "fp16"
    checkpoint_path: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class InstallJob:
    id: str
    repo_id: str
    precision: Precision = "fp16"
    checkpoint_path: str | None = None
    status: Any = None
    progress_pct: float | None = None
    model_id: str | None = None
    error: str | None = None
    conversion_id: str | None = None
