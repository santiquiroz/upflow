from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from app.api.auth_deps import current_user_from_request, get_current_user, require
from app.config import (
    AUDIO_ENHANCE_MODES,
    AUDIO_OUTPUT_FORMATS,
    AUDIO_RESTORE_MODES,
    INTERP_ENGINES,
    RIFE_ENGINE,
    Settings,
    VideoProfile,
    get_settings,
)
from app.exceptions import ModelNotFoundError, ModelProtectedError, QueueFullError, QuotaExceededError
from app.models import (
    DownloadJob,
    AudioJob,
    GenerationJob,
    JobStatus,
    TranscribeJob,
    UpdateStatus,
    UpscaleJob,
    VideoUpscaleJob,
    Shape3dJob,
)
from app.schemas import (
    InstallVulkanModelRequest,
    VulkanInstallStatusResponse,
    CreateDownloadJobRequest,
    DownloadJobResponse,
    DownloadJobsListResponse,
    MediaProbeResponse,
    AnalyzeVideoResponse,
    AudioCapabilitiesResponse,
    AudioJobResponse,
    AudioJobsListResponse,
    AudioStemDownloadResponse,
    AudioTrackResponse,
    CapabilityDomainResponse,
    CapabilityResponse,
    CleanupStepResponse,
    CapabilityTreeResponse,
    ConversionStatusResponse,
    CreateConversionResponse,
    CreateGenerationJobRequest,
    CreateInstallResponse,
    CreateJobResponse,
    DeviceInfoResponse,
    DevicesResponse,
    EditableSettingStatusResponse,
    EditableSettingsResponse,
    EngineInfoResponse,
    GenerationCapabilitiesResponse,
    GenerationJobResponse,
    GenerationJobsListResponse,
    GenerationModelSummary,
    LossyQualityResponse,
    MasteringPresetResponse,
    SeparationModelResponse,
    SeparationStemResponse,
    PromptPresetResponse,
    PromptPresetsResponse,
    CreateSavedPromptRequest,
    SavedPromptResponse,
    SavedPromptsResponse,
    SynthesizeSpeechRequest,
    TranslationPairResponse,
    TranslationPairsResponse,
    TtsCapabilitiesResponse,
    GeneratePartRequest,
    Shape3dJobRequest,
    Shape3dJobResponse,
    Shape3dJobsListResponse,
    SizeEstimateRequest,
    SizeEstimateResponse,
    GeneratedPartResponse,
    MeshRepairResponse,
    PartKindResponse,
    PartKindsResponse,
    PartParamResponse,
    PrintCheckResponse,
    PrinterResponse,
    PrintersResponse,
    VoiceConversionCapabilitiesResponse,
    RealtimeCapabilitiesResponse,
    RealtimePresetResponse,
    RealtimeStartedResponse,
    StartRealtimeRequest,
    VideoGenerationCapabilitiesResponse,
    VideoModelSummary,
    HealthResponse,
    HfModelSearchResultResponse,
    InstallModelRequest,
    InitImageResponse,
    InstallStatusResponse,
    JobResponse,
    JobsListResponse,
    ModelResponse,
    ModelSearchResponse,
    ModelsResponse,
    PreflightResponse,
    ProvisionJobResponse,
    SubtitleTrackResponse,
    SupportedModelResponse,
    TranscribeJobResponse,
    TranscribeJobsListResponse,
    UpdateCheckResponse,
    UpdateSettingRequest,
    UpdateSettingResponse,
    UpscalerPreflightResponse,
    VideoCapabilitiesResponse,
    VideoJobResponse,
    VideoJobsListResponse,
    VideoProfileResponse,
    VoiceCatalogResponse,
)
from app.services.asr_installer import AsrModelInstaller
from app.services.audio_conversion import (
    DEFAULT_LOSSY_QUALITY,
    LOSSY_OUTPUT_FORMATS,
    LOSSY_QUALITY_BITRATES,
)
from app.services.audio_job_manager import AudioJobManager
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import Permission
from app.services.capabilities import (
    ResolvedCapability,
    group_by_domain,
    resolve_capabilities,
    resolve_one,
)
from app.services.compat_strategy import CompatStrategy, InstallOptions, strategy_for
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.generation_onnx import generation_dependencies_available
from app.services.generation_converter import GenerationModelConverter
from app.services.generation_installer import (
    CheckpointNotFoundError,
    GenerationModelInstaller,
)
from app.services.generation_compat import classify
from app.services.engines.sdcpp_video import VIDEO_MODEL_PREFIX
from app.services.generation_job_manager import (
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_FRAMES,
    MAX_VIDEO_FRAMES,
    GenerationJobManager,
)
from app.services.generation_preflight import preflight
from app.services.generation_variants import available_precisions_from_names
from app.services.hf_client import (
    ASR_SEARCH_TASK_TAGS,
    GENERATION_SEARCH_TASK_TAGS,
    HfClient,
)
from app.services.job_manager import JobManager
from app.services.media_tools import MediaTools
from app.services.model_installer import ModelInstaller
from app.services.model_preflight import preflight_upscaler
from app.services.missing_pack import missing_pack_message
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.pack_provisioner import PackProvisioner, ProvisionJob, UnknownPackError
from app.services.settings_service import editable_settings_status, update_setting
from app.services.storage import StorageService
from app.services.download_job_manager import (
    DownloadJobManager,
    describe_failure,
    validate_url,
)
from fetchflow import engine as fetch_engine
import io

import soundfile

from app.services.engines.tts_kokoro import (
    SAMPLE_RATE as TTS_SAMPLE_RATE,
    KokoroTtsEngine,
    TtsUnavailable,
    available_voices,
)
from app.services.phonemize import text_to_phonemes
import tempfile

import numpy

from app.services.engines.voice_convert import (
    MAX_SECONDS as VOICE_MAX_SECONDS,
    SAMPLE_RATE as VOICE_SAMPLE_RATE,
    VoiceConversionEngine,
    VoiceConversionUnavailable,
)
from app.services.media_decode import build_decode_to_wav_command, needs_decoding
from app.services.process_runner import run_guarded_process
from app.services.prompt_presets import PROMPT_PRESETS
from app.services.saved_prompts import SavedPromptStore
from app.services.subtitles import SUBTITLE_FORMATS, TranscriptSegment, render_segments
from app.services.mesh_fit import PRINTER_BEDS
from app.services.mesh_repair import repair_mesh
from app.services.engines.shape3d import Shape3dUnavailable
from app.services.parametric_parts import PartError
from app.services.shape3d_job_manager import Shape3dJobManager
from app.services.size_estimate import SizeEstimateUnavailable, estimate_longest_mm
from app.services.part_catalog import PART_KINDS, build_part
from app.services.stl_reader import StlUnreadable, read_stl
from app.services.stl_writer import write_stl
from app.services.print_check import PrintCheckUnavailable, check_stl_for_printing
from app.services.vendor_paths import kokoro_dir, translation_dir
from app.services.translation_catalog import INSTALLABLE_PAIRS
from app.services.translate import (
    TranslationEngine,
    TranslationUnavailable,
    parse_pair,
)
from app.services.transcribe_job_manager import TranscribeJobManager
from app.services.stream_analysis import parse_audio_tracks, parse_subtitle_tracks
from app.services.update_service import UpdateService
from app.services.video_job_manager import VideoJobManager
from app.services.voice_chain import delivery_choices, step_catalog

router = APIRouter(prefix="/api/v1", tags=["api"])

logger = logging.getLogger(__name__)

FORBIDDEN_FILENAME_CHARS = frozenset(':<>"|?*')
WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(10)}
    | {f"LPT{i}" for i in range(10)}
)


def _strip_forbidden_chars(name: str) -> str:
    return "".join(char for char in name if char not in FORBIDDEN_FILENAME_CHARS)


def _escape_reserved_stem(name: str) -> str:
    stem = Path(name).stem.upper()
    if stem in WINDOWS_RESERVED_STEMS:
        return f"_{name}"
    return name


def _original_filename_from_staged_upload(uploads_path: Path, upload_token: str, default: str) -> str:
    # Best-effort only: create_job's own _resolve_source_path raises the real
    # error for a missing token, so this never needs to fail the request.
    matches = sorted(uploads_path.glob(f"{upload_token}-*"))
    if not matches:
        return default
    return matches[0].name[len(upload_token) + 1 :]


def sanitize_filename(filename: str | None, default: str) -> str:
    """Produces a filesystem-safe name for the on-disk upload path.

    Strips characters invalid on Windows (`: < > " | ? *`) and escapes
    reserved device stems (CON, NUL, COM1...) that would otherwise collide
    with OS device names regardless of extension.
    """
    candidate = Path(filename or default).name
    stripped = _strip_forbidden_chars(candidate).strip()
    if not stripped:
        stripped = default
    return _escape_reserved_stem(stripped)


def _parse_chain_steps(raw: object) -> list[str]:
    """Los pasos llegan como lista separada por comas en un campo de formulario.

    Lo usan las DOS cadenas (voz y limpieza). El ORDEN de lo que llega no
    importa en ninguna: cada catalogo reordena segun su propia causalidad, y un
    request no deberia poder invertirla.

    Acepta `object` y no `str | None` a proposito: los tests de ruta de este
    repo llaman las corrutinas DIRECTO, sin pasar por FastAPI, asi que un
    parametro no provisto llega como el sentinel `Form(...)` en vez de None.
    Tratar cualquier cosa que no sea str como "no vino" cubre los dos caminos.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


class ResolvedVideoJobFields(NamedTuple):
    model_name: str
    scale: int
    output_container: str
    video_codec: str
    video_preset: str
    crf: int
    keep_audio: bool
    fps_multiplier: int
    target_fps: str | None
    audio_enhance: str | None


def resolve_video_job_fields(
    profile: VideoProfile,
    model_name: str | None,
    scale: int | None,
    output_container: str | None,
    video_codec: str | None,
    video_preset: str | None,
    crf: int | None,
    keep_audio: bool | None,
    fps_multiplier: int | None = None,
    target_fps: str | None = None,
    audio_enhance: str | None = None,
) -> ResolvedVideoJobFields:
    """Resolves per-request overrides against the profile default.

    Uses `is not None` (not `or`) for numeric fields so an explicit 0 from the
    caller is preserved instead of being silently replaced by the profile default.
    target_fps and audio_enhance have no profile default (explicit per-job
    only) — passed through as-is.
    """
    return ResolvedVideoJobFields(
        model_name=model_name or profile["model_key"],
        scale=scale if scale is not None else profile["scale"],
        output_container=output_container or "mp4",
        video_codec=video_codec or profile["video_codec"],
        video_preset=video_preset or profile["video_preset"],
        crf=crf if crf is not None else profile["crf"],
        keep_audio=keep_audio if keep_audio is not None else profile["keep_audio"],
        fps_multiplier=fps_multiplier if fps_multiplier is not None else profile.get("fps_multiplier", 1),
        target_fps=target_fps,
        audio_enhance=audio_enhance,
    )


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def get_video_job_manager(request: Request) -> VideoJobManager:
    return request.app.state.video_job_manager


def get_audio_job_manager(request: Request) -> AudioJobManager:
    return request.app.state.audio_job_manager


def get_generation_job_manager(request: Request) -> GenerationJobManager:
    return request.app.state.generation_job_manager


def get_generation_installer(request: Request) -> GenerationModelInstaller:
    return request.app.state.generation_installer


def get_generation_converter(request: Request) -> GenerationModelConverter:
    return request.app.state.generation_converter


def get_storage(request: Request) -> StorageService:
    return request.app.state.storage


def get_media_tools(request: Request) -> MediaTools:
    return request.app.state.media_tools


def get_devices_service(request: Request) -> DevicesService:
    return request.app.state.devices_service


def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


def get_hf_client(request: Request) -> HfClient:
    return request.app.state.hf_client


def get_model_installer(request: Request) -> ModelInstaller:
    return request.app.state.model_installer


def get_update_service(request: Request) -> UpdateService:
    return request.app.state.update_service


async def resolve_request_device(device: str | None, devices: DevicesService, settings: Settings) -> str:
    """Resolves the `device` Form param to a concrete device id.

    An explicit device (including the "auto" sentinel) is passed through
    untouched -- "auto" is validated/resolved downstream by the job manager
    and its device_router. `None` means the caller didn't pin a device: if
    ENABLE_AUTO_ROUTE is on, that implicitly means "auto" too; otherwise it
    defaults to settings.DEFAULT_DEVICE via DevicesService.resolve_default --
    real hardware enumeration, so it is dispatched through asyncio.to_thread
    rather than blocking the event loop.
    """
    if device is not None:
        return device
    if settings.enable_auto_route:
        return AUTO_DEVICE_ID
    device_list = await asyncio.to_thread(devices.list_devices)
    return devices.resolve_default(device_list)["id"]


def _progress_pct_from_metadata(metadata: dict[str, Any]) -> float | None:
    progress = metadata.get("progress")
    return progress * 100 if isinstance(progress, (int, float)) else None


def job_to_response(job: UpscaleJob) -> JobResponse:
    download_url = f"/api/v1/jobs/{job.id}/download" if job.status == JobStatus.completed else None
    return JobResponse(
        job_id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        model_name=job.model_name,
        scale=job.scale,
        output_format=job.output_format,
        model_id=job.model_id,
        device=job.device,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        metadata=job.metadata,
        progress_pct=_progress_pct_from_metadata(job.metadata),
        download_url=download_url,
        owner_id=job.owner_id,
    )


def _can_view_job(job: Any, user: AuthenticatedUser) -> bool:
    return Permission.jobs_read_all in user.permissions or job.owner_id == user.id


def _can_cancel_job(job: Any, user: AuthenticatedUser) -> bool:
    return Permission.jobs_cancel_any in user.permissions or job.owner_id == user.id


def _require_read_all_if_requested(all_users: bool, current_user: AuthenticatedUser) -> None:
    if all_users and Permission.jobs_read_all not in current_user.permissions:
        raise HTTPException(status_code=403, detail="No tenés permiso para ver los jobs de todos los usuarios")


def video_job_to_response(job: VideoUpscaleJob) -> VideoJobResponse:
    download_url = f"/api/v1/video/jobs/{job.id}/download" if job.status == JobStatus.completed else None
    return VideoJobResponse(
        job_id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        model_name=job.model_name,
        scale=job.scale,
        output_container=job.output_container,
        video_codec=job.video_codec,
        video_preset=job.video_preset,
        crf=job.crf,
        keep_audio=job.keep_audio,
        target_height=job.target_height,
        fps_multiplier=job.fps_multiplier,
        target_fps=job.target_fps,
        audio_enhance=job.audio_enhance,
        audio_restore=job.audio_restore,
        audio_track_indices=job.audio_track_indices,
        keep_subtitles=job.keep_subtitles,
        audio_output_format=job.audio_output_format,
        interp_engine=job.interp_engine,
        model_id=job.model_id,
        device=job.device,
        backend=job.backend,
        video_encoder=job.video_encoder,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        metadata=job.metadata,
        progress_pct=_progress_pct_from_metadata(job.metadata),
        download_url=download_url,
        owner_id=job.owner_id,
    )


def _audio_separation_spec(job: AudioJob):
    if not job.separate or job.separation_model is None:
        return None
    from app.services.engines.separation_models import SEPARATION_MODELS

    return SEPARATION_MODELS.get(job.separation_model)


def _audio_stem_downloads(job: AudioJob) -> list[AudioStemDownloadResponse] | None:
    """Una descarga por stem de un job de separacion completado, ORDENADAS: la
    primera es la que el usuario quiere (la misma que downloadUrl).

    Son tantas como declare el modelo — dos en karaoke y limpieza, cuatro en
    los multi-stem — y salen del catalogo, no de contar archivos."""
    spec = _audio_separation_spec(job)
    if spec is None or job.status != JobStatus.completed or not job.stem_output_paths:
        return None
    base = f"/api/v1/audio/jobs/{job.id}/download"
    return [
        AudioStemDownloadResponse(id=stem.id, label_key=stem.label_key, url=f"{base}?stem={stem.id}")
        for stem in spec.stems
    ]


def _audio_vocals_download_url(job: AudioJob) -> str | None:
    # Compat v0.59, y SOLO eso: el campo promete que downloadUrl trae la
    # instrumental y esto la voz. Un modelo multi-stem rompe esa promesa —
    # downloadUrl trae bateria — asi que ahi no se emite y el cliente usa
    # `stems`. Emitirlo igual daria una URL que funciona debajo de un nombre
    # que miente.
    if job.status != JobStatus.completed or not job.stem_output_paths:
        return None
    spec = _audio_separation_spec(job)
    if spec is not None and spec.stem_ids() != ("instrumental", "vocals"):
        return None
    return f"/api/v1/audio/jobs/{job.id}/download?stem=vocals"


def audio_job_to_response(job: AudioJob) -> AudioJobResponse:
    download_url = f"/api/v1/audio/jobs/{job.id}/download" if job.status == JobStatus.completed else None
    vocals_download_url = _audio_vocals_download_url(job)
    return AudioJobResponse(
        id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        denoise=job.denoise,
        restore=job.restore,
        device=job.device,
        output_format=job.output_format,
        lossy_quality=job.lossy_quality,
        master=job.master,
        voice_steps=list(job.voice_steps),
        voice_delivery=job.voice_delivery,
        voice_presence_db=job.voice_presence_db,
        cleanup_steps=list(job.cleanup_steps),
        separate=job.separate,
        separation_model=job.separation_model,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress_pct=_progress_pct_from_metadata(job.metadata),
        stages=job.metadata.get("stages"),
        metadata=job.metadata,
        error=job.error,
        download_url=download_url,
        stems=_audio_stem_downloads(job),
        vocals_download_url=vocals_download_url,
        owner_id=job.owner_id,
    )


def generation_job_to_response(job: GenerationJob) -> GenerationJobResponse:
    download_url = (
        f"/api/v1/generation/jobs/{job.id}/download" if job.status == JobStatus.completed else None
    )
    return GenerationJobResponse(
        id=job.id, status=job.status, prompt=job.prompt, negative_prompt=job.negative_prompt,
        model_id=job.model_id, steps=job.steps, guidance=job.guidance, width=job.width,
        height=job.height, seed=job.seed, device=job.device, auto_upscale=job.auto_upscale,
        seed_was_random=bool(job.metadata.get("seedWasRandom")),
        execution_provider=job.metadata.get("executionProvider"),
        strength=job.strength if job.init_image_path is not None else None,
        scheduler=job.scheduler,
        speed_class=job.metadata.get("speedClass"),
        precision=job.metadata.get("precision"),
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
        progress_pct=_progress_pct_from_metadata(job.metadata), stages=job.metadata.get("stages"),
        error=job.error, download_url=download_url, owner_id=job.owner_id,
        upscale_error=job.metadata.get("upscaleError"),
        # La URL de descarga no lleva extensión, así que sin esta bandera la UI
        # no puede saber si pintar una imagen o un reproductor.
        is_video=job.model_id.startswith(VIDEO_MODEL_PREFIX),
    )


def model_entry_to_response(entry: ModelEntry) -> ModelResponse:
    return ModelResponse(
        id=entry.id,
        name=entry.name,
        kind=entry.kind.value,
        source=entry.source,
        scale=entry.scale,
        arch=entry.arch,
        size_bytes=entry.size_bytes,
        status=entry.status.value,
        error=entry.error,
    )


def update_status_to_response(status: UpdateStatus) -> UpdateCheckResponse:
    return UpdateCheckResponse(
        current_version=status.current_version,
        latest_version=status.latest_version,
        update_available=status.update_available,
        release_url=status.release_url,
        published_at=status.published_at,
        checked_at=status.checked_at,
        error=status.error,
    )


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    settings: Settings = Depends(get_settings),
    jobs: JobManager = Depends(get_job_manager),
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        engine=settings.engine,
        # gpu_concurrency now reports the per-device GPU concurrency (the
        # role GPU_CONCURRENCY used to play before per-device semaphores) --
        # the JSON field name is unchanged so existing API consumers keep
        # working.
        gpu_concurrency=settings.per_device_gpu_concurrency,
        queue_depth=jobs.queue_depth(),
        video_queue_depth=video_jobs.queue_depth(),
    )


@router.get("/engine", response_model=EngineInfoResponse)
async def engine_info(request: Request, settings: Settings = Depends(get_settings)) -> EngineInfoResponse:
    engine = request.app.state.engine
    media_tools = request.app.state.media_tools
    return EngineInfoResponse(
        engine=settings.engine,
        configured_binary=settings.engine_binary,
        configured_models_dir=settings.engine_models_dir,
        available=engine.available(),
        default_model=settings.default_model,
        allowed_scales=settings.allowed_scale_values,
        supported_models=[SupportedModelResponse(**item) for item in settings.model_catalog],
        max_upload_mb=settings.max_upload_mb,
        max_video_upload_mb=settings.max_video_upload_mb,
        video_profiles=[VideoProfileResponse(**item) for item in settings.video_profile_catalog],
        ffmpeg_available=media_tools.available(),
    )


@router.get("/update-check", response_model=UpdateCheckResponse)
async def update_check(
    force: bool = Query(default=False),
    updates: UpdateService = Depends(get_update_service),
) -> UpdateCheckResponse:
    status = await updates.check(force=force)
    return update_status_to_response(status)


@router.get("/devices", response_model=DevicesResponse)
async def list_devices(devices: DevicesService = Depends(get_devices_service)) -> DevicesResponse:
    from app.services import ep_registry

    def snapshot() -> tuple[list[dict], list[DeviceInfoResponse]]:
        # active_ep_for_device puede disparar la inicialización del registry
        # (enumeración DXGI + registro de plugins): fuera del event loop.
        raw = devices.list_devices()
        enriched = []
        for item in raw:
            ep_status = ep_registry.active_ep_for_device(item["id"], devices.settings)
            enriched.append(
                DeviceInfoResponse(
                    **item,
                    active_ep=ep_status.ep_name,
                    ep_label=ep_status.label,
                    ep_state=ep_status.state,
                    ep_detail=ep_status.detail,
                )
            )
        return raw, enriched

    raw_list, device_list = await asyncio.to_thread(snapshot)
    default_device = devices.resolve_default(raw_list)
    return DevicesResponse(devices=device_list, default_device_id=default_device["id"])


@router.post(
    "/jobs", response_model=CreateJobResponse, status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(default="realesrgan-x4plus"),
    model_id: str | None = Form(default=None),
    device: str | None = Form(default=None),
    scale: int = Form(default=4),
    output_format: str = Form(default="png"),
    jobs: JobManager = Depends(get_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    devices: DevicesService = Depends(get_devices_service),
) -> CreateJobResponse:
    original_name = Path(file.filename or "upload.png").name
    safe_name = sanitize_filename(original_name, default="upload.png")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    resolved_device = await resolve_request_device(device, devices, settings)
    current_user = current_user_from_request(request)

    job: UpscaleJob | None = None
    try:
        await storage.save_upload(file, destination)
        job = await jobs.create_job(
            source_path=destination,
            original_filename=original_name,
            model_name=model_name,
            model_id=model_id,
            device=resolved_device,
            scale=scale,
            output_format=output_format,
            job_id=token,
            owner=current_user,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating image job")
        raise HTTPException(status_code=500, detail="Failed to process the uploaded image") from exc
    finally:
        if job is None and destination.exists():
            destination.unlink(missing_ok=True)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/v1/jobs/{job.id}",
        download_url=None,
    )


@router.post(
    "/video/jobs", response_model=CreateJobResponse, status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_video_job(
    request: Request,
    file: UploadFile | None = File(default=None),
    upload_token: str | None = Form(default=None),
    profile_key: str = Form(default="anime-balanced-2x"),
    model_name: str | None = Form(default=None),
    scale: int | None = Form(default=None),
    output_container: str | None = Form(default=None),
    video_codec: str | None = Form(default=None),
    video_preset: str | None = Form(default=None),
    crf: int | None = Form(default=None),
    keep_audio: bool | None = Form(default=None),
    fps_multiplier: int | None = Form(default=None),
    target_fps: str | None = Form(default=None),
    # Alto de salida pedido. Presente = el usuario pidio una RESOLUCION y no un
    # multiplicador ciego, que es lo que llevaba a pedir 15360x8640 sin querer.
    target_height: int | None = Form(default=None),
    audio_enhance: str | None = Form(default=None),
    audio_restore: str | None = Form(default=None),
    audio_track_indices: str | None = Form(default=None),
    keep_subtitles: bool = Form(default=False),
    audio_output_format: str | None = Form(default=None),
    interp_engine: str | None = Form(default=None),
    model_id: str | None = Form(default=None),
    device: str | None = Form(default=None),
    backend: str | None = Form(default=None),
    video_encoder: str | None = Form(default=None),
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    devices: DevicesService = Depends(get_devices_service),
) -> CreateJobResponse:
    profile = settings.get_video_profile(profile_key)
    if not profile:
        raise HTTPException(status_code=400, detail=f"Unknown profile: {profile_key}")

    # FastAPI passes these as their declared type (str|None, bool); a direct
    # unit-test call that omits one receives the Form() sentinel instead
    # (always truthy), so normalize anything of the wrong type to its real
    # default -- same pattern already used below for backend/video_encoder/
    # interp_engine.
    upload_token_value = upload_token if isinstance(upload_token, str) else None
    audio_track_indices_value = audio_track_indices if isinstance(audio_track_indices, str) else None
    keep_subtitles_value = keep_subtitles if isinstance(keep_subtitles, bool) else False

    has_file = bool(file and file.filename)
    if has_file == bool(upload_token_value):
        raise HTTPException(status_code=400, detail="Provide exactly one of file or upload_token")

    backend_value = backend if isinstance(backend, str) else None
    video_encoder_value = video_encoder if isinstance(video_encoder, str) else "auto"
    audio_output_format_value = audio_output_format if isinstance(audio_output_format, str) else "auto"
    interp_engine_value = interp_engine if isinstance(interp_engine, str) else RIFE_ENGINE

    resolved_device = await resolve_request_device(device, devices, settings)
    current_user = current_user_from_request(request)

    resolved = resolve_video_job_fields(
        profile,
        model_name,
        scale,
        output_container,
        video_codec,
        video_preset,
        crf,
        keep_audio,
        fps_multiplier,
        target_fps,
        audio_enhance,
    )

    destination: Path | None = None
    job: VideoUpscaleJob | None = None
    try:
        parsed_audio_track_indices = (
            [int(i) for i in audio_track_indices_value.split(",") if i.strip()]
            if audio_track_indices_value
            else None
        )
        if has_file:
            original_name = Path(file.filename or "upload.mp4").name
            safe_name = sanitize_filename(original_name, default="upload.mp4")
            new_upload_token = uuid4().hex
            destination = settings.uploads_path / f"{new_upload_token}-{safe_name}"
            await storage.save_upload(file, destination, max_mb=settings.max_video_upload_mb)
            source_path = destination
            resolved_upload_token = None
            # The freshly-generated upload token doubles as the job id here (as
            # before this task): one request == one upload == one job, so no
            # collision risk. When reusing a staged upload_token below, the job
            # id must instead be independent -- see the else branch.
            new_job_id = new_upload_token
        else:
            original_name = _original_filename_from_staged_upload(
                settings.uploads_path, upload_token_value, default="upload.mp4"
            )
            source_path = None
            resolved_upload_token = upload_token_value
            # A staged upload_token can be reused across multiple job-creation
            # attempts (e.g. retrying with different settings after a
            # validation error), so it must NOT double as the job id or a
            # second successful job would overwrite the first in jobs[].
            new_job_id = uuid4().hex

        job = await video_jobs.create_job(
            source_path=source_path,
            upload_token=resolved_upload_token,
            original_filename=original_name,
            model_name=resolved.model_name,
            scale=resolved.scale,
            output_container=resolved.output_container,
            video_codec=resolved.video_codec,
            video_preset=resolved.video_preset,
            crf=resolved.crf,
            keep_audio=resolved.keep_audio,
            target_height=target_height if isinstance(target_height, int) else None,
            fps_multiplier=resolved.fps_multiplier,
            target_fps=resolved.target_fps,
            audio_enhance=resolved.audio_enhance,
            audio_restore=audio_restore,
            audio_track_indices=parsed_audio_track_indices,
            keep_subtitles=keep_subtitles_value,
            audio_output_format=audio_output_format_value,
            interp_engine=interp_engine_value,
            model_id=model_id,
            device=resolved_device,
            backend=backend_value,
            video_encoder=video_encoder_value,
            job_id=new_job_id,
            owner=current_user,
        )
        job.metadata["profileKey"] = profile_key
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating video job")
        raise HTTPException(status_code=500, detail="Failed to process the uploaded video") from exc
    finally:
        # Only ever clean up a destination THIS request wrote (the `file`
        # path). A staged upload_token pre-dates this request (written by a
        # prior /video/analyze call) and must survive a failed attempt here
        # so the caller can retry with different job parameters without
        # re-uploading the video.
        if job is None and destination is not None and destination.exists():
            destination.unlink(missing_ok=True)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/v1/video/jobs/{job.id}",
        download_url=None,
    )


@router.post(
    "/video/analyze", response_model=AnalyzeVideoResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def analyze_video(
    file: UploadFile = File(...),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    media_tools: MediaTools = Depends(get_media_tools),
) -> AnalyzeVideoResponse:
    original_name = Path(file.filename or "upload.mp4").name
    safe_name = sanitize_filename(original_name, default="upload.mp4")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"

    try:
        await storage.save_upload(file, destination, max_mb=settings.max_video_upload_mb)
    except ValueError as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probe: dict[str, Any] | None = None
    try:
        probe = await media_tools.ffprobe_json(destination)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid video") from exc
    except RuntimeError as exc:
        logger.exception("ffprobe unavailable while analyzing uploaded video")
        raise HTTPException(status_code=500, detail="Video analysis is unavailable") from exc
    except Exception as exc:
        logger.exception("Unexpected error while analyzing uploaded video")
        raise HTTPException(status_code=500, detail="Failed to analyze the uploaded video") from exc
    finally:
        if probe is None and destination.exists():
            destination.unlink(missing_ok=True)

    audio_tracks = parse_audio_tracks(probe)
    subtitle_tracks = parse_subtitle_tracks(probe)
    return AnalyzeVideoResponse(
        upload_token=token,
        audio_tracks=[
            AudioTrackResponse(
                index=t.index, codec=t.codec, channels=t.channels, is_default=t.is_default, language=t.language
            )
            for t in audio_tracks
        ],
        subtitle_tracks=[
            SubtitleTrackResponse(index=t.index, codec=t.codec, language=t.language) for t in subtitle_tracks
        ],
    )


@router.get("/jobs", response_model=JobsListResponse)
async def list_jobs(
    all_users: bool = Query(default=False, alias="all"),
    jobs: JobManager = Depends(get_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> JobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return JobsListResponse(jobs=[job_to_response(job) for job in visible])


@router.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require(Permission.jobs_read_own))])
async def get_job(
    job_id: str,
    jobs: JobManager = Depends(get_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> JobResponse:
    job = jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_response(job)


@router.post(
    "/jobs/{job_id}/cancel", response_model=JobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_job(
    job_id: str,
    jobs: JobManager = Depends(get_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> JobResponse:
    job = jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Job not found")
    if not jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return job_to_response(job)


@router.get("/video/jobs", response_model=VideoJobsListResponse)
async def list_video_jobs(
    all_users: bool = Query(default=False, alias="all"),
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> VideoJobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in video_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return VideoJobsListResponse(jobs=[video_job_to_response(job) for job in visible])


@router.get(
    "/video/jobs/{job_id}", response_model=VideoJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_video_job(
    job_id: str,
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> VideoJobResponse:
    job = video_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Video job not found")
    return video_job_to_response(job)


@router.post(
    "/video/jobs/{job_id}/cancel", response_model=VideoJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_video_job(
    job_id: str,
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> VideoJobResponse:
    job = video_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Video job not found")
    if not video_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return video_job_to_response(job)


@router.get("/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_job(
    job_id: str,
    jobs: JobManager = Depends(get_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> FileResponse:
    job = jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")


@router.get("/video/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_video_job(
    job_id: str,
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> FileResponse:
    job = video_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Video job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Video job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")


@router.post(
    "/audio/jobs", response_model=CreateJobResponse, status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_audio_job(
    request: Request,
    file: UploadFile = File(...),
    denoise: str | None = Form(default=None),
    restore: str | None = Form(default=None),
    device: str | None = Form(default=None),
    output_format: str = Form(default="flac"),
    # Calidad de los destinos con perdida (mp3/m4a). Se ignora en wav/flac.
    lossy_quality: str = Form(default=DEFAULT_LOSSY_QUALITY),
    voice_steps: str | None = Form(default=None),
    voice_delivery: str | None = Form(default=None),
    master: str | None = Form(default=None),
    voice_presence_db: float | None = Form(default=None),
    # Cadena de limpieza: CSV de ids del catalogo (GET /audio/capabilities ->
    # cleanupSteps). El orden que llegue da igual, lo fija el catalogo.
    cleanup_steps: str | None = Form(default=None),
    separate: bool = Form(default=False),
    separation_model: str | None = Form(default=None),
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> CreateJobResponse:
    original_name = Path(file.filename or "upload.wav").name
    safe_name = sanitize_filename(original_name, default="upload.wav")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    current_user = current_user_from_request(request)

    job: AudioJob | None = None
    try:
        await storage.save_upload(file, destination, max_mb=settings.max_audio_upload_mb)
        job = await audio_jobs.create_job(
            source_path=destination,
            original_filename=original_name,
            denoise=denoise,
            restore=restore,
            device=device,
            output_format=output_format,
            # isinstance y no truthiness: llamada directa (tests) => el default
            # es el FieldInfo de Form(), que es truthy pero no es un str.
            lossy_quality=(
                lossy_quality
                if isinstance(lossy_quality, str) and lossy_quality
                else DEFAULT_LOSSY_QUALITY
            ),
            voice_steps=_parse_chain_steps(voice_steps),
            voice_delivery=voice_delivery if isinstance(voice_delivery, str) else None,
            master=master if isinstance(master, str) and master else None,
            voice_presence_db=(
                voice_presence_db if isinstance(voice_presence_db, (int, float)) else None
            ),
            cleanup_steps=_parse_chain_steps(cleanup_steps),
            # isinstance y no truthiness: llamada directa (tests) => el default
            # es el FieldInfo de Form(), que es truthy.
            separate=separate if isinstance(separate, bool) else False,
            separation_model=(
                separation_model if isinstance(separation_model, str) and separation_model else None
            ),
            job_id=token,
            owner=current_user,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating audio job")
        raise HTTPException(status_code=500, detail="Failed to process the uploaded audio") from exc
    finally:
        if job is None and destination.exists():
            destination.unlink(missing_ok=True)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/v1/audio/jobs/{job.id}",
        download_url=None,
    )


def get_download_job_manager(request: Request) -> DownloadJobManager:
    return request.app.state.download_jobs


def download_job_to_response(job: DownloadJob) -> DownloadJobResponse:
    return DownloadJobResponse(
        id=job.id,
        status=job.status,
        url=job.url,
        max_height=job.max_height,
        audio_only=job.audio_only,
        audio_format=job.audio_format,
        audio_bitrate_kbps=job.audio_bitrate_kbps,
        video_container=job.video_container,
        media_title=job.media_title,
        media_uploader=job.media_uploader,
        extractor=job.extractor,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress_pct=job.progress_pct,
        downloaded_bytes=job.downloaded_bytes,
        total_bytes=job.total_bytes,
        # Solo el NOMBRE: la ruta absoluta expondria la estructura de directorios del
        # servidor sin darle nada util a quien la lee.
        output_files=[path.name for path in job.output_paths],
        # El DIRECTORIO si viaja (a diferencia de las rutas completas por archivo): es
        # la carpeta que el usuario ya ve en Ajustes, y sin ella la UI decia el nombre
        # del archivo sin decir donde buscarlo.
        output_directory=str(job.output_paths[0].parent) if job.output_paths else "",
        error=job.error,
        owner_id=job.owner_id,
    )


@router.post(
    "/download/probe",
    response_model=MediaProbeResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def probe_media(payload: CreateDownloadJobRequest) -> MediaProbeResponse:
    """Que hay en esta URL, sin descargar.

    Existe para que la UI muestre titulo, duracion y calidades ANTES de comprometerse, y
    sobre todo para que se vea que una URL es una playlist de 200 items antes de
    disparar 200 descargas.
    """
    try:
        validate_url(payload.url)
        info = await asyncio.to_thread(fetch_engine.probe, payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except fetch_engine.FetchUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - el motivo del sitio es lo util
        raise HTTPException(status_code=422, detail=describe_failure(exc)) from exc
    return MediaProbeResponse(
        title=info.title,
        duration_seconds=info.duration_seconds,
        uploader=info.uploader,
        extractor=info.extractor,
        is_playlist=info.is_playlist,
        entry_count=info.entry_count,
        available_heights=list(info.available_heights),
        thumbnail_url=info.thumbnail_url,
    )


@router.post(
    "/download/jobs",
    response_model=DownloadJobResponse,
    status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_download_job(
    payload: CreateDownloadJobRequest,
    download_jobs: DownloadJobManager = Depends(get_download_job_manager),
    request: Request = None,
) -> DownloadJobResponse:
    try:
        job = await download_jobs.create_job(
            url=payload.url,
            max_height=payload.max_height,
            audio_only=payload.audio_only,
            audio_format=payload.audio_format,
            audio_bitrate_kbps=payload.audio_bitrate_kbps,
            video_container=payload.video_container,
            include_playlist=payload.include_playlist,
            playlist_limit=payload.playlist_limit,
            subtitle_languages=payload.subtitle_languages,
            owner=current_user_from_request(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return download_job_to_response(job)


@router.get("/download/jobs", response_model=DownloadJobsListResponse)
async def list_download_jobs(
    all_users: bool = Query(default=False, alias="all"),
    download_jobs: DownloadJobManager = Depends(get_download_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> DownloadJobsListResponse:
    """Las descargas del usuario.

    Sin esto, recargar el navegador perdia la descarga: seguia corriendo en el
    servidor pero la UI no tenia como preguntar cual quedo viva.
    """
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in download_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return DownloadJobsListResponse(jobs=[download_job_to_response(job) for job in visible])


@router.get(
    "/download/jobs/{job_id}",
    response_model=DownloadJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_download_job(
    job_id: str,
    download_jobs: DownloadJobManager = Depends(get_download_job_manager),
    request: Request = None,
) -> DownloadJobResponse:
    job = download_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    # 404 y no 403: un 403 confirmaria que el job existe, y con el la URL que otro
    # usuario decidio bajar.
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Download job not found")
    return download_job_to_response(job)


@router.post(
    "/download/jobs/{job_id}/cancel",
    response_model=DownloadJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_download_job(
    job_id: str,
    download_jobs: DownloadJobManager = Depends(get_download_job_manager),
    request: Request = None,
) -> DownloadJobResponse:
    job = download_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Download job not found")
    download_jobs.cancel_job(job_id)
    return download_job_to_response(job)


@router.get(
    "/download/jobs/{job_id}/download",
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def download_download_job_file(
    job_id: str,
    index: int = Query(default=0, ge=0),
    download_jobs: DownloadJobManager = Depends(get_download_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> FileResponse:
    """El archivo producido, servido por HTTP.

    Sin esto un usuario remoto ve el NOMBRE del archivo y la ruta de otra maquina:
    la descarga termina y el resultado queda inalcanzable. `index` porque una
    playlist produce varios archivos y cada uno necesita su propio link.
    """
    job = download_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Download job not found")
    if job.status != JobStatus.completed or not job.output_paths:
        raise HTTPException(status_code=409, detail="Download job is not completed yet")
    if index >= len(job.output_paths):
        raise HTTPException(status_code=404, detail="No such file in this download job")
    path = job.output_paths[index]
    # El sweeper de retencion puede borrar el archivo antes de que el job expire.
    if not path.is_file():
        raise HTTPException(status_code=404, detail="El archivo ya no esta en disco")
    return FileResponse(path=path, filename=path.name, media_type="application/octet-stream")


def get_transcribe_job_manager(request: Request) -> TranscribeJobManager:
    return request.app.state.transcribe_jobs


def transcribe_job_to_response(job: TranscribeJob) -> TranscribeJobResponse:
    return TranscribeJobResponse(
        id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        model_id=job.model_id,
        language=job.language,
        device=job.device,
        output_mode=job.output_mode,
        target_language=job.target_language,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress_pct=job.progress_pct,
        text=job.text,
        error=job.error,
        owner_id=job.owner_id,
        download_url=(
            f"/api/v1/transcribe/jobs/{job.id}/download"
            if job.status == JobStatus.completed and job.output_path
            else None
        ),
        video_url=(
            f"/api/v1/transcribe/jobs/{job.id}/download?fmt=video"
            if job.subtitled_video_path is not None
            else None
        ),
        dub_overflow_segments=job.dub_overflow_segments,
    )


@router.post(
    "/transcribe/jobs", response_model=CreateJobResponse, status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_transcribe_job(
    request: Request,
    file: UploadFile = File(...),
    model_id: str = Form(...),
    language: str | None = Form(default=None),
    device: str | None = Form(default=None),
    output_mode: str = Form(default="text"),
    target_language: str | None = Form(default=None),
    transcribe_jobs: TranscribeJobManager = Depends(get_transcribe_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> CreateJobResponse:
    original_name = Path(file.filename or "upload.wav").name
    safe_name = sanitize_filename(original_name, default="upload.wav")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    current_user = current_user_from_request(request)

    job: TranscribeJob | None = None
    try:
        await storage.save_upload(file, destination, max_mb=settings.max_audio_upload_mb)
        job = await transcribe_jobs.create_job(
            source_path=destination,
            original_filename=original_name,
            model_id=model_id,
            # Los tests de ruta llaman las corrutinas DIRECTO, sin FastAPI, asi que
            # un Form no provisto llega como su sentinel en vez de None.
            language=language if isinstance(language, str) and language else None,
            device=device if isinstance(device, str) and device else None,
            job_id=token,
            owner=current_user,
            output_mode=output_mode if isinstance(output_mode, str) and output_mode else "text",
            target_language=(
                target_language if isinstance(target_language, str) and target_language else None
            ),
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating transcribe job")
        raise HTTPException(
            status_code=500, detail="Failed to process the uploaded audio"
        ) from exc
    finally:
        if job is None and destination.exists():
            destination.unlink(missing_ok=True)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/v1/transcribe/jobs/{job.id}",
        download_url=None,
    )


@router.get("/transcribe/jobs", response_model=TranscribeJobsListResponse)
async def list_transcribe_jobs(
    all_users: bool = Query(default=False, alias="all"),
    transcribe_jobs: TranscribeJobManager = Depends(get_transcribe_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> TranscribeJobsListResponse:
    """Los trabajos de transcripcion del usuario.

    Sin esto, recargar el navegador perdia el trabajo: seguia corriendo en el
    servidor pero la UI no tenia como preguntar cual quedo vivo.
    """
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in transcribe_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return TranscribeJobsListResponse(jobs=[transcribe_job_to_response(job) for job in visible])


@router.get(
    "/transcribe/jobs/{job_id}",
    response_model=TranscribeJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_transcribe_job(
    job_id: str,
    transcribe_jobs: TranscribeJobManager = Depends(get_transcribe_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case injection
    # still recognizes it -- `lenient_issubclass` rejects unions. Direct/unit-test
    # calls that omit this kwarg still get `None`.
    request: Request = None,
) -> TranscribeJobResponse:
    job = transcribe_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    # 404 y no 403 a proposito: un 403 confirmaria que el job existe. Una
    # transcripcion es el contenido de un audio ajeno, asi que ni su existencia se
    # filtra.
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Transcribe job not found")
    return transcribe_job_to_response(job)


@router.post(
    "/transcribe/jobs/{job_id}/cancel",
    response_model=TranscribeJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_transcribe_job(
    job_id: str,
    transcribe_jobs: TranscribeJobManager = Depends(get_transcribe_job_manager),
    request: Request = None,
) -> TranscribeJobResponse:
    job = transcribe_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Transcribe job not found")
    transcribe_jobs.cancel_job(job_id)
    return transcribe_job_to_response(job)


_TRANSLATION = TranslationEngine(translation_dir(get_settings()))


def get_translation() -> TranslationEngine:
    return _TRANSLATION


@router.get("/translation/pairs", response_model=TranslationPairsResponse)
async def list_translation_pairs(
    engine: TranslationEngine = Depends(get_translation),
) -> TranslationPairsResponse:
    instalados = list(engine.available_pairs())
    ya_estan = {f"{p.source}-{p.target}" for p in instalados}
    return TranslationPairsResponse(
        pairs=[TranslationPairResponse(source=p.source, target=p.target) for p in instalados],
        # Solo los que faltan: ofrecer bajar algo que ya esta es ruido.
        installable=[par for par in INSTALLABLE_PAIRS if par not in ya_estan],
    )


@router.get(
    "/transcribe/jobs/{job_id}/download",
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def download_transcribe_job(
    job_id: str,
    transcribe_jobs: TranscribeJobManager = Depends(get_transcribe_job_manager),
    request: Request = None,
    fmt: str = "txt",
    translate_to: str | None = None,
    translation: TranslationEngine = Depends(get_translation),
) -> Response:
    job = transcribe_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Transcribe job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Transcribe job is not completed yet")
    if translate_to and fmt == "txt" and not job.segments:
        raise HTTPException(status_code=409, detail="This job has no segments to translate")
    if fmt == "video":
        # El video con subtitulos no se rinde al vuelo: lo dejo ffmpeg al
        # terminar el job, y solo si se pidio ese modo de salida.
        if job.subtitled_video_path is None or not job.subtitled_video_path.exists():
            raise HTTPException(
                status_code=409, detail="This job has no subtitled video"
            )
        return FileResponse(
            path=job.subtitled_video_path,
            filename=Path(job.original_filename).name,
            media_type="video/mp4",
        )
    if fmt not in SUBTITLE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unknown subtitle format: {fmt}")
    spec = SUBTITLE_FORMATS[fmt]
    stem = Path(job.original_filename).stem
    if fmt == "txt":
        return FileResponse(
            path=job.output_path, filename=f"{stem}{spec.extension}", media_type=spec.media_type
        )
    # Los subtitulos se rinden al vuelo desde los segmentos: el job ya los tiene,
    # asi que no hace falta escribir un archivo por formato al terminar.
    segments = list(job.segments)
    if translate_to:
        # Se traduce SEGMENTO POR SEGMENTO y se emparejan por indice: los
        # tiempos pertenecen a cada segmento y traducir el texto corrido los
        # perderia.
        pair = parse_pair(job.language or "en", translate_to)
        traducidos = translation.translate([s.text for s in segments], pair)
        segments = [
            TranscriptSegment(start=s.start, end=s.end, text=t)
            for s, t in zip(segments, traducidos)
        ]
    body = render_segments(segments, fmt)
    return Response(
        content=body.encode("utf-8"),
        media_type=spec.media_type,
        headers={"Content-Disposition": f'attachment; filename="{stem}{spec.extension}"'},
    )


# --- generacion de voz -----------------------------------------------------
# La sintesis devuelve el WAV directo y no crea un job: 1,90 s de audio salen en
# menos de medio segundo (medido 2026-08-04), asi que encolarlo costaria mas que
# hacerlo.

_TTS_ENGINE = KokoroTtsEngine(get_settings())


def get_tts_engine() -> KokoroTtsEngine:
    return _TTS_ENGINE


def tts_model_dir(settings: Settings) -> Path:
    return kokoro_dir(settings)


@router.get("/tts/capabilities", response_model=TtsCapabilitiesResponse)
async def tts_capabilities(
    engine: KokoroTtsEngine = Depends(get_tts_engine),
    settings_dep: Settings = Depends(get_settings),
    model_dir: Path | None = None,
) -> TtsCapabilitiesResponse:
    directory = model_dir or tts_model_dir(settings_dep)
    if not engine.available(directory):
        return TtsCapabilitiesResponse(
            available=False,
            reason=missing_pack_message("kokoro"),
            missing_pack="kokoro",
        )
    return TtsCapabilitiesResponse(available=True, voices=available_voices(directory))


@router.post("/tts/synthesize")
async def synthesize_speech(
    payload: SynthesizeSpeechRequest,
    engine: KokoroTtsEngine = Depends(get_tts_engine),
    settings_dep: Settings = Depends(get_settings),
    model_dir: Path | None = None,
) -> Response:
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="No hay texto que sintetizar.")
    directory = model_dir or tts_model_dir(settings_dep)
    if not engine.available(directory):
        raise HTTPException(
            status_code=409,
            detail={"reason": missing_pack_message("kokoro"), "missingPack": "kokoro"},
        )
    phonemes = text_to_phonemes(payload.text, payload.language)
    if not phonemes:
        raise HTTPException(status_code=400, detail="No se pudo convertir ese texto a fonemas.")
    try:
        audio = await asyncio.to_thread(
            engine.synthesize, model_dir=directory, phonemes=phonemes, voice=payload.voice
        )
    except (TtsUnavailable, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    buffer = io.BytesIO()
    soundfile.write(buffer, audio, TTS_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="voz.wav"'},
    )


def get_asr_installer(request: Request) -> AsrModelInstaller:
    return request.app.state.asr_installer


@router.get("/asr/models/search", response_model=ModelSearchResponse)
async def search_asr_models(
    q: str = Query("", max_length=200),
    hf_client: HfClient = Depends(get_hf_client),
) -> ModelSearchResponse:
    """Modelos de reconocimiento de voz en Hugging Face.

    El filtro por TAG es lo que decide que es un modelo de ASR: la clasificacion por
    nombres de archivo no puede distinguirlo de otro modelo de audio, y decir lo
    contrario seria prometer una deteccion que no existe.
    """
    try:
        results = await hf_client.search(q, task_tags=ASR_SEARCH_TASK_TAGS, sort="downloads")
    except Exception as exc:
        logger.exception("Hugging Face ASR search failed for query %r", q)
        raise HTTPException(status_code=502, detail="Hugging Face search failed") from exc
    return _search_results_to_response(results, strategy_for("audio"))


@router.post(
    "/asr/models/install", response_model=CreateInstallResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def install_asr_model(
    payload: InstallModelRequest,
    installer: AsrModelInstaller = Depends(get_asr_installer),
) -> CreateInstallResponse:
    install_id = await installer.install_from_hf(payload.repo_id)
    return CreateInstallResponse(
        install_id=install_id,
        status_url=f"/api/v1/asr/models/install/{install_id}",
    )


@router.get("/asr/models/install/{install_id}", response_model=InstallStatusResponse)
async def get_asr_install_status(
    install_id: str,
    installer: AsrModelInstaller = Depends(get_asr_installer),
) -> InstallStatusResponse:
    job = installer.status(install_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Install job not found")
    return InstallStatusResponse(
        install_id=job.id,
        repo_id=job.repo_id,
        status=job.status.value,
        progress_pct=job.progress_pct,
        model_id=job.model_id,
        error=job.error,
    )


@router.get("/video/capabilities", response_model=VideoCapabilitiesResponse)
async def video_capabilities(settings: Settings = Depends(get_settings)) -> VideoCapabilitiesResponse:
    interp_engines = [
        engine for engine in sorted(INTERP_ENGINES) if settings.interp_engine_available(engine)
    ]
    return VideoCapabilitiesResponse(interp_engines=interp_engines)


@router.get("/audio/capabilities", response_model=AudioCapabilitiesResponse)
async def audio_capabilities(settings: Settings = Depends(get_settings)) -> AudioCapabilitiesResponse:
    denoise_modes = [mode for mode in sorted(AUDIO_ENHANCE_MODES) if settings.audio_enhance_available(mode)]
    restore_modes = [
        mode for mode in sorted(AUDIO_RESTORE_MODES) if settings.audio_restore_mode_available(mode)
    ]
    from app.services.audio_mastering import MASTERING_PRESETS
    from app.services.cleanup_chain import OVERPROCESSING_PASS_THRESHOLD, cleanup_catalog
    from app.services.engines.separation_models import SEPARATION_MODELS

    installed_separation_models = set(settings.karaoke_installed_models())
    return AudioCapabilitiesResponse(
        cleanup_steps=[
            CleanupStepResponse(
                id=step.model_id,
                name=SEPARATION_MODELS[step.model_id].name,
                family=step.family,
                covers=list(step.covers),
                installed=step.model_id in installed_separation_models,
                description_key=SEPARATION_MODELS[step.model_id].description_key,
            )
            for step in cleanup_catalog()
        ],
        cleanup_overprocessing_threshold=OVERPROCESSING_PASS_THRESHOLD,
        denoise_modes=denoise_modes,
        restore_available=bool(restore_modes),
        restore_modes=restore_modes,
        output_formats=sorted(AUDIO_OUTPUT_FORMATS),
        lossy_formats=sorted(LOSSY_OUTPUT_FORMATS),
        # En orden de calidad DESCENDENTE: el orden del catalogo, no alfabetico.
        lossy_qualities=[
            LossyQualityResponse(id=quality, bitrates=bitrates)
            for quality, bitrates in LOSSY_QUALITY_BITRATES.items()
        ],
        default_lossy_quality=DEFAULT_LOSSY_QUALITY,
        mastering_presets=[
            MasteringPresetResponse(
                id=p.id,
                label_key=p.label_key,
                description_key=p.description_key,
                target_lufs=p.target_lufs,
            )
            for p in MASTERING_PRESETS
        ],
        separation_models=[
            SeparationModelResponse(
                id=spec.id,
                name=spec.name,
                installed=spec.id in installed_separation_models,
                primary_stem=spec.primary_stem,
                category=spec.category,
                architecture=spec.architecture,
                description_key=spec.description_key,
                warning_key=spec.warning_key,
                stems=[
                    SeparationStemResponse(id=stem.id, label_key=stem.label_key)
                    for stem in spec.stems
                ],
            )
            for spec in SEPARATION_MODELS.values()
        ],
    )


@router.get("/audio/voice-catalog", response_model=VoiceCatalogResponse)
async def get_voice_catalog() -> VoiceCatalogResponse:
    """Los pasos y destinos de la cadena de voz.

    La ESTRUCTURA viene del backend a proposito: el orden de los pasos tiene
    causalidad (ver build_filter_chain) y los numeros de loudness son
    especificaciones publicadas, asi que hay una sola fuente de verdad. La COPIA
    en cambio viaja como clave de traduccion y la arma el frontend, que es la
    capa que conoce el idioma activo.
    """
    return VoiceCatalogResponse(
        steps=[
            {
                "id": info.id,
                "label_key": info.label_key,
                "description_key": info.description_key,
                "kind": info.kind,
                "default_enabled": info.default_enabled,
            }
            for info in step_catalog()
        ],
        deliveries=[
            {
                "id": choice["id"],
                "label_key": choice["labelKey"],
                "description_key": choice["descriptionKey"],
                "lufs": choice["lufs"],
                "true_peak_db": choice["truePeakDb"],
            }
            for choice in delivery_choices()
        ],
    )


@router.get("/audio/jobs", response_model=AudioJobsListResponse)
async def list_audio_jobs(
    all_users: bool = Query(default=False, alias="all"),
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> AudioJobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in audio_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return AudioJobsListResponse(jobs=[audio_job_to_response(job) for job in visible])


@router.get(
    "/audio/jobs/{job_id}", response_model=AudioJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_audio_job(
    job_id: str,
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> AudioJobResponse:
    job = audio_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Audio job not found")
    return audio_job_to_response(job)


@router.post(
    "/audio/jobs/{job_id}/cancel", response_model=AudioJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_audio_job(
    job_id: str,
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> AudioJobResponse:
    job = audio_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Audio job not found")
    if not audio_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return audio_job_to_response(job)


# Ids validos cuando el job NO tiene un modelo de separacion del catalogo
# (jobs clasicos): compat v0.59, stem=vocals responde 409 explicando.
LEGACY_AUDIO_DOWNLOAD_STEMS = ("instrumental", "vocals")


def _valid_stems_for(job: AudioJob) -> tuple[str, ...]:
    spec = _audio_separation_spec(job)
    if spec is None:
        return LEGACY_AUDIO_DOWNLOAD_STEMS
    return spec.stem_ids()


def _stem_output_path(job: AudioJob, stem: str, valid_stems: tuple[str, ...]) -> Path:
    # El mapa por id es la fuente de verdad: con cuatro stems, "el que no es el
    # primero" no identifica un archivo. output_path solo cubre el caso legacy
    # de un job sin mapa (separacion previa a stem_output_paths, o sin modelo
    # del catalogo), donde el primer stem ES la unica salida.
    path = job.stem_output_paths.get(stem)
    if path is not None:
        return path
    if stem == valid_stems[0] and job.output_path is not None:
        return job.output_path
    raise HTTPException(
        status_code=409,
        detail=f"This job did not run separation; there is no {stem} stem",
    )


@router.get("/audio/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_audio_job(
    job_id: str,
    # CONTRATO consumido tambien por el flujo MCP: los ids de stem validos son
    # los del modelo del job (karaoke: instrumental|vocals; reverb_hq:
    # dry|wet). Sin stem se sirve el principal (= output_path, que en jobs sin
    # separacion es la unica salida). 400 stem invalido listando los validos
    # DEL JOB; 409 al pedir el secundario en un job sin separacion.
    stem: str | None = Query(default=None),
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> FileResponse:
    if not isinstance(stem, str):
        # Llamada directa (tests): el default es el FieldInfo de Query().
        stem = None
    job = audio_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Audio job not found")
    valid_stems = _valid_stems_for(job)
    if stem is None:
        stem = valid_stems[0]
    if stem not in valid_stems:
        raise HTTPException(
            status_code=400,
            detail=f"stem must be one of {', '.join(valid_stems)}",
        )
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Audio job is not completed yet")
    output_path = _stem_output_path(job, stem, valid_stems)
    return FileResponse(path=output_path, filename=output_path.name, media_type="application/octet-stream")


@router.get("/models", response_model=ModelsResponse)
async def list_models(registry: ModelRegistry = Depends(get_model_registry)) -> ModelsResponse:
    return ModelsResponse(models=[model_entry_to_response(entry) for entry in registry.list()])


def _result_to_response(item, strategy: CompatStrategy) -> HfModelSearchResultResponse:
    verdict, reason = strategy.classify(item.filenames, item.gated)
    # Un repo ready_onnx no tiene paso de export, asi que no hay nada que elegir:
    # no se ofrecen opciones (ver alcance de B en el spec de 2026-07-28).
    options = (
        strategy.install_options(item.filenames)
        if verdict == "needs_conversion"
        else InstallOptions()
    )
    return HfModelSearchResultResponse(
        id=item.id,
        author=item.author,
        pipeline_tag=item.pipeline_tag,
        downloads=item.downloads,
        likes=item.likes,
        tags=list(item.tags),
        compat=verdict,
        compat_reason_key=reason.key,
        compat_reason_params=dict(reason.params),
        available_precisions=list(options.precisions),
    )


def _search_results_to_response(
    results: list, strategy: CompatStrategy
) -> ModelSearchResponse:
    return ModelSearchResponse(
        results=[_result_to_response(item, strategy) for item in results]
    )


@router.get("/models/search", response_model=ModelSearchResponse)
async def search_models(
    q: str = Query(..., min_length=1),
    hf_client: HfClient = Depends(get_hf_client),
) -> ModelSearchResponse:
    try:
        results = await hf_client.search(q)
    except Exception as exc:
        logger.exception("Hugging Face search failed for query %r", q)
        raise HTTPException(status_code=502, detail="Hugging Face search failed") from exc
    return _search_results_to_response(results, strategy_for("image"))
@router.get("/models/preflight", response_model=UpscalerPreflightResponse)
async def preflight_upscaler_model(
    request: Request,
    repo_id: str = Query(..., alias="repoId"),
    hf_client: HfClient = Depends(get_hf_client),
    settings: Settings = Depends(get_settings),
) -> UpscalerPreflightResponse:
    """Capacidad y compatibilidad de un repo de upscaler antes de instalarlo.

    Reusa las mediciones genericas del pre-flight de generacion. NO estima pico de
    VRAM: el factor de vram_estimate asume activaciones que crecen con la
    resolucion, y para un upscaler que hace tiling esa premisa es falsa. La VRAM
    libre viaja como dato, sin veredicto de "no entra".
    """
    report = await preflight_upscaler(
        hf_client=hf_client,
        devices_service=request.app.state.devices_service,
        settings=settings,
        probes=request.app.state.resource_probes,
        repo_id=repo_id,
        strategy=strategy_for("image"),
    )
    return UpscalerPreflightResponse(**asdict(report))


@router.post(
    "/models/install", response_model=CreateInstallResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def install_model(
    payload: InstallModelRequest,
    installer: ModelInstaller = Depends(get_model_installer),
) -> CreateInstallResponse:
    try:
        install_id = await installer.install_from_hf(payload.repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateInstallResponse(
        install_id=install_id, status_url=f"/api/v1/models/install/{install_id}"
    )


@router.get("/models/install/{install_id}", response_model=InstallStatusResponse)
async def get_install_status(
    install_id: str, installer: ModelInstaller = Depends(get_model_installer)
) -> InstallStatusResponse:
    job = installer.status(install_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Install job not found")
    return InstallStatusResponse(
        install_id=job.id,
        repo_id=job.repo_id,
        status=job.status.value,
        progress_pct=job.progress_pct,
        model_id=job.model_id,
        error=job.error,
        conversion_id=job.conversion_id,
    )


@router.delete(
    "/models/{model_id}", status_code=204,
    dependencies=[Depends(require(Permission.models_delete))],
)
async def delete_model(
    model_id: str, installer: ModelInstaller = Depends(get_model_installer)
) -> Response:
    try:
        await installer.delete(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelProtectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)


INIT_IMAGE_TOKEN_PATTERN = "%s-*"


def _resolve_init_image(settings: Settings, token: str | None) -> Path | None:
    """Traduce el token a la ruta staged, sin confiar en el token.

    El token viene del cliente, asi que se valida que sea hexadecimal antes de
    usarlo en un glob: sin eso, un token con separadores de ruta podria hacer que
    el glob mire fuera del directorio de uploads.
    """
    if token is None:
        return None
    if not token or len(token) > 64 or any(char not in "0123456789abcdef" for char in token):
        raise HTTPException(status_code=400, detail="init_image_token invalido")
    matches = sorted(settings.uploads_path.glob(INIT_IMAGE_TOKEN_PATTERN % token))
    if not matches:
        raise HTTPException(status_code=400, detail="init_image_token desconocido o expirado")
    return matches[0]


@router.post(
    "/generation/init-image",
    response_model=InitImageResponse,
    status_code=201,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def upload_init_image(
    file: UploadFile = File(...),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> InitImageResponse:
    """Deja la imagen de partida en staging y devuelve su token.

    Va aparte del job para que POST /generation/jobs siga siendo JSON: volverlo
    multipart cambiaria el contrato de todos sus clientes por una capacidad que no
    todos usan. Es el mismo patron que /video/analyze ya usa.
    """
    original_name = Path(file.filename or "init.png").name
    safe_name = sanitize_filename(original_name, default="init.png")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    await storage.save_upload(file, destination)

    try:
        from PIL import Image

        with Image.open(destination) as image:
            width, height = image.size
    except Exception as exc:  # noqa: BLE001 - un archivo que no es imagen se rechaza aca
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f"El archivo no es una imagen legible: {exc}"
        ) from exc

    return InitImageResponse(
        init_image_token=token,
        original_filename=original_name,
        width=width,
        height=height,
    )


def _video_aware_sampling(payload, settings: Settings) -> tuple[int, float]:
    """Los defaults de imagen (25 pasos, CFG 7.5) queman un modelo de video
    destilado, que se entrenó con 4 pasos y CFG 1. Si el que llama no los pidió
    explícitamente, mandan los del modelo."""
    from app.services.engines.sdcpp_video import resolve_video_model

    model = resolve_video_model(payload.model_id, settings)
    if model is None:
        return payload.steps, payload.guidance
    chosen = payload.model_fields_set
    return (
        payload.steps if "steps" in chosen else model.default_steps,
        payload.guidance if "guidance" in chosen else model.default_guidance,
    )


@router.post(
    "/generation/jobs", response_model=GenerationJobResponse, status_code=201,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_generation_job(
    payload: CreateGenerationJobRequest,
    generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    settings: Settings = Depends(get_settings),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> GenerationJobResponse:
    current_user = current_user_from_request(request)
    init_image_path = _resolve_init_image(settings, payload.init_image_token)
    mask_image_path = _resolve_init_image(settings, payload.mask_image_token)
    if payload.strength is not None:
        strength = payload.strength
    else:
        strength = 0.85 if mask_image_path is not None else 0.6
    steps, guidance = _video_aware_sampling(payload, settings)
    try:
        job = await generation_jobs.create_job(
            prompt=payload.prompt, negative_prompt=payload.negative_prompt, model_id=payload.model_id,
            steps=steps, guidance=guidance, width=payload.width, height=payload.height,
            seed=payload.seed, scheduler=payload.scheduler,
            device=payload.device, frames=payload.frames, fps=payload.fps,
            init_image_path=init_image_path, strength=strength,
            mask_image_path=mask_image_path,
            auto_upscale=payload.auto_upscale,
            upscale_model_name=payload.upscale_model_name, upscale_scale=payload.upscale_scale,
            upscale_model_id=payload.upscale_model_id, owner=current_user,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating generation job")
        raise HTTPException(status_code=500, detail="Failed to create the generation job") from exc
    return generation_job_to_response(job)


@router.get("/generation/jobs", response_model=GenerationJobsListResponse)
async def list_generation_jobs(
    all_users: bool = Query(default=False, alias="all"),
    generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> GenerationJobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in generation_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return GenerationJobsListResponse(jobs=[generation_job_to_response(job) for job in visible])


@router.get(
    "/generation/jobs/{job_id}", response_model=GenerationJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_generation_job(
    job_id: str, generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> GenerationJobResponse:
    job = generation_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Generation job not found")
    return generation_job_to_response(job)


@router.post(
    "/generation/jobs/{job_id}/cancel", response_model=GenerationJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_generation_job(
    job_id: str, generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> GenerationJobResponse:
    job = generation_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Generation job not found")
    if not generation_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return generation_job_to_response(job)


@router.get("/generation/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_generation_job(
    job_id: str, generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> FileResponse:
    job = generation_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Generation job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Generation job is not completed yet")
    media_type = "video/webm" if job.output_path.suffix.lower() == ".webm" else "image/png"
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type=media_type)


def _entry_supports_inpaint(settings: Settings, entry: Any) -> bool:
    from app.services.engines.generation_onnx import _read_declared_class_name
    from app.services.generation_pipeline_modes import supports_inpaint

    try:
        declared = _read_declared_class_name(settings.models_path / (entry.file_path or ""))
    except Exception:  # noqa: BLE001 -- una lectura fallida no es un veredicto (mismo criterio que el job manager)
        return True
    return supports_inpaint(declared)


def _entry_inpaint_only(settings: Settings, entry: Any) -> bool:
    from app.services.engines.generation_onnx import _read_declared_class_name
    from app.services.generation_pipeline_modes import is_dedicated_inpaint_class

    try:
        declared = _read_declared_class_name(settings.models_path / (entry.file_path or ""))
    except Exception:  # noqa: BLE001 -- una lectura fallida no es un veredicto
        return False
    return is_dedicated_inpaint_class(declared)


def get_vulkan_installer(request: Request):
    return request.app.state.vulkan_installer


@router.post(
    "/generation/models/vulkan", response_model=CreateInstallResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def install_vulkan_model(
    payload: InstallVulkanModelRequest,
    installer=Depends(get_vulkan_installer),
) -> CreateInstallResponse:
    """Instala un checkpoint suelto para el lane Vulkan: solo se descarga."""
    try:
        install_id = await installer.install(payload.repo_id, payload.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateInstallResponse(
        install_id=install_id,
        status_url=f"/api/v1/generation/models/vulkan/{install_id}",
    )


@router.get("/generation/models/vulkan/{install_id}", response_model=VulkanInstallStatusResponse)
async def vulkan_install_status(
    install_id: str, installer=Depends(get_vulkan_installer)
) -> VulkanInstallStatusResponse:
    job = installer.status(install_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Install not found")
    return VulkanInstallStatusResponse(
        install_id=job.id, repo_id=job.repo_id, status=job.status.value,
        progress_pct=job.progress_pct, model_id=job.model_id, error=job.error,
    )


@router.get("/realtime/capabilities", response_model=RealtimeCapabilitiesResponse)
async def realtime_capabilities(
    settings: Settings = Depends(get_settings),
) -> RealtimeCapabilitiesResponse:
    from app.services.realtime_service import RealtimeService, available_presets

    service = RealtimeService(settings)
    if not service.available():
        return RealtimeCapabilitiesResponse(
            available=False,
            reason=(
                "El overlay de tiempo real no está instalado. Se baja aparte porque "
                "usa Magpie, que es software libre con licencia GPL y corre como "
                "programa separado."
            ),
        )
    return RealtimeCapabilitiesResponse(
        available=True,
        presets=[
            RealtimePresetResponse(
                id=p.id, label_key=p.label_key, description_key=p.description_key
            )
            for p in available_presets()
        ],
    )


@router.post(
    "/realtime/start", response_model=RealtimeStartedResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def start_realtime(
    payload: StartRealtimeRequest, settings: Settings = Depends(get_settings)
) -> RealtimeStartedResponse:
    from app.services.realtime_service import RealtimeService

    try:
        pid = RealtimeService(settings).start(
            preset=payload.preset, max_frame_rate=payload.max_frame_rate
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RealtimeStartedResponse(pid=pid, preset=payload.preset)


@router.get("/generation/video/capabilities", response_model=VideoGenerationCapabilitiesResponse)
async def video_generation_capabilities(
    settings: Settings = Depends(get_settings),
) -> VideoGenerationCapabilitiesResponse:
    from app.services.engines.sdcpp_video import list_video_models

    # Este lane no depende de optimum ni de ningún execution provider: corre por
    # Vulkan en cualquier GPU. Si el pack no está bajado, simplemente no hay modelos.
    models = [
        VideoModelSummary(
            id=model.id,
            name=f"{model.name} (Vulkan)",
            fast=model.turbo,
            default_steps=model.default_steps,
            default_guidance=model.default_guidance,
        )
        for model in list_video_models(settings)
    ]
    return VideoGenerationCapabilitiesResponse(
        available=bool(models),
        models=models,
        default_frames=DEFAULT_VIDEO_FRAMES,
        default_fps=DEFAULT_VIDEO_FPS,
        max_frames=MAX_VIDEO_FRAMES,
    )


@router.get("/generation/capabilities", response_model=GenerationCapabilitiesResponse)
async def generation_capabilities(
    registry: ModelRegistry = Depends(get_model_registry),
    devices_service: DevicesService = Depends(get_devices_service),
    settings: Settings = Depends(get_settings),
) -> GenerationCapabilitiesResponse:
    available, reason = generation_dependencies_available()
    if not available:
        return GenerationCapabilitiesResponse(available=False, reason=reason, cpu_only=True)
    # error queda afuera: una conversion fallida se ve en Models con su motivo;
    # el dropdown de Generate no es lugar para un modelo que no existe en disco.
    from app.services.generation_speed import speed_class as _speed_class

    models = [
        GenerationModelSummary(
            id=entry.id,
            name=entry.name,
            status=entry.status.value,
            supports_inpaint=_entry_supports_inpaint(settings, entry),
            speed=_speed_class(entry.name),
            inpaint_only=_entry_inpaint_only(settings, entry),
        )
        for entry in registry.list()
        if entry.kind == ModelKind.diffusion_onnx and entry.status != ModelStatus.error
    ]
    if settings.sdcpp_available():
        from app.services.engines.sdcpp_models import list_sdcpp_models

        # Un modelo por checkpoint: corren tal cual, sin conversion.
        models.extend(
            GenerationModelSummary(
                id=model.id, name=f"{model.name} (Vulkan)", supports_inpaint=False
            )
            for model in list_sdcpp_models(settings)
        )
    if settings.migan_available():
        from app.services.engines.migan_eraser import ERASER_MODEL_ID, ERASER_MODEL_LABEL

        # Se ofrece PRIMERO en el Editor: para sacar algo tarda menos de un segundo
        # contra los minutos de la difusión, y no inventa nada.
        models.insert(
            0,
            GenerationModelSummary(
                id=ERASER_MODEL_ID, name=ERASER_MODEL_LABEL, supports_inpaint=True, erase_only=True
            ),
        )
    device_infos = devices_service.list_devices()
    return GenerationCapabilitiesResponse(
        available=True,
        models=models,
        devices=[info["id"] for info in device_infos],
        cpu_only=all(info["kind"] != "gpu" for info in device_infos),
    )


@router.post(
    "/generation/models", response_model=CreateInstallResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def install_generation_model(
    payload: InstallModelRequest,
    installer: GenerationModelInstaller = Depends(get_generation_installer),
) -> CreateInstallResponse:
    try:
        install_id = await installer.install_from_hf(
            payload.repo_id,
            precision=payload.precision or "fp16",
            checkpoint_path=payload.checkpoint_path,
        )
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateInstallResponse(
        install_id=install_id, status_url=f"/api/v1/generation/models/install/{install_id}"
    )


@router.get("/generation/models/install/{install_id}", response_model=InstallStatusResponse)
async def get_generation_install_status(
    install_id: str, installer: GenerationModelInstaller = Depends(get_generation_installer)
) -> InstallStatusResponse:
    job = installer.status(install_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Install job not found")
    return InstallStatusResponse(
        install_id=job.id,
        repo_id=job.repo_id,
        status=job.status.value,
        progress_pct=job.progress_pct,
        model_id=job.model_id,
        error=job.error,
        conversion_id=job.conversion_id,
    )


@router.post(
    "/generation/models/convert", response_model=CreateConversionResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def convert_generation_model(
    payload: InstallModelRequest,
    converter: GenerationModelConverter = Depends(get_generation_converter),
) -> CreateConversionResponse:
    try:
        conversion_id = await converter.convert_from_hf(
            payload.repo_id,
            precision=payload.precision or "fp16",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateConversionResponse(
        conversion_id=conversion_id,
        status_url=f"/api/v1/generation/models/convert/{conversion_id}",
    )


@router.post(
    "/generation/models/{model_id}/create-inpaint",
    response_model=CreateConversionResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def create_inpaint_version(
    model_id: str,
    converter: GenerationModelConverter = Depends(get_generation_converter),
    registry: ModelRegistry = Depends(get_model_registry),
    settings: Settings = Depends(get_settings),
) -> CreateConversionResponse:
    from app.services.generation_inpaint_merge import (
        InpaintMergeUnsupportedError,
        merge_family_for,
    )

    from app.services.generation_installer import (
        _ensure_checkpoint_listed,
        _generation_model_id,
    )

    entry = registry.get(model_id)
    if entry is None or entry.kind != ModelKind.diffusion_onnx:
        raise HTTPException(status_code=404, detail="Modelo de generación no encontrado")
    if not entry.source.startswith("hf:"):
        raise HTTPException(
            status_code=400,
            detail="Este modelo no tiene un repo de origen en Hugging Face para mergear",
        )
    source_repo = entry.source[3:]
    if entry.id != _generation_model_id(source_repo) and entry.checkpoint_path is None:
        # Instalado desde un checkpoint suelto por una versión anterior que no
        # persistía el archivo de origen: el source solo guarda el repo, y
        # mergear bajaría OTROS pesos que los que el usuario instaló.
        raise HTTPException(
            status_code=400,
            detail=(
                "Este modelo se instaló desde un checkpoint único con una "
                "versión anterior que no guardaba de qué archivo vino, así que "
                "no se puede saber qué pesos mergear. Reinstalá el modelo (la "
                "instalación ahora guarda el checkpoint de origen) y volvé a "
                "crear la versión de inpainting."
            ),
        )
    if _entry_inpaint_only(settings, entry):
        raise HTTPException(status_code=400, detail="Este modelo ya es de inpainting")
    try:
        # Validar la familia ANTES de encolar 20GB de descargas: un SD3/LCM sin
        # inpaint oficial debe fallar acá con el motivo, no a los 40 minutos.
        from app.services.engines.generation_onnx import _read_declared_class_name

        declared = _read_declared_class_name(settings.models_path / (entry.file_path or ""))
        merge_family_for(declared)
    except InpaintMergeUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- sin model_index legible no hay merge posible
        raise HTTPException(
            status_code=400, detail="No se pudo leer la clase del modelo instalado"
        ) from exc
    # El repo de origen debe publicar pesos PyTorch: los repos puramente ONNX
    # (los "_amdgpu", por ejemplo) fallarían tras bajar ~14GB. Validar acá
    # cumple el contrato "validar ANTES de encolar" del resto del endpoint.
    try:
        source_files = await converter.hf_client.repo_files(source_repo)
    except Exception as exc:  # noqa: BLE001 -- sin listado no hay veredicto honesto
        raise HTTPException(
            status_code=400, detail=f"No se pudo listar el repo de origen: {exc}"
        ) from exc
    if entry.checkpoint_path is not None:
        # Instalado desde un checkpoint suelto: los pesos PyTorch son ESE
        # archivo. Verificar que siga publicado antes de encolar la descarga.
        try:
            _ensure_checkpoint_listed(source_files, source_repo, entry.checkpoint_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        if not _repo_has_torch_weights(source_files):
            raise HTTPException(
                status_code=400,
                detail=(
                    "El repo de origen solo publica pesos ONNX: el merge de inpainting "
                    "necesita los pesos PyTorch originales del modelo."
                ),
            )
    try:
        conversion_id = await converter.convert_inpaint_merge(
            source_repo, checkpoint_path=entry.checkpoint_path
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateConversionResponse(
        conversion_id=conversion_id,
        status_url=f"/api/v1/generation/models/convert/{conversion_id}",
    )


def _repo_has_torch_weights(files: list) -> bool:
    return any(
        "/" in hf_file.path
        and hf_file.path.endswith((".safetensors", ".bin"))
        and ".onnx" not in hf_file.path
        for hf_file in files
    )


def _optimize_source_repo(entry: ModelEntry) -> str:
    """Repo de origen del que se pueden volver a bajar los pesos torch.

    La fusión de grafo no se puede hacer sobre el ONNX fp16 instalado: hay que
    re-exportar los pesos originales en fp32. Sin origen no hay optimización.
    """
    from app.services.generation_optimize import (
        OptimizeUnsupportedError,
        is_inpaint_merge,
        is_optimized,
    )

    if is_optimized(entry.id):
        raise OptimizeUnsupportedError("Este modelo ya es la versión optimizada")
    if is_inpaint_merge(entry.id):
        # Los pesos de un merge de inpainting sólo existen como el ONNX que
        # produjo ESE merge: re-exportar desde el repo daría el UNet sin mergear
        # y la variante "optimizada" sería otro modelo disfrazado.
        raise OptimizeUnsupportedError(
            "Una versión de inpainting no se puede optimizar: sus pesos son el "
            "resultado del merge y no existen como modelo de origen. Optimizá el "
            "modelo base."
        )
    if not entry.source.startswith("hf:"):
        raise OptimizeUnsupportedError(
            "Este modelo no tiene un repo de origen en Hugging Face del que "
            "re-exportar los pesos"
        )
    return entry.source[3:]


async def _optimize_architecture_for(entry: ModelEntry, settings: Settings):
    from app.services.engines.generation_onnx import _read_declared_class_name
    from app.services.generation_optimize import OptimizeUnsupportedError, architecture_for

    try:
        declared = _read_declared_class_name(
            settings.models_path / (entry.file_path or "")
        )
    except Exception as exc:  # noqa: BLE001 - sin model_index no hay veredicto
        raise OptimizeUnsupportedError(
            "No se pudo leer la clase del modelo instalado"
        ) from exc
    return architecture_for(declared)


@router.post(
    "/generation/models/{model_id}/optimize",
    response_model=CreateConversionResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def optimize_generation_model(
    model_id: str,
    request: Request,
    converter: GenerationModelConverter = Depends(get_generation_converter),
    registry: ModelRegistry = Depends(get_model_registry),
    settings: Settings = Depends(get_settings),
) -> CreateConversionResponse:
    """Crea la variante optimizada por fusión de grafo de un modelo instalado.

    Todo se valida ACÁ: la conversión tarda entre 3 y 10 minutos y pide decenas
    de GB de RAM, así que un rechazo tiene que llegar antes de encolar, no a la
    mitad del trabajo.
    """
    from app.services.generation_optimize import (
        OptimizeUnsupportedError,
        ensure_enough_ram,
        optimized_model_id,
    )
    from app.services.model_preflight import measure_free_ram

    entry = registry.get(model_id)
    if entry is None or entry.kind != ModelKind.diffusion_onnx:
        raise HTTPException(status_code=404, detail="Modelo de generación no encontrado")
    if entry.status != ModelStatus.installed:
        raise HTTPException(
            status_code=400, detail="Este modelo todavía no terminó de instalarse"
        )
    existing = registry.get(optimized_model_id(model_id))
    if existing is not None and existing.status != ModelStatus.error:
        raise HTTPException(
            status_code=400, detail="Este modelo ya tiene una versión optimizada"
        )
    try:
        source_repo = _optimize_source_repo(entry)
        architecture = await _optimize_architecture_for(entry, settings)
        ensure_enough_ram(
            architecture,
            measure_free_ram(getattr(request.app.state, "resource_probes", {})),
        )
    except OptimizeUnsupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        source_files = await converter.hf_client.repo_files(source_repo)
    except Exception as exc:  # noqa: BLE001 - sin listado no hay veredicto honesto
        raise HTTPException(
            status_code=400, detail=f"No se pudo listar el repo de origen: {exc}"
        ) from exc
    if entry.checkpoint_path is None and not _repo_has_torch_weights(source_files):
        raise HTTPException(
            status_code=400,
            detail=(
                "El repo de origen solo publica pesos ONNX: la optimización "
                "necesita re-exportar los pesos PyTorch originales en fp32."
            ),
        )
    try:
        conversion_id = await converter.optimize_installed(
            source_model_id=entry.id,
            source_model_name=entry.name,
            repo_id=source_repo,
            checkpoint_path=entry.checkpoint_path,
            installed_dir=settings.models_path / (entry.file_path or ""),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateConversionResponse(
        conversion_id=conversion_id,
        status_url=f"/api/v1/generation/models/convert/{conversion_id}",
    )


def _conversion_to_response(job) -> ConversionStatusResponse:
    progress = job.metadata.get("progress")
    return ConversionStatusResponse(
        conversion_id=job.id,
        repo_id=job.repo_id,
        status=job.status,
        progress_pct=round(progress * 100, 1) if progress is not None else None,
        stage=job.metadata.get("stage"),
        stages=job.metadata.get("stages"),
        model_id=job.model_id,
        error=job.error,
    )


@router.get("/generation/models/conversions", response_model=list[ConversionStatusResponse])
async def list_active_conversions(
    converter: GenerationModelConverter = Depends(get_generation_converter),
) -> list[ConversionStatusResponse]:
    """Las conversiones que siguen corriendo.

    Existe porque el id de la conversion vivia SOLO en la pantalla: al cambiar de
    seccion se iba con ella y la barra no podia volver a engancharse. La
    conversion nunca se perdia, pero el usuario veia que si — y convertir un SDXL
    tarda cerca de media hora.
    """
    return [_conversion_to_response(job) for job in converter.active()]


@router.post(
    "/generation/models/convert/{conversion_id}/cancel",
    response_model=ConversionStatusResponse,
)
async def cancel_conversion(
    conversion_id: str,
    converter: GenerationModelConverter = Depends(get_generation_converter),
) -> ConversionStatusResponse:
    """Corta una conversion en curso.

    El corte cae en el limite del siguiente submodelo: el export vive dentro de
    una libreria que no se puede interrumpir a la mitad.
    """
    if not converter.cancel(conversion_id):
        job = converter.status(conversion_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Conversion job not found")
        raise HTTPException(
            status_code=409, detail="Esa conversion ya termino: no hay nada que cortar."
        )
    job = converter.status(conversion_id)
    assert job is not None
    return _conversion_to_response(job)


@router.get("/generation/models/convert/{conversion_id}", response_model=ConversionStatusResponse)
async def get_conversion_status(
    conversion_id: str,
    converter: GenerationModelConverter = Depends(get_generation_converter),
) -> ConversionStatusResponse:
    job = converter.status(conversion_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Conversion job not found")
    progress = job.metadata.get("progress")
    return ConversionStatusResponse(
        conversion_id=job.id,
        repo_id=job.repo_id,
        status=job.status,
        progress_pct=round(progress * 100, 1) if progress is not None else None,
        stage=job.metadata.get("stage"),
        stages=job.metadata.get("stages"),
        model_id=job.model_id,
        error=job.error,
    )


@router.get("/generation/models/search", response_model=ModelSearchResponse)
async def search_generation_models(
    q: str = Query("", max_length=200),
    hf_client: HfClient = Depends(get_hf_client),
) -> ModelSearchResponse:
    # Query vacia = browse por descargas: el usuario ve modelos sin tener que
    # saber el repo_id exacto de antemano.
    try:
        results = await hf_client.search(
            q, task_tags=GENERATION_SEARCH_TASK_TAGS, sort=None if q else "downloads"
        )
    except Exception as exc:
        logger.exception("Hugging Face generation search failed for query %r", q)
        raise HTTPException(status_code=502, detail="Hugging Face search failed") from exc
    return _search_results_to_response(results, strategy_for("generate"))


@router.get("/generation/models/preflight", response_model=PreflightResponse)
async def preflight_generation_model(
    request: Request,
    repo_id: str = Query(..., alias="repoId"),
    width: int = Query(512, ge=64, le=4096),
    height: int = Query(512, ge=64, le=4096),
    hf_client: HfClient = Depends(get_hf_client),
    settings: Settings = Depends(get_settings),
) -> PreflightResponse:
    report = await preflight(
        hf_client=hf_client,
        devices_service=request.app.state.devices_service,
        settings=settings,
        probes=request.app.state.resource_probes,
        repo_id=repo_id,
        width=width,
        height=height,
    )
    return PreflightResponse(**asdict(report))

@router.get("/capabilities/tree", response_model=CapabilityTreeResponse)
async def capability_tree(
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
) -> CapabilityTreeResponse:
    """El arbol de lo que la app puede hacer, resuelto contra esta maquina.

    El frontend no puede mentir sobre lo que hay porque no decide: el status sale
    de mirar el disco y el registro, no de un flag persistido.
    """
    grouped = group_by_domain(resolve_capabilities(settings, registry))
    return CapabilityTreeResponse(
        domains=[
            CapabilityDomainResponse(
                domain=group.domain,
                label_key=group.label_key,
                capabilities=[_capability_to_response(item) for item in group.capabilities],
                roadmap=[_capability_to_response(item) for item in group.roadmap],
            )
            for group in grouped
        ]
    )


def _capability_to_response(item: ResolvedCapability) -> CapabilityResponse:
    return CapabilityResponse(
        id=item.id,
        domain=item.domain,
        label_key=item.label_key,
        status=item.status,
        provisioning=item.provisioning,
        job_kind=item.job_kind,
        strategies=list(item.strategies),
        missing_packs=list(item.missing_packs),
        unavailable_reason_key=item.unavailable_reason_key,
        setup_reason_key=item.setup_reason_key,
        activatable_settings=list(item.activatable_settings),
    )

def _resolved_by_id(settings: Settings, registry: ModelRegistry, capability_id: str) -> ResolvedCapability:
    try:
        return resolve_one(capability_id, settings, registry)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Capacidad desconocida: {capability_id!r}"
        ) from None


def _pack_to_provision(item: ResolvedCapability) -> str:
    # No se mira `provisioning`: video.upscale es de registro y aun asi necesita
    # el binario del motor. Lo que decide es si falta un paquete concreto.
    if not item.missing_packs:
        if item.status == "available":
            # Caso real: el primer click descargó el pack, la tarjeta quedó
            # vieja y el segundo click llegaba acá con un error críptico.
            raise HTTPException(
                status_code=409,
                detail="La capacidad ya está lista: el paquete ya se instaló. Recargá la página.",
            )
        raise HTTPException(
            status_code=400,
            detail=f"La capacidad {item.id!r} no tiene ningun paquete pendiente de descarga.",
        )
    return item.missing_packs[0]


def _provision_job_to_response(job: ProvisionJob) -> ProvisionJobResponse:
    return ProvisionJobResponse(
        job_id=job.id,
        pack=job.pack,
        status=job.status.value,
        error=job.error,
        status_url=f"/api/v1/capabilities/provision/{job.id}",
    )


@router.post(
    "/capabilities/{capability_id}/provision",
    response_model=ProvisionJobResponse,
    status_code=202,
)
async def provision_capability(
    capability_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ProvisionJobResponse:
    item = _resolved_by_id(settings, registry, capability_id)
    pack = _pack_to_provision(item)
    provisioner: PackProvisioner = request.app.state.pack_provisioner
    try:
        job_id = await provisioner.provision(pack)
    except UnknownPackError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = provisioner.status(job_id)
    assert job is not None
    return _provision_job_to_response(job)


@router.post("/packs/{pack}/provision", response_model=ProvisionJobResponse, status_code=202)
async def provision_pack(
    pack: str,
    request: Request,
    variant: str | None = None,
) -> ProvisionJobResponse:
    """Baja un paquete por su nombre, sin pasar por una capacidad.

    Casi toda pantalla sabe QUE le falta pero no a que capacidad pertenece, y
    varios paquetes existen sin figurar en el catalogo. Sin esta ruta esas
    pantallas no pueden ofrecer el boton, que es como se llego a tener 36
    mensajes diciendole al usuario que abriera una terminal.
    """
    provisioner: PackProvisioner = request.app.state.pack_provisioner
    try:
        job_id = await provisioner.provision(pack, variant)
    except (UnknownPackError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = provisioner.status(job_id)
    assert job is not None
    return _provision_job_to_response(job)


@router.get("/capabilities/provision/{job_id}", response_model=ProvisionJobResponse)
async def provision_status(job_id: str, request: Request) -> ProvisionJobResponse:
    provisioner: PackProvisioner = request.app.state.pack_provisioner
    job = provisioner.status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job de descarga desconocido")
    return _provision_job_to_response(job)



# Gates con los permisos que C ya define (mismo patrón que capability_routes):
# settings_read para leer, settings_write para escribir. Con AUTH_MODE=off el
# usuario off-mode tiene todos los permisos y esto es transparente.
@router.get(
    "/settings",
    response_model=EditableSettingsResponse,
    dependencies=[Depends(require(Permission.settings_read))],
)
async def get_editable_settings(
    settings: Settings = Depends(get_settings),
) -> EditableSettingsResponse:
    return EditableSettingsResponse(
        settings=[
            EditableSettingStatusResponse(**item)
            for item in editable_settings_status(settings)
        ]
    )


@router.get("/generation/prompt-presets", response_model=PromptPresetsResponse)
async def list_prompt_presets() -> PromptPresetsResponse:
    return PromptPresetsResponse(
        presets=[
            PromptPresetResponse(
                id=preset.id,
                mode=preset.mode,
                label_key=preset.label_key,
                prompt=preset.prompt,
                negative_prompt=preset.negative_prompt,
            )
            for preset in PROMPT_PRESETS
        ]
    )


# --- prompts guardados por el usuario --------------------------------------
# Son DATO del usuario y no copia de la app: se guardan tal cual y no se
# traducen. Los presets de fabrica viven en /generation/prompt-presets.

_SAVED_PROMPTS = SavedPromptStore(get_settings())


def get_saved_prompts() -> SavedPromptStore:
    return _SAVED_PROMPTS


def _owner_id(request: Request | None) -> str:
    user = current_user_from_request(request) if request is not None else None
    # Sin auth encendida sigue habiendo un dueño: la instalacion local.
    return user.id if user is not None else "local"


def _to_response(saved: Any) -> SavedPromptResponse:
    return SavedPromptResponse(
        id=saved.id,
        name=saved.name,
        prompt=saved.prompt,
        negative_prompt=saved.negative_prompt,
        mode=saved.mode,
    )


@router.get("/generation/saved-prompts", response_model=SavedPromptsResponse)
async def list_saved_prompts(
    request: Request = None,
    store: SavedPromptStore = Depends(get_saved_prompts),
) -> SavedPromptsResponse:
    return SavedPromptsResponse(
        prompts=[_to_response(p) for p in store.list_for(_owner_id(request))]
    )


@router.post("/generation/saved-prompts", response_model=SavedPromptResponse, status_code=201)
async def create_saved_prompt(
    payload: CreateSavedPromptRequest,
    request: Request = None,
    store: SavedPromptStore = Depends(get_saved_prompts),
) -> SavedPromptResponse:
    try:
        saved = store.save(
            owner_id=_owner_id(request),
            name=payload.name,
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            mode=payload.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(saved)


@router.delete("/generation/saved-prompts/{prompt_id}", status_code=204)
async def delete_saved_prompt(
    prompt_id: str,
    request: Request = None,
    store: SavedPromptStore = Depends(get_saved_prompts),
) -> Response:
    if not store.delete(owner_id=_owner_id(request), prompt_id=prompt_id):
        # 404 y no 403 a proposito: quien no es dueño no tiene por que saber
        # que ese prompt existe.
        raise HTTPException(status_code=404, detail="Saved prompt not found")
    return Response(status_code=204)


# --- conversion de voz -----------------------------------------------------
# Convierte una grabacion para que suene como otra. Devuelve el WAV directo, sin
# job: el maximo son 60 s de audio y la conversion tarda menos que eso.

_VOICE_CONVERSION = VoiceConversionEngine(Path(get_settings().runtime_dir).parent / "vendor")


def get_voice_conversion() -> VoiceConversionEngine:
    return _VOICE_CONVERSION


@router.get("/voice/conversion/capabilities", response_model=VoiceConversionCapabilitiesResponse)
async def voice_conversion_capabilities(
    engine: VoiceConversionEngine = Depends(get_voice_conversion),
) -> VoiceConversionCapabilitiesResponse:
    if not engine.available():
        return VoiceConversionCapabilitiesResponse(
            available=False,
            reason=missing_pack_message("voice-conversion"),
            missing_pack="voice-conversion",
            max_seconds=VOICE_MAX_SECONDS,
        )
    return VoiceConversionCapabilitiesResponse(available=True, max_seconds=VOICE_MAX_SECONDS)


async def _decoded_upload(upload: UploadFile, settings: Settings) -> Any:
    """Deja el audio en mono 16 kHz, que es lo unico que el modelo entiende."""
    import soundfile

    suffix = Path(upload.filename or "audio").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(await upload.read())
        origen = Path(handle.name)
    try:
        if needs_decoding(origen):
            destino = origen.with_suffix(".decoded.wav")
            command = build_decode_to_wav_command(
                ffmpeg=str(settings.ffmpeg_binary_path),
                source=origen,
                destination=destino,
                sample_rate=VOICE_SAMPLE_RATE,
            )
            # run_guarded_process y no create_subprocess_exec pelado: un upload
            # malformado podia dejar este ffmpeg colgado sin techo bloqueando el
            # request para siempre.
            _stdout, _stderr, returncode = await run_guarded_process(
                command, settings.subprocess_timeout
            )
            if returncode != 0 or not destino.exists():
                raise HTTPException(status_code=400, detail="No se pudo leer ese archivo de audio.")
            origen.unlink(missing_ok=True)
            origen = destino
        data, rate = soundfile.read(str(origen), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if rate != VOICE_SAMPLE_RATE:
            objetivo = int(len(mono) * VOICE_SAMPLE_RATE / rate)
            mono = numpy.interp(
                numpy.linspace(0, len(mono) - 1, objetivo), numpy.arange(len(mono)), mono
            ).astype("float32")
        return mono
    finally:
        origen.unlink(missing_ok=True)


@router.post("/voice/conversion")
async def convert_voice(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    engine: VoiceConversionEngine = Depends(get_voice_conversion),
    settings_dep: Settings = Depends(get_settings),
) -> Response:
    if not engine.available():
        raise HTTPException(
            status_code=409,
            detail={
                "reason": missing_pack_message("voice-conversion"),
                "missingPack": "voice-conversion",
            },
        )
    origen = await _decoded_upload(source, settings_dep)
    muestra = await _decoded_upload(reference, settings_dep)
    try:
        convertido = await asyncio.to_thread(engine.convert, source=origen, reference=muestra)
    except VoiceConversionUnavailable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    buffer = io.BytesIO()
    soundfile.write(buffer, convertido, VOICE_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="voz-convertida.wav"'},
    )


@router.get("/print/printers", response_model=PrintersResponse)
async def list_printers() -> PrintersResponse:
    return PrintersResponse(
        printers=[
            PrinterResponse(id=nombre, bed_mm=cama)
            for nombre, cama in sorted(PRINTER_BEDS.items())
        ]
    )


@router.post("/print/check", response_model=PrintCheckResponse)
async def check_print(
    file: UploadFile = File(...),
    printer: str = Form(default="ender-3"),
    target_axis: str | None = Form(default=None),
    target_mm: float | None = Form(default=None),
    settings_dep: Settings = Depends(get_settings),
    storage: StorageService = Depends(get_storage),
) -> PrintCheckResponse:
    """Dice si un STL se imprime en esa maquina, y que arreglarle.

    No hace falta ningun modelo: sirve para cualquier STL, venga de donde venga.
    """
    safe_name = sanitize_filename(Path(file.filename or "pieza.stl").name, default="pieza.stl")
    destino = settings_dep.uploads_path / f"{uuid4().hex}-{safe_name}"
    try:
        await storage.save_upload(file, destino, max_mb=settings_dep.max_upload_mb)
        reporte = await asyncio.to_thread(
            check_stl_for_printing,
            destino,
            printer=printer,
            target_axis=target_axis if isinstance(target_axis, str) and target_axis else None,
            target_mm=target_mm if isinstance(target_mm, (int, float)) else None,
        )
    except PrintCheckUnavailable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        # El archivo no queda: un STL de 50 MB por consulta llenaria el disco en
        # una tarde, y el veredicto ya no lo necesita.
        destino.unlink(missing_ok=True)

    return PrintCheckResponse(
        can_print=reporte.can_print,
        size_mm=reporte.size,
        triangle_count=reporte.mesh.triangle_count,
        watertight=reporte.mesh.is_watertight,
        manifold=reporte.mesh.is_manifold,
        volume_mm3=reporte.mesh.volume,
        overhang_ratio=reporte.overhang_ratio,
        blockers=reporte.blockers,
        advice=reporte.advice,
    )


# Los STL reparados/generados se sirven por token opaco, no por job: el token
# solo debe servirle a quien lo creo. El registro vive en memoria (igual que
# los jobs — un reinicio lo vacia) con un techo para que no crezca sin limite.
MAX_PRINT_TOKENS = 512


def _print_token_owners(request: Request) -> dict[str, str | None]:
    owners = getattr(request.app.state, "print_token_owners", None)
    if owners is None:
        owners = {}
        request.app.state.print_token_owners = owners
    return owners


def _register_print_token(request: Request, token: str) -> None:
    owners = _print_token_owners(request)
    while len(owners) >= MAX_PRINT_TOKENS:
        owners.pop(next(iter(owners)))
    owners[token] = _owner_id(request)


def _require_print_token_owner(
    request: Request, token: str, current_user: AuthenticatedUser, detail: str
) -> None:
    """Mismo 404 para "no existe" y "no es tuyo" — igual que con los jobs."""
    if Permission.jobs_read_all in current_user.permissions:
        return
    owners = _print_token_owners(request)
    if token not in owners or owners[token] != _owner_id(request):
        raise HTTPException(status_code=404, detail=detail)


@router.post("/print/repair", response_model=MeshRepairResponse)
async def repair_print_mesh(
    request: Request,
    file: UploadFile = File(...),
    settings_dep: Settings = Depends(get_settings),
    storage: StorageService = Depends(get_storage),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> MeshRepairResponse:
    """Tapa los agujeros de la malla y devuelve como quedo, medido.

    El archivo se entrega aunque NO haya quedado cerrada: el reporte dice la
    verdad y el usuario decide si le sirve.
    """
    safe_name = sanitize_filename(Path(file.filename or "pieza.stl").name, default="pieza.stl")
    origen = settings_dep.uploads_path / f"{uuid4().hex}-{safe_name}"
    token = uuid4().hex
    destino = settings_dep.outputs_path / f"{token}.repaired.stl"
    try:
        await storage.save_upload(file, origen, max_mb=settings_dep.max_upload_mb)
        triangulos = await asyncio.to_thread(read_stl, origen)
        reparada, reporte = await asyncio.to_thread(repair_mesh, triangulos)
        await asyncio.to_thread(write_stl, destino, reparada)
    except (StlUnreadable, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        origen.unlink(missing_ok=True)

    _register_print_token(request, token)
    return MeshRepairResponse(
        can_print=reporte.printable,
        watertight=reporte.is_watertight,
        manifold=reporte.is_manifold,
        triangle_count=reporte.triangle_count,
        volume_mm3=reporte.volume,
        blockers=reporte.problems,
        download_url=f"/api/v1/print/repaired/{token}",
    )


@router.get("/print/repaired/{token}")
async def download_repaired_mesh(
    token: str,
    request: Request,
    settings_dep: Settings = Depends(get_settings),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> FileResponse:
    # Solo hexadecimal: cualquier otra cosa podria salirse de la carpeta.
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise HTTPException(status_code=404, detail="Malla reparada no encontrada")
    _require_print_token_owner(request, token, current_user, "Malla reparada no encontrada")
    archivo = settings_dep.outputs_path / f"{token}.repaired.stl"
    if not archivo.exists():
        raise HTTPException(status_code=404, detail="Malla reparada no encontrada")
    return FileResponse(
        path=archivo, filename="pieza-reparada.stl", media_type="model/stl"
    )


@router.get("/print/parts", response_model=PartKindsResponse)
async def list_part_kinds() -> PartKindsResponse:
    return PartKindsResponse(
        kinds=[
            PartKindResponse(
                id=kind.id,
                label_key=kind.label_key,
                description_key=kind.description_key,
                params=[
                    PartParamResponse(
                        name=p.name, label_key=p.label_key, default=p.default, minimum=p.minimum
                    )
                    for p in kind.params
                ],
            )
            for kind in PART_KINDS
        ]
    )


@router.post("/print/parts", response_model=GeneratedPartResponse)
async def generate_part(
    payload: GeneratePartRequest,
    request: Request,
    settings_dep: Settings = Depends(get_settings),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> GeneratedPartResponse:
    """Construye una pieza con cotas EXACTAS y la devuelve ya verificada.

    Lo que sale de aca pasa por el mismo banco que verifica un STL ajeno. Generar
    y verificar con la misma herramienta no probaria nada; que el verificador sea
    independiente del generador es lo que hace que la verificacion signifique algo.
    """
    try:
        triangulos = await asyncio.to_thread(build_part, payload.kind, payload.params)
    except PartError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = uuid4().hex
    destino = settings_dep.outputs_path / f"{token}.part.stl"
    await asyncio.to_thread(write_stl, destino, triangulos)

    try:
        reporte = await asyncio.to_thread(
            check_stl_for_printing, destino, printer=payload.printer
        )
    except PrintCheckUnavailable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _register_print_token(request, token)
    return GeneratedPartResponse(
        can_print=reporte.can_print,
        size_mm=reporte.size,
        volume_mm3=reporte.mesh.volume,
        triangle_count=reporte.mesh.triangle_count,
        overhang_ratio=reporte.overhang_ratio,
        blockers=reporte.blockers,
        advice=reporte.advice,
        download_url=f"/api/v1/print/parts/{token}",
    )


@router.get("/print/parts/{token}")
async def download_generated_part(
    token: str,
    request: Request,
    settings_dep: Settings = Depends(get_settings),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> FileResponse:
    if not re.fullmatch(r"[0-9a-f]{32}", token):
        raise HTTPException(status_code=404, detail="Pieza no encontrada")
    _require_print_token_owner(request, token, current_user, "Pieza no encontrada")
    archivo = settings_dep.outputs_path / f"{token}.part.stl"
    if not archivo.exists():
        raise HTTPException(status_code=404, detail="Pieza no encontrada")
    return FileResponse(path=archivo, filename="pieza.stl", media_type="model/stl")


def get_shape3d_jobs(request: Request) -> Shape3dJobManager:
    return request.app.state.shape3d_jobs


def _shape3d_job_for(
    job_id: str, jobs: Shape3dJobManager, request: Request | None
) -> Shape3dJob:
    """El trabajo, si existe Y es de quien lo pide.

    Mismo 404 para "no existe" y "no es tuyo": un 403 confirmaria que el trabajo
    existe, que es informacion de otro usuario.
    """
    job = jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    return job


def shape3d_job_to_response(job: Shape3dJob) -> Shape3dJobResponse:
    return Shape3dJobResponse(
        id=job.id,
        status=job.status,
        prompt=job.prompt,
        printer=job.printer,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        can_print=job.can_print,
        size_mm=job.size_mm,
        triangle_count=job.triangle_count,
        blockers=job.blockers,
        advice=job.advice,
        error=job.error,
        source=job.source,
        target_mm=job.target_mm,
        target_mm_source=job.target_mm_source,
        target_mm_reference=job.target_mm_reference,
        code=job.code,
        retries=job.retries,
        download_url=(
            f"/api/v1/print/generate/{job.id}/download"
            if job.status == JobStatus.completed and job.output_path
            else None
        ),
        owner_id=job.owner_id,
    )


@router.post(
    "/print/estimate-size", response_model=SizeEstimateResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def estimate_print_size(
    payload: SizeEstimateRequest,
    jobs: Shape3dJobManager = Depends(get_shape3d_jobs),
) -> SizeEstimateResponse:
    """Cuanto mide de verdad el objeto descrito. Es una SUGERENCIA, no una cota.

    No escala nada ni encola nada: devuelve un numero para que el usuario lo
    confirme o lo cambie. Sin servidor de modelo configurado no hay sugerencia y
    el carril de malla sigue con su default, igual que antes de que esto
    existiera.
    """
    if not payload.prompt.strip():
        raise HTTPException(status_code=400, detail="Hace falta una descripcion del objeto.")
    if jobs.cad_client is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No hay servidor de modelo configurado para estimar el tamano. "
                "Levanta uno local (Ollama, LM Studio o llama.cpp server) y "
                "apuntalo desde Ajustes."
            ),
        )
    try:
        # `to_thread` porque el cliente habla HTTP con urllib, que es bloqueante:
        # sin esto, una estimacion lenta congela el loop entero.
        estimacion = await asyncio.to_thread(
            estimate_longest_mm, payload.prompt, client=jobs.cad_client
        )
    except SizeEstimateUnavailable as exc:
        # 502 y no 500: el que no respondio lo que se esperaba es el servidor del
        # modelo, que esta rio arriba de esta app.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return SizeEstimateResponse(
        longest_mm=estimacion.longest_mm, reference=estimacion.reference
    )


@router.post(
    "/print/generate", response_model=Shape3dJobResponse, status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_shape3d_job(
    payload: Shape3dJobRequest,
    request: Request,
    jobs: Shape3dJobManager = Depends(get_shape3d_jobs),
    settings: Settings = Depends(get_settings),
) -> Shape3dJobResponse:
    """Encola una malla desde texto o desde una foto. Tarda unos dos minutos.

    Lo que devuelve al terminar NO es solo el archivo: viaja con el veredicto del
    banco, porque una malla generada que no cierra no es una pieza.
    """
    # El token se resuelve ANTES de encolar: un token vencido tiene que ser un
    # 400 inmediato, no un job que muere dos minutos despues.
    image_path = _resolve_init_image(settings, payload.image_token)
    try:
        job = await jobs.create_job(
            prompt=payload.prompt,
            printer=payload.printer,
            source=payload.source,
            target_mm=payload.target_mm,
            target_mm_source=payload.target_mm_source,
            target_mm_reference=payload.target_mm_reference,
            expected_size=payload.expected_size,
            image_path=image_path,
            owner=current_user_from_request(request),
        )
    except Shape3dUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return shape3d_job_to_response(job)


@router.get("/print/generate", response_model=Shape3dJobsListResponse)
async def list_shape3d_jobs(
    all_users: bool = Query(default=False, alias="all"),
    jobs: Shape3dJobManager = Depends(get_shape3d_jobs),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> Shape3dJobsListResponse:
    """Las piezas generadas por el usuario.

    Generar una malla son minutos: sin listado, recargar el navegador dejaba el
    trabajo corriendo en el servidor y sin forma de volver a el.
    """
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return Shape3dJobsListResponse(jobs=[shape3d_job_to_response(job) for job in visible])


@router.get(
    "/print/generate/{job_id}", response_model=Shape3dJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_shape3d_job(
    job_id: str,
    jobs: Shape3dJobManager = Depends(get_shape3d_jobs),
    request: Request = None,
) -> Shape3dJobResponse:
    return shape3d_job_to_response(_shape3d_job_for(job_id, jobs, request))


@router.post(
    "/print/generate/{job_id}/cancel", response_model=Shape3dJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_shape3d_job(
    job_id: str,
    jobs: Shape3dJobManager = Depends(get_shape3d_jobs),
    request: Request = None,
) -> Shape3dJobResponse:
    job = _shape3d_job_for(job_id, jobs, request)
    jobs.cancel_job(job_id)
    return shape3d_job_to_response(job)


@router.get(
    "/print/generate/{job_id}/download",
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def download_shape3d_job(
    job_id: str,
    jobs: Shape3dJobManager = Depends(get_shape3d_jobs),
    request: Request = None,
) -> FileResponse:
    job = _shape3d_job_for(job_id, jobs, request)
    if job.output_path is None or not job.output_path.exists():
        raise HTTPException(status_code=404, detail="Malla no encontrada")
    return FileResponse(
        path=job.output_path, filename="pieza-generada.stl", media_type="model/stl"
    )


@router.patch(
    "/settings",
    response_model=UpdateSettingResponse,
    dependencies=[Depends(require(Permission.settings_write))],
)
async def patch_setting(payload: UpdateSettingRequest) -> UpdateSettingResponse:
    try:
        await asyncio.to_thread(update_setting, payload.key, payload.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UpdateSettingResponse(key=payload.key)
